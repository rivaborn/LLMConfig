"""FastAPI application: REST API + the static web UI.

Read endpoints are open (LAN perimeter); write endpoints (load/unload/pull/alias
edits/download) require `X-API-Key` only when LLMCONFIG_API_KEY is set.
"""
from __future__ import annotations

import shlex
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from . import doctor as doctor_mod
from .config import PACKAGE_DIR, get_settings
from .gpu import query_gpu
from .idle import IdleReaper, classify_usage
from .jobs import JobManager
from .leases import LeaseConflict, LeaseError, LeaseManager, LeaseNotActive, LeaseSweeper, UnknownUnit
from .monitor import Monitor
from .openai_gateway import OpenAIGateway, build_gateway_router
from .orchestrator import Orchestrator
from .registry import make_registry
from .schemas import (
    GpuOut,
    Job,
    LaneUsageOut,
    Lease,
    LeaseBrief,
    LeaseClaimRequest,
    LeaseClaimResponse,
    LeaseListResponse,
    LeaseRenewRequest,
    LeaseRevokeRequest,
    LoadRequest,
    ModelsResponse,
    SparkModelEntry,
    StatusResponse,
    UnloadRequest,
    UsageResponse,
    VllmAliasEntry,
)
from .spark_unit import SparkUnit
from .wsl import run_wsl

WEB_DIR = PACKAGE_DIR / "web"


def create_app() -> FastAPI:
    settings = get_settings()
    registry = make_registry(settings)
    jobs = JobManager()
    orch = Orchestrator(settings, registry, jobs)
    leases = LeaseManager(settings, orch)
    gateway = OpenAIGateway(orch, jobs, settings, leases)
    monitor = Monitor(settings, orch)
    reaper = IdleReaper(settings, orch, monitor, leases)
    sweeper = LeaseSweeper(settings, orch, leases)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Auto-load each lane's configured default model (fire-and-forget Jobs).
        orch.autoload_defaults()
        monitor.start()  # begin sampling GPU/LLM telemetry for the Monitor tab
        reaper.start()   # idle auto-unload policy (reads the monitor's util samples)
        sweeper.start()  # lease expiry + deferred preemption frees
        yield
        await sweeper.stop()
        await reaper.stop()  # before the monitor: the reaper reads its samples
        await monitor.stop()
        # Release the WSL keepalive so the distro can idle-shut-down cleanly when
        # the control app stops (an already-loaded vLLM model goes with it).
        orch.keepalive.stop()
        await gateway.aclose()  # close the /v1 forwarding client
        await orch.aclose()  # close pooled HTTP clients

    app = FastAPI(title="LLMConfig", version=__version__, lifespan=lifespan,
                  description="GPU-arbitrated control plane for Ollama + vLLM.")
    app.state.settings = settings
    app.state.registry = registry
    app.state.jobs = jobs
    app.state.orch = orch
    app.state.gateway = gateway
    app.state.monitor = monitor
    app.state.reaper = reaper
    app.state.leases = leases
    app.state.sweeper = sweeper

    async def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
        if settings.auth_enabled and x_api_key != settings.llmconfig_api_key:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    write = [Depends(require_key)]

    def _lane(lane_id: str):
        try:
            return orch.lane(lane_id)
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e))

    def _current_util(uuid: str) -> Optional[float]:
        """The Monitor's latest utilization sample for one GPU (None when unmonitored)."""
        for g in monitor.snapshot().get("gpus", []):
            if g.get("uuid") == uuid:
                return g.get("util_pct")
        return None

    async def _status_with_usage() -> StatusResponse:
        resp = await orch.status()
        for ls in resp.lanes:
            ls.usage = classify_usage(ls, _current_util(orch.lane(ls.id).cfg.gpu_uuid), settings)
            # Additive — `usage` keeps its three values because off-box consumers
            # switch on it; the lease is reported alongside, never as a 4th state.
            ls.lease = leases.brief(ls.id)  # sync, no await
        return resp

    if (WEB_DIR / "static").is_dir():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

    # ------------------------------------------------------------------ #
    # UI + read endpoints (open)
    # ------------------------------------------------------------------ #
    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        idx = WEB_DIR / "templates" / "index.html"
        if not idx.is_file():
            return HTMLResponse("<h1>LLMConfig</h1><p>UI not installed; use the REST API at /docs.</p>")
        html = idx.read_text(encoding="utf-8")
        # Cache-bust the static assets: StaticFiles sends no Cache-Control, so
        # browsers heuristically serve a stale style.css/app.js after a redeploy
        # (the tab rules would be missing → views stack). Tag each asset URL with
        # the newest static-file mtime so a changed file always fetches fresh.
        try:
            static_files = list((WEB_DIR / "static").glob("*.*"))
            token = str(int(max(p.stat().st_mtime for p in static_files)))
        except (ValueError, OSError):  # missing static dir / empty — degrade, never 500
            static_files, token = [], __version__
        # Derived from what's actually on disk rather than a hardcoded list, so a
        # newly added asset can't silently ship un-busted.
        for asset in sorted(p.name for p in static_files):
            html = html.replace(f"/static/{asset}\"", f"/static/{asset}?v={token}\"")
        # Always revalidate the HTML itself so new tokens are picked up.
        return HTMLResponse(html, headers={"Cache-Control": "no-cache"})

    @app.get("/api/status", response_model=StatusResponse)
    async def api_status() -> StatusResponse:
        return await _status_with_usage()

    @app.get("/api/usage", response_model=UsageResponse)
    async def api_usage(lane: str = "primary") -> UsageResponse:
        """Compact per-lane tri-state: free (nothing loaded) / idle (model loaded,
        unused) / active (model loaded and in use). Top level mirrors `?lane=`."""
        _lane(lane)  # validate the lane id (400 on unknown)
        resp = await _status_with_usage()
        lanes = [
            LaneUsageOut(
                lane=ls.id,
                state=ls.usage or "free",
                model=ls.loaded.model if ls.loaded else None,
                models=[m.model for m in ls.loaded_models],
                idle_s=ls.idle_s,
                lease=ls.lease,
            )
            for ls in resp.lanes
        ]
        mirror = next((u for u in lanes if u.lane == lane), lanes[0])
        return UsageResponse(
            lane=mirror.lane, state=mirror.state, model=mirror.model,
            models=mirror.models, idle_s=mirror.idle_s, lease=mirror.lease, lanes=lanes,
        )

    @app.get("/api/models", response_model=ModelsResponse)
    async def api_models(lane: str = "primary") -> ModelsResponse:
        ln = _lane(lane)
        resp = ModelsResponse()
        if isinstance(ln, SparkUnit):
            try:
                resp.spark = await ln.spark.list_models()
            except Exception as e:
                resp.spark_error = f"{type(e).__name__}: {e}"
            return resp
        try:
            resp.ollama = await ln.ollama.list_models()
        except Exception as e:
            resp.ollama_error = f"{type(e).__name__}: {e}"
        try:
            resp.vllm = await ln.vllm.list_aliases()
        except Exception as e:
            resp.vllm_error = f"{type(e).__name__}: {e}"
        return resp

    @app.get("/api/gpu", response_model=GpuOut)
    async def api_gpu(lane: str = "primary") -> GpuOut:
        ln = _lane(lane)
        if isinstance(ln, SparkUnit):  # remote node — its own nvidia-smi over SSH
            return GpuOut.from_info(await ln.spark.gpu())
        return GpuOut.from_info(await query_gpu(settings, uuid=ln.cfg.gpu_uuid))

    @app.get("/api/lanes")
    async def api_lanes() -> list[dict]:
        """Every LLM unit, in display order — what the UI turns into tabs/cards."""
        return [
            {
                "id": cfg.id,
                "name": cfg.name,
                "kind": getattr(cfg, "kind", "gpu"),
                "host": getattr(cfg, "host", ""),
                "enabled": cfg.enabled,
                # `default` is the first entry, kept for existing clients; `defaults`
                # is the real answer on a unit that can hold several models.
                "default": orch.default_for(cfg.id),
                "defaults": orch.defaults_for(cfg.id),
                "max_models": getattr(cfg, "max_models", 1),
            }
            for cfg in settings.units()
        ]

    @app.get("/api/jobs", response_model=list[Job])
    async def api_jobs() -> list[Job]:
        return jobs.list()

    @app.get("/api/jobs/{jid}", response_model=Job)
    async def api_job(jid: str) -> Job:
        job = jobs.get(jid)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.get("/api/doctor")
    async def api_doctor() -> dict:
        report = await doctor_mod.run_doctor(settings, registry)
        return report.model_dump()

    @app.get("/api/monitor")
    async def api_monitor() -> dict:
        """Latest GPU thermals/power/VRAM + Ollama split (the Monitor tab readouts)."""
        return monitor.snapshot()

    @app.get("/api/monitor/history")
    async def api_monitor_history(window: float = 3600.0) -> dict:
        """Bucketed telemetry history over the last `window` seconds."""
        return monitor.history(window)

    @app.get("/api/vllm/aliases", response_model=list[VllmAliasEntry])
    async def api_aliases(lane: str = "primary") -> list[VllmAliasEntry]:
        return _lane(lane).registry.entries()

    @app.get("/api/lanes/{lane_id}/default")
    async def api_lane_default(lane_id: str) -> dict:
        _lane(lane_id)  # validate
        return {"lane": lane_id, "default": orch.default_for(lane_id)}

    # ------------------------------------------------------------------ #
    # Write endpoints (X-API-Key when configured)
    # ------------------------------------------------------------------ #
    def _require_lease_ok(unit_id: str, x_llm_lease: Optional[str],
                          model: str = "") -> None:
        """Honour a non-preemptible lease on the load/unload endpoints too.

        Without this the /v1 gate is hollow: chat traffic gets a 409 while anyone
        can POST /api/load a different model over the held unit — the biggest
        interruption there is. The holder passes by sending its own lease id as
        X-LLM-Lease; preemptible leases stay advisory here, like on /v1.

        Scoped to `model` so a per-model claim on a multi-model Spark refuses only
        the model it names; a claim with no model still covers the whole node.
        """
        if not settings.lease_enabled:
            return
        blocker = leases.blocks_unleased(unit_id, model or None)
        if blocker is None or (x_llm_lease or "").strip() == blocker.id:
            return
        held = f"model '{blocker.model}' on {unit_id}" if blocker.model else f"unit '{unit_id}'"
        brief = leases.brief(unit_id)
        raise HTTPException(status_code=409, detail={
            "error": "lease_held",
            "message": f"{held} is held by '{blocker.holder}' with a "
                       f"non-preemptible lease — pass its id as X-LLM-Lease, or "
                       f"revoke it (POST /api/leases/{blocker.id}/revoke)",
            "unit": unit_id,
            "model": blocker.model or None,
            "lease": brief.model_dump() if brief else None,
        })

    @app.post("/api/load", response_model=Job, dependencies=write)
    async def api_load(req: LoadRequest,
                       x_llm_lease: Optional[str] = Header(default=None)) -> Job:
        unit = _lane(req.lane)
        # Catch the kind mismatch here — otherwise server="spark" on a GPU lane
        # falls into the vLLM path and fails with a misleading "unknown alias".
        if isinstance(unit, SparkUnit) != (req.server == "spark"):
            want = "'spark'" if isinstance(unit, SparkUnit) else "'ollama' or 'vllm'"
            raise HTTPException(status_code=400,
                                detail=f"unit '{req.lane}' takes server {want}, not '{req.server}'")
        _require_lease_ok(req.lane, x_llm_lease, req.model)
        return orch.load(req)

    @app.post("/api/unload", response_model=StatusResponse, dependencies=write)
    async def api_unload(req: UnloadRequest,
                         x_llm_lease: Optional[str] = Header(default=None)) -> StatusResponse:
        _lane(req.lane)
        # `req.model` frees one model and leaves co-residents running; without it the
        # whole unit is freed, which is what the UI's unit-level Unload still means.
        _require_lease_ok(req.lane, x_llm_lease, req.model or "")
        return await orch.unload(req)

    @app.post("/api/ollama/pull", response_model=Job, dependencies=write)
    async def api_pull(body: dict) -> Job:
        name = body.get("model") or body.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="missing 'model'")
        job = jobs.create(kind=f"pull:{name}")

        async def run(job: Job) -> dict:
            def on_evt(evt: dict) -> None:
                status = evt.get("status", "")
                total, completed = evt.get("total"), evt.get("completed")
                if total and completed:
                    job.progress = round(completed / total, 3)
                    jobs.log(job, f"{status} {int(100 * completed / total)}%")
                else:
                    jobs.log(job, status)

            await orch.ollama.pull(name, on_event=on_evt)
            return {"model": name}

        return jobs.start(job, run)

    @app.delete("/api/ollama/{name:path}", dependencies=write)
    async def api_delete(name: str) -> dict:
        await orch.ollama.delete(name)
        return {"deleted": name}

    @app.post("/api/vllm/aliases", response_model=VllmAliasEntry, dependencies=write)
    async def api_alias_create(entry: VllmAliasEntry, lane: str = "primary") -> VllmAliasEntry:
        _lane(lane).registry.upsert(entry)
        return entry

    @app.put("/api/vllm/aliases/{alias}", response_model=VllmAliasEntry, dependencies=write)
    async def api_alias_upsert(alias: str, entry: VllmAliasEntry, lane: str = "primary") -> VllmAliasEntry:
        entry.alias = alias
        _lane(lane).registry.upsert(entry)
        return entry

    @app.delete("/api/vllm/aliases/{alias}", dependencies=write)
    async def api_alias_delete(alias: str, lane: str = "primary") -> dict:
        if not _lane(lane).registry.remove(alias):
            raise HTTPException(status_code=404, detail="alias not found")
        return {"deleted": alias}

    @app.put("/api/lanes/{lane_id}/default", dependencies=write)
    async def api_lane_default_set(lane_id: str, body: dict) -> dict:
        """Set, add to, or clear a unit's startup defaults.

        `mode` picks between the two meanings a multi-model unit needs: `replace`
        (the default, and the only sane one for a GPU lane) makes this the unit's
        sole default; `add`/`remove` co-schedule on a Spark.
        """
        _lane(lane_id)  # validate
        server = (body.get("server") or "").strip()
        model = (body.get("model") or "").strip()
        mode = (body.get("mode") or "replace").strip()
        if mode not in ("replace", "add", "remove"):
            raise HTTPException(status_code=400, detail="mode must be 'replace', 'add' or 'remove'")
        if mode == "remove":
            if not model:
                raise HTTPException(status_code=400, detail="'remove' needs a model")
            orch.defaults.remove(lane_id, model)
        elif not model:
            orch.defaults.clear(lane_id)
        elif server not in ("ollama", "vllm", "spark"):
            raise HTTPException(status_code=400, detail="server must be 'ollama', 'vllm' or 'spark'")
        elif mode == "add":
            orch.defaults.add(lane_id, server, model)
        else:
            orch.defaults.set(lane_id, server, model)
        return {"lane": lane_id, "default": orch.default_for(lane_id),
                "defaults": orch.defaults_for(lane_id)}

    # ---- curated Spark model catalog (mirrors the vLLM alias endpoints) ----
    def _spark(lane_id: str) -> SparkUnit:
        unit = _lane(lane_id)
        if not isinstance(unit, SparkUnit):
            raise HTTPException(status_code=400, detail=f"unit '{lane_id}' is not a DGX Spark")
        return unit

    @app.get("/api/spark/models", response_model=list[SparkModelEntry])
    async def api_spark_models(lane: str) -> list[SparkModelEntry]:
        return _spark(lane).registry.entries()

    @app.put("/api/spark/models/{alias}", response_model=SparkModelEntry, dependencies=write)
    async def api_spark_model_upsert(alias: str, entry: SparkModelEntry, lane: str) -> SparkModelEntry:
        entry.alias = alias
        _spark(lane).registry.upsert(entry)
        return entry

    @app.delete("/api/spark/models/{alias}", dependencies=write)
    async def api_spark_model_delete(alias: str, lane: str) -> dict:
        if not _spark(lane).registry.remove(alias):
            raise HTTPException(status_code=404, detail="model not found")
        return {"deleted": alias}

    @app.post("/api/vllm/download", response_model=Job, dependencies=write)
    async def api_vllm_download(body: dict) -> Job:
        repo = body.get("repo") or body.get("hf_repo")
        if not repo:
            raise HTTPException(status_code=400, detail="missing 'repo'")
        job = jobs.create(kind=f"download:{repo}")

        async def run(job: Job) -> dict:
            # `hf` lives in the WSL venv, not on the login PATH, so activate it first
            # (a bare `hf download` under `bash -lc` fails "command not found"). Shell-
            # quote the token + repo so an odd HF_TOKEN value can't break parsing — an
            # unquoted prefix previously produced a `bash: unexpected EOF matching '`.
            prefix = (
                f"source {shlex.quote(settings.vllm_venv_activate)} "
                "&& export HF_HUB_ENABLE_HF_TRANSFER=1 && "
            )
            if settings.hf_token:
                prefix += f"HF_TOKEN={shlex.quote(settings.hf_token)} "
            jobs.log(job, f"hf download {repo} (may take a long time)…")
            r = await run_wsl(
                f"{prefix}hf download {shlex.quote(repo)}",
                login=True, timeout=3 * 60 * 60, settings=settings,
            )
            if not r.ok:
                raise RuntimeError(r.text()[:2000] or "hf download failed")
            return {"repo": repo, "output": r.out[-500:]}

        return jobs.start(job, run)

    # ------------------------------------------------------------------ #
    # Leases — resource sharing between callers (see leases.py)
    #
    # ADVISORY: clients that bypass LLMConfig (direct Ollama on :11434/:11435, the
    # vLLM relay on :11437) are ungated, so a non-preemptible lease is a cooperation
    # contract, not a hard exclusivity guarantee.
    # ------------------------------------------------------------------ #
    def _lease_http(e: LeaseError) -> HTTPException:
        status = 400 if isinstance(e, UnknownUnit) else 409
        detail: dict = {"error": e.code, "message": e.message}
        if e.lease is not None:
            brief = leases.brief(e.lease.unit)
            detail["unit"] = e.lease.unit
            # HTTPException.detail is plain-json encoded, so hand it dicts.
            detail["lease"] = brief.model_dump() if brief else e.lease.model_dump()
        return HTTPException(status_code=status, detail=detail)

    @app.post("/api/leases", response_model=LeaseClaimResponse, status_code=201,
              dependencies=write)
    async def api_lease_claim(req: LeaseClaimRequest, response: Response) -> LeaseClaimResponse:
        try:
            lease, displaced = leases.claim(req)
        except LeaseError as e:
            raise _lease_http(e)
        if lease.renew_count:
            response.status_code = 200  # same holder re-claiming → extended in place

        # A non-preemptible lease is a FORWARD guarantee: work already running keeps
        # the unit until it finishes (there is no job cancellation). Report that so
        # the new holder can choose to wait rather than assume exclusivity now.
        busy = None
        try:
            # Only the claimed unit — a full _status_with_usage() would probe every
            # unit (local nvidia-smi + each Spark) just to inspect this one.
            st = await orch.unit(lease.unit).status()
            usage = classify_usage(st, _current_util(orch.unit(lease.unit).cfg.gpu_uuid), settings)
            if st.swap_in_progress or usage == "active":
                busy = {
                    "active_job_id": st.active_job_id,
                    "usage": usage,
                    "swap_in_progress": st.swap_in_progress,
                    # `loaded` stays the primary occupant for existing clients; the
                    # list is what a claimant on a multi-model node actually needs.
                    "loaded": st.loaded.model_dump() if st.loaded else None,
                    "loaded_models": [m.model_dump() for m in st.loaded_models],
                }
        except Exception:  # noqa: BLE001 — a status hiccup must not fail the claim
            pass
        return LeaseClaimResponse(
            lease=lease,
            displaced=(
                None if displaced is None else
                LeaseBrief(id=displaced.id, holder=displaced.holder,
                           preemptible=displaced.preemptible, priority=displaced.priority,
                           expires_at=displaced.expires_at, model=displaced.model)
            ),
            busy_with=busy,
        )

    @app.get("/api/leases", response_model=LeaseListResponse)
    async def api_leases(unit: Optional[str] = None, active: bool = False) -> LeaseListResponse:
        # `?unit=` (empty) means "all units", not "units named ''" — httpx clients
        # serialize params={"unit": None} as an empty string, not an absent param.
        return LeaseListResponse(
            leases=leases.list(unit=unit or None, active_only=active),
            server_started_at=leases.started_at,
        )

    @app.get("/api/leases/{lease_id}", response_model=Lease)
    async def api_lease(lease_id: str) -> Lease:
        lease = leases.get(lease_id)
        if lease is None:
            # Distinguish "never existed" from "lost in a restart" — leases are
            # in-memory, so a restart is a normal reason for a valid id to vanish.
            raise HTTPException(status_code=404, detail={
                "error": "lease_unknown",
                "message": f"no lease '{lease_id}'"
                           + (" (the server restarted; re-claim and retry)"
                              if leases.looks_like_pre_restart(lease_id) else ""),
                "server_started_at": leases.started_at,
            })
        return lease

    @app.post("/api/leases/{lease_id}/renew", response_model=Lease, dependencies=write)
    async def api_lease_renew(lease_id: str, req: LeaseRenewRequest) -> Lease:
        try:
            return leases.renew(lease_id, req.ttl_s)
        except UnknownUnit as e:
            raise HTTPException(status_code=404, detail={"error": "lease_unknown", "message": e.message})
        except LeaseNotActive as e:
            raise _lease_http(e)

    @app.delete("/api/leases/{lease_id}", response_model=Lease, dependencies=write)
    async def api_lease_release(lease_id: str) -> Lease:
        try:
            return leases.release(lease_id)  # idempotent
        except UnknownUnit as e:
            raise HTTPException(status_code=404, detail={"error": "lease_unknown", "message": e.message})

    @app.post("/api/leases/{lease_id}/revoke", response_model=Lease, dependencies=write)
    async def api_lease_revoke(lease_id: str, req: LeaseRevokeRequest) -> Lease:
        """Operator break-glass — takes a unit back even from a non-preemptible lease
        (which `force` on a claim deliberately cannot do)."""
        try:
            return leases.revoke(lease_id, by=req.by, reason=req.reason, free=req.free)
        except UnknownUnit as e:
            raise HTTPException(status_code=404, detail={"error": "lease_unknown", "message": e.message})

    # ------------------------------------------------------------------ #
    # OpenAI-compatible /v1 gateway (auto-loads on first request, then proxies).
    # opencode points each provider's baseURL here; the picked model triggers the
    # load. LAN inference path — open like the other read/proxy endpoints.
    # ------------------------------------------------------------------ #
    app.include_router(build_gateway_router(gateway))

    return app


app = create_app()
