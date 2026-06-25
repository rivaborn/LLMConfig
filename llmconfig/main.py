"""FastAPI application: REST API + the static web UI.

Read endpoints are open (LAN perimeter); write endpoints (load/unload/pull/alias
edits/download) require `X-API-Key` only when LLMCONFIG_API_KEY is set.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from . import doctor as doctor_mod
from .config import PACKAGE_DIR, get_settings
from .gpu import query_gpu
from .jobs import JobManager
from .monitor import Monitor
from .openai_gateway import OpenAIGateway, build_gateway_router
from .orchestrator import Orchestrator
from .registry import make_registry
from .schemas import (
    GpuOut,
    Job,
    LoadRequest,
    ModelsResponse,
    StatusResponse,
    UnloadRequest,
    VllmAliasEntry,
)
from .wsl import run_wsl

WEB_DIR = PACKAGE_DIR / "web"


def create_app() -> FastAPI:
    settings = get_settings()
    registry = make_registry(settings)
    jobs = JobManager()
    orch = Orchestrator(settings, registry, jobs)
    gateway = OpenAIGateway(orch, jobs, settings)
    monitor = Monitor(settings, orch)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Auto-load each lane's configured default model (fire-and-forget Jobs).
        orch.autoload_defaults()
        monitor.start()  # begin sampling GPU/LLM telemetry for the Monitor tab
        yield
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

    async def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
        if settings.auth_enabled and x_api_key != settings.llmconfig_api_key:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    write = [Depends(require_key)]

    def _lane(lane_id: str):
        try:
            return orch.lane(lane_id)
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e))

    if (WEB_DIR / "static").is_dir():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

    # ------------------------------------------------------------------ #
    # UI + read endpoints (open)
    # ------------------------------------------------------------------ #
    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        idx = WEB_DIR / "templates" / "index.html"
        if idx.is_file():
            return HTMLResponse(idx.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>LLMConfig</h1><p>UI not installed; use the REST API at /docs.</p>")

    @app.get("/api/status", response_model=StatusResponse)
    async def api_status() -> StatusResponse:
        return await orch.status()

    @app.get("/api/models", response_model=ModelsResponse)
    async def api_models(lane: str = "primary") -> ModelsResponse:
        ln = _lane(lane)
        resp = ModelsResponse()
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
        return GpuOut.from_info(await query_gpu(settings, uuid=_lane(lane).cfg.gpu_uuid))

    @app.get("/api/lanes")
    async def api_lanes() -> list[dict]:
        return [
            {"id": cfg.id, "name": cfg.name, "enabled": cfg.enabled, "default": orch.default_for(cfg.id)}
            for cfg in settings.lanes()
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
    @app.post("/api/load", response_model=Job, dependencies=write)
    async def api_load(req: LoadRequest) -> Job:
        return orch.load(req)

    @app.post("/api/unload", response_model=StatusResponse, dependencies=write)
    async def api_unload(req: UnloadRequest) -> StatusResponse:
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
        _lane(lane_id)  # validate
        server = (body.get("server") or "").strip()
        model = (body.get("model") or "").strip()
        if not model:
            orch.defaults.clear(lane_id)
        elif server not in ("ollama", "vllm"):
            raise HTTPException(status_code=400, detail="server must be 'ollama' or 'vllm'")
        else:
            orch.defaults.set(lane_id, server, model)
        return {"lane": lane_id, "default": orch.default_for(lane_id)}

    @app.post("/api/vllm/download", response_model=Job, dependencies=write)
    async def api_vllm_download(body: dict) -> Job:
        repo = body.get("repo") or body.get("hf_repo")
        if not repo:
            raise HTTPException(status_code=400, detail="missing 'repo'")
        job = jobs.create(kind=f"download:{repo}")

        async def run(job: Job) -> dict:
            env = f"HF_TOKEN={settings.hf_token} " if settings.hf_token else ""
            jobs.log(job, f"hf download {repo} (may take a long time)…")
            r = await run_wsl(f"{env}hf download {repo}", login=True, timeout=3 * 60 * 60, settings=settings)
            if not r.ok:
                raise RuntimeError(r.text()[:2000] or "hf download failed")
            return {"repo": repo, "output": r.out[-500:]}

        return jobs.start(job, run)

    # ------------------------------------------------------------------ #
    # OpenAI-compatible /v1 gateway (auto-loads on first request, then proxies).
    # opencode points each provider's baseURL here; the picked model triggers the
    # load. LAN inference path — open like the other read/proxy endpoints.
    # ------------------------------------------------------------------ #
    app.include_router(build_gateway_router(gateway))

    return app


app = create_app()
