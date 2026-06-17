"""FastAPI application: REST API + the static web UI.

Read endpoints are open (LAN perimeter); write endpoints (load/unload/pull/alias
edits/download) require `X-API-Key` only when LLMCONFIG_API_KEY is set.
"""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from . import doctor as doctor_mod
from .config import PACKAGE_DIR, get_settings
from .gpu import query_gpu
from .jobs import JobManager
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

    app = FastAPI(title="LLMConfig", version=__version__,
                  description="GPU-arbitrated control plane for Ollama + vLLM.")
    app.state.settings = settings
    app.state.registry = registry
    app.state.jobs = jobs
    app.state.orch = orch

    async def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
        if settings.auth_enabled and x_api_key != settings.llmconfig_api_key:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    write = [Depends(require_key)]

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
    async def api_models() -> ModelsResponse:
        resp = ModelsResponse()
        try:
            resp.ollama = await orch.ollama.list_models()
        except Exception as e:
            resp.ollama_error = f"{type(e).__name__}: {e}"
        try:
            resp.vllm = await orch.vllm.list_aliases()
        except Exception as e:
            resp.vllm_error = f"{type(e).__name__}: {e}"
        return resp

    @app.get("/api/gpu", response_model=GpuOut)
    async def api_gpu() -> GpuOut:
        return GpuOut.from_info(await query_gpu(settings))

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

    @app.get("/api/vllm/aliases", response_model=list[VllmAliasEntry])
    async def api_aliases() -> list[VllmAliasEntry]:
        return registry.entries()

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
    async def api_alias_create(entry: VllmAliasEntry) -> VllmAliasEntry:
        registry.upsert(entry)
        return entry

    @app.put("/api/vllm/aliases/{alias}", response_model=VllmAliasEntry, dependencies=write)
    async def api_alias_upsert(alias: str, entry: VllmAliasEntry) -> VllmAliasEntry:
        entry.alias = alias
        registry.upsert(entry)
        return entry

    @app.delete("/api/vllm/aliases/{alias}", dependencies=write)
    async def api_alias_delete(alias: str) -> dict:
        if not registry.remove(alias):
            raise HTTPException(status_code=404, detail="alias not found")
        return {"deleted": alias}

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

    return app


app = create_app()
