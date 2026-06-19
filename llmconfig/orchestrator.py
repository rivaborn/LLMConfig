"""Coordinator over one or more GPU lanes.

Each `Lane` (see `lane.py`) owns the arbitration for a single card; the Orchestrator
just builds the lanes from `settings.lanes()`, routes load/unload to the right lane,
and aggregates per-lane status. The WSL keepalive is shared (one distro hosts every
lane's vLLM relay).
"""
from __future__ import annotations

import asyncio
from typing import Optional

from .backends.ollama import OllamaBackend
from .backends.vllm import VllmBackend
from .config import Settings
from .jobs import JobManager
from .gpu import GpuInfo, query_all_gpus
from .lane import Lane
from .lane_state import LaneDefaults
from .registry import DEFAULT_COMPANION_REGISTRY, Registry
from .schemas import Job, LaneStatus, LoadRequest, StatusResponse, UnloadRequest
from .wsl import WslKeepalive


class Orchestrator:
    def __init__(self, settings: Settings, registry: Registry, jobs: JobManager):
        self.s = settings
        self.jobs = jobs
        self.keepalive = WslKeepalive(settings)
        self.defaults = LaneDefaults(settings)
        self.lanes: dict[str, Lane] = {}
        for cfg in settings.lanes():
            # The primary lane reuses the registry the app already loaded; the companion
            # lane loads its own (small, 8 GB-friendly) catalog from its configured path.
            if cfg.id == "primary":
                reg = registry
            else:
                reg = Registry(cfg.registry_path, default_path=DEFAULT_COMPANION_REGISTRY)
            self.lanes[cfg.id] = Lane(settings, cfg, reg, jobs, self.keepalive)

    # ---- lane access / back-compat shims ----
    def lane(self, lane_id: str) -> Lane:
        lane = self.lanes.get(lane_id)
        if lane is None:
            raise KeyError(f"unknown lane '{lane_id}' (have: {', '.join(self.lanes)})")
        return lane

    @property
    def primary(self) -> Lane:
        return self.lanes["primary"]

    @property
    def ollama(self) -> OllamaBackend:
        return self.primary.ollama

    @property
    def vllm(self) -> VllmBackend:
        return self.primary.vllm

    # ---- status (aggregate) ----
    async def status(self) -> StatusResponse:
        gpus = await query_all_gpus(self.s)  # one nvidia-smi for every lane's card

        async def _lane_status(lane: Lane) -> LaneStatus:
            gpu = gpus.get(lane.cfg.gpu_uuid) or GpuInfo(
                found=False, uuid=lane.cfg.gpu_uuid, error=f"GPU {lane.cfg.gpu_uuid} not present"
            )
            return await lane.status(gpu=gpu)

        lane_statuses: list[LaneStatus] = list(
            await asyncio.gather(*(_lane_status(lane) for lane in self.lanes.values()))
        )
        primary = next((s for s in lane_statuses if s.id == "primary"), lane_statuses[0])
        return StatusResponse(
            owner=primary.owner,
            ollama_up=primary.ollama_up,
            vllm_up=primary.vllm_up,
            loaded=primary.loaded,
            gpu=primary.gpu,
            swap_in_progress=primary.swap_in_progress,
            active_job_id=primary.active_job_id,
            lanes=lane_statuses,
        )

    # ---- load / unload (routed to a lane) ----
    def load(self, req: LoadRequest) -> Job:
        return self.lane(req.lane).load(req)

    async def unload(self, req: UnloadRequest) -> StatusResponse:
        await self.lane(req.lane).unload(req)
        return await self.status()

    # ---- per-lane defaults ("what runs on this card") ----
    def default_for(self, lane_id: str) -> Optional[dict]:
        """Persisted override, else the static config seed (companion_default_*)."""
        d = self.defaults.get(lane_id)
        if d:
            return d
        for cfg in self.s.lanes():
            if cfg.id == lane_id and cfg.default_model and cfg.default_server in ("ollama", "vllm"):
                return {"server": cfg.default_server, "model": cfg.default_model}
        return None

    def autoload_defaults(self) -> list[Job]:
        """Fire (don't await) a load Job for every enabled lane that has a default."""
        jobs: list[Job] = []
        for cfg in self.s.lanes():
            if not cfg.enabled:
                continue
            d = self.default_for(cfg.id)
            if not d or d["server"] not in ("ollama", "vllm") or not d["model"]:
                continue
            req = LoadRequest(server=d["server"], model=d["model"], lane=cfg.id)
            jobs.append(self.lane(cfg.id).load(req))
        return jobs

    async def aclose(self) -> None:
        """Close every lane's pooled HTTP clients (call on app shutdown)."""
        for lane in self.lanes.values():
            await lane.ollama.aclose()
            await lane.vllm.aclose()
