"""Coordinator over every LLM unit.

Two kinds of unit exist and the orchestrator is what makes them interchangeable:

* a **`Lane`** (`lane.py`) — one local card arbitrated Ollama-XOR-vLLM, with the
  eviction-wait gate. The RTX 3090 (`primary`) and RTX 3070 Ti (`companion`).
* a **`SparkUnit`** (`spark_unit.py`) — one remote DGX Spark node driven by
  `sparkrun`. The node is the unit; no local GPU, no keepalive.

Both satisfy the same duck-typed contract (see `UNIT_METHODS`), so routing,
status aggregation, defaults, and autoload are written once against `self.units`.
`self.lanes` still exposes only the GPU lanes, because the WSL keepalive and the
Ollama/vLLM back-compat shims are meaningless for a remote node.
"""
from __future__ import annotations

import asyncio
from typing import Optional, Union

from .backends.ollama import OllamaBackend
from .backends.vllm import VllmBackend
from .config import Settings, SparkConfig
from .jobs import JobManager
from .gpu import GpuInfo, query_all_gpus
from .lane import Lane
from .lane_state import LaneDefaults
from .registry import DEFAULT_COMPANION_REGISTRY, Registry, SparkRegistry
from .schemas import Job, LaneStatus, LoadRequest, StatusResponse, UnloadRequest
from .spark_unit import SparkUnit
from .wsl import WslKeepalive

Unit = Union[Lane, SparkUnit]

# The duck-typed surface every unit must provide. Kept here as documentation and
# as the assertion target in tests — there is no ABC, because `Lane` predates the
# split and shares none of `SparkUnit`'s internals.
UNIT_METHODS = ("status", "load", "unload", "touch", "aclose")


class Orchestrator:
    def __init__(self, settings: Settings, registry: Registry, jobs: JobManager):
        self.s = settings
        self.jobs = jobs
        self.keepalive = WslKeepalive(settings)
        self.defaults = LaneDefaults(settings)
        self.units: dict[str, Unit] = {}

        for cfg in settings.lanes():
            # The primary lane reuses the registry the app already loaded; the companion
            # lane loads its own (small, 8 GB-friendly) catalog from its configured path.
            if cfg.id == "primary":
                reg = registry
            else:
                reg = Registry(cfg.registry_path, default_path=DEFAULT_COMPANION_REGISTRY)
            self.units[cfg.id] = Lane(settings, cfg, reg, jobs, self.keepalive)

        for scfg in settings.sparks():
            self.units[scfg.id] = SparkUnit(settings, scfg, SparkRegistry(scfg.registry_path), jobs)

    # ---- unit access ----
    def unit(self, unit_id: str) -> Unit:
        u = self.units.get(unit_id)
        if u is None:
            raise KeyError(f"unknown unit '{unit_id}' (have: {', '.join(self.units)})")
        return u

    # `lane()` is the historical name and still the one the REST layer uses.
    lane = unit

    @property
    def lanes(self) -> dict[str, Lane]:
        """Only the local GPU lanes — what the keepalive and idle-reaper vLLM logic
        legitimately care about. Use `units` for anything unit-kind-agnostic."""
        return {k: v for k, v in self.units.items() if isinstance(v, Lane)}

    @property
    def sparks(self) -> dict[str, SparkUnit]:
        return {k: v for k, v in self.units.items() if isinstance(v, SparkUnit)}

    @property
    def primary(self) -> Lane:
        return self.units["primary"]  # type: ignore[return-value]

    @property
    def ollama(self) -> OllamaBackend:
        return self.primary.ollama

    @property
    def vllm(self) -> VllmBackend:
        return self.primary.vllm

    # ---- status (aggregate) ----
    async def status(self) -> StatusResponse:
        # One nvidia-smi for every LOCAL card. Sparks aren't in this namespace —
        # each fetches its own telemetry over SSH inside its status() call.
        gpus = await query_all_gpus(self.s)

        async def _unit_status(u: Unit) -> LaneStatus:
            if isinstance(u, Lane):
                gpu = gpus.get(u.cfg.gpu_uuid) or GpuInfo(
                    found=False, uuid=u.cfg.gpu_uuid, error=f"GPU {u.cfg.gpu_uuid} not present"
                )
                return await u.status(gpu=gpu)
            return await u.status()

        statuses: list[LaneStatus] = list(
            await asyncio.gather(*(_unit_status(u) for u in self.units.values()))
        )
        primary = next((s for s in statuses if s.id == "primary"), statuses[0])
        return StatusResponse(
            owner=primary.owner,
            ollama_up=primary.ollama_up,
            vllm_up=primary.vllm_up,
            loaded=primary.loaded,
            gpu=primary.gpu,
            swap_in_progress=primary.swap_in_progress,
            active_job_id=primary.active_job_id,
            lanes=statuses,
        )

    # ---- load / unload (routed to a unit) ----
    def load(self, req: LoadRequest) -> Job:
        return self.unit(req.lane).load(req)

    async def unload(self, req: UnloadRequest) -> StatusResponse:
        await self.unit(req.lane).unload(req)
        return await self.status()

    # ---- per-unit defaults ("what runs on this unit") ----
    def defaults_for(self, unit_id: str) -> list[dict]:
        """Persisted overrides, else the static config seed (a list — a Spark can
        hold several models, a GPU lane exactly one).

        An EXPLICITLY empty persisted list is honoured as "load nothing" — the
        cookbook's tombstone — and does NOT fall through to the .env seed;
        only a unit with no persisted entry at all takes the seed.
        """
        persisted = self.defaults.entries_or_none(unit_id)
        if persisted is not None:
            return persisted
        for cfg in self.s.units():
            if cfg.id != unit_id or not cfg.default_model:
                continue
            if isinstance(cfg, SparkConfig):
                return [{"server": "spark", "model": cfg.default_model}]
            if cfg.default_server in ("ollama", "vllm"):
                return [{"server": cfg.default_server, "model": cfg.default_model}]
        return []

    def default_for(self, unit_id: str) -> Optional[dict]:
        """The unit's first default — the back-compat scalar view of `defaults_for`."""
        d = self.defaults_for(unit_id)
        return d[0] if d else None

    def autoload_defaults(self) -> list[Job]:
        """Fire (don't await) a load Job for every default on every enabled unit.

        One job PER MODEL: the unit's own lock serialises them, so a Spark asked for
        an embedder and a reranker loads them back-to-back rather than dropping all
        but the last.
        """
        jobs: list[Job] = []
        for cfg in self.s.units():
            if not cfg.enabled:
                continue
            for d in self.defaults_for(cfg.id):
                if d["server"] not in ("ollama", "vllm", "spark") or not d["model"]:
                    continue
                req = LoadRequest(server=d["server"], model=d["model"], lane=cfg.id)
                jobs.append(self.unit(cfg.id).load(req))
        return jobs

    def attach_load_times(self, load_times) -> None:
        """Hand every unit the LoadTimes recorder (same fanout as attach_leases)."""
        for u in self.units.values():
            u.load_times = load_times

    def attach_leases(self, leases) -> None:
        """Hand every Spark unit the LeaseManager for under-lock victim checks.

        Called from create_app AFTER the LeaseManager exists (it is constructed
        with the orchestrator, so the reference cannot be a constructor arg).
        """
        for u in self.units.values():
            if hasattr(u, "declared_budgets"):     # duck-typed: only Sparks
                u.leases = leases

    async def aclose(self) -> None:
        """Close every unit's pooled HTTP clients (call on app shutdown)."""
        for u in self.units.values():
            await u.aclose()
