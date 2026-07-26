"""One DGX Spark node as an LLM unit — the remote sibling of `Lane`.

`Lane` arbitrates two servers over one local card and must prove the VRAM is
free before loading. A Spark node needs none of that: the node *is* the unit and
runs exactly one sparkrun workload, so a swap is simply stop → run → wait-ready.
There is no eviction-wait gate, no WSL keepalive (the workload lives on the node,
not in WSL), and no local nvidia-smi.

What it deliberately *does* share with `Lane` is the surface the rest of the app
consumes — `cfg` / `registry` / `touch()` / `status()` / `load()` / `unload()`,
one `asyncio.Lock` per unit, and the `Job` + streamed-log pattern — so the
orchestrator, the idle reaper, the REST layer, and the UI treat both kinds
through a single code path. See `UNIT_METHODS` in `orchestrator.py` for the
duck-typed contract.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

from .backends.spark import SparkBackend
from .config import Settings, SparkConfig
from .gpu import GpuInfo
from .jobs import JobManager
from .registry import SparkRegistry
from .schemas import (GpuOut, Job, LaneStatus, LoadedModel, LoadRequest, ServedModel,
                      UnloadRequest)


class SparkUnit:
    kind = "spark"

    def __init__(
        self,
        settings: Settings,
        cfg: SparkConfig,
        registry: SparkRegistry,
        jobs: JobManager,
    ):
        self.s = settings
        self.cfg = cfg
        self.registry = registry
        self.jobs = jobs
        self.spark = SparkBackend(settings, cfg, registry)
        self._lock = asyncio.Lock()
        self._active_job_id: Optional[str] = None
        self.last_activity: float = time.time()
        # Per-model activity clocks, keyed by catalog alias; see touch()/idle_for().
        self.model_activity: dict[str, float] = {}
        # Telemetry cache. `status()` is polled by the UI every ~2.5 s, but the
        # node's nvidia-smi is an SSH round-trip that costs seconds — and a
        # powered-off Spark costs the full connect timeout. So status() never
        # awaits SSH: it serves the last sample and kicks off a background
        # refresh when stale. Without this, one dead node would add seconds of
        # latency to every /api/status for every client.
        self._gpu_cached: Optional[GpuInfo] = None
        self._gpu_ts: float = 0.0
        self._gpu_task: Optional[asyncio.Task] = None
        self._gpu_ttl: float = 20.0
        # Circuit breaker for the HTTP probe. On the LAN it answers in ~1 ms, but
        # a node that is off (or unroutable) costs the full connect timeout on
        # every poll. After a few consecutive misses, back off so one dead Spark
        # can't drag out /api/status for every client.
        self._served_fails: int = 0
        self._served_ts: float = 0.0
        self._probe_backoff_s: float = 15.0
        self._fails_before_backoff: int = 3

    def canonical_model(self, model: str) -> str:
        """Fold any name for a model onto one key: its catalog alias.

        Callers name a model three ways — the gateway passes whatever the client
        asked for, a load passes the alias, and residency (so the idle reaper and
        the UI) reports the node's *served* name. Anything keyed by model has to
        agree on one of them or it silently splits in two: two activity clocks
        that never look idle, or a lease on `m1` that fails to shield `served-1`.

        Optional part of the unit contract — callers use
        `getattr(unit, "canonical_model", None)`, and a `Lane` (one model, named
        one way) simply doesn't need it.
        """
        entry = self.registry.get(model) or self.registry.find_by_served_name(model)
        return entry.alias if entry else model

    def touch(self, ts: float | None = None, model: str | None = None) -> None:
        """Record activity for the idle reaper. Never moves the clock backwards.

        Keeps a clock PER MODEL as well as for the unit. The unit clock is the max
        across models, so on a multi-model node it says only "something here is
        busy" — reaping off that alone would let one busy model keep every idle
        neighbour resident forever, which on a shared 128 GB pool is exactly the
        memory you wanted back.
        """
        now = time.time() if ts is None else ts
        self.last_activity = max(self.last_activity, now)
        if model:
            key = self.canonical_model(model)
            self.model_activity[key] = max(self.model_activity.get(key, 0.0), now)

    def idle_for(self, model: str) -> float:
        """Seconds since this model was last used.

        Falls back to the unit clock for a model we have never seen used — a model
        loaded before a restart, say — so an unknown model is treated as active
        rather than instantly reapable.
        """
        return time.time() - self.model_activity.get(self.canonical_model(model), self.last_activity)

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #
    async def _served_slots(self) -> dict[int, ServedModel]:
        """`spark.served_slots()` behind the backoff breaker (see `_served_fails`).

        The breaker is per NODE, not per slot: when a Spark is powered off every
        port times out together, so backing off once spares /api/status all of
        them (invariant 9).
        """
        now = time.time()
        if (self._served_fails >= self._fails_before_backoff
                and now - self._served_ts < self._probe_backoff_s):
            return {}  # presumed still down; re-probe once the backoff expires
        self._served_ts = now
        slots = await self.spark.served_slots()
        self._served_fails = 0 if slots else self._served_fails + 1
        return slots

    def _refresh_gpu_soon(self) -> None:
        """Kick a background nvidia-smi refresh if the cached sample is stale."""
        if self._gpu_task is not None and not self._gpu_task.done():
            return
        if time.time() - self._gpu_ts < self._gpu_ttl:
            return

        async def _refresh() -> None:
            try:
                info = await self.spark.gpu()
            except Exception:  # noqa: BLE001 — telemetry must never raise into status()
                info = GpuInfo(found=False, uuid=self.cfg.gpu_uuid, error="probe failed")
            self._gpu_cached, self._gpu_ts = info, time.time()

        try:
            self._gpu_task = asyncio.create_task(_refresh())
        except RuntimeError:  # no running loop (sync context) — skip silently
            self._gpu_task = None

    async def status(self, gpu: GpuInfo | None = None) -> LaneStatus:
        """Current state of the node.

        `gpu` is accepted (and ignored) so the orchestrator can call every unit
        identically; a Spark's telemetry comes from its own remote nvidia-smi,
        never from the control box's local card list.

        Only the fast HTTP probe is awaited here — see `_refresh_gpu_soon`.
        """
        slots = await self._served_slots()
        self._refresh_gpu_soon()
        remote_gpu = self._gpu_cached or GpuInfo(
            found=False, uuid=self.cfg.gpu_uuid, error="awaiting first telemetry sample"
        )

        # Serving over HTTP proves the node is up. Otherwise fall back to whether
        # the last nvidia-smi sample succeeded — an idle-but-alive node still
        # answers SSH, a powered-off one doesn't.
        reachable = bool(slots) or remote_gpu.found

        # One LoadedModel per occupied slot, ordered by port so the list — and
        # therefore the back-compat scalar below — is stable across polls.
        loaded_models: list[LoadedModel] = [
            LoadedModel(
                server="spark",
                model=info.name or "",
                root=info.root,
                context_len=info.context_len,
                # Node-wide occupancy: the unified pool is shared, so this is the
                # whole node's figure on every entry, not a per-model share.
                gpu_vram_pct=remote_gpu.vram_pct,
                fully_on_gpu=True,
                port=port,
            )
            for port, info in sorted(slots.items())
        ]
        # Forget clocks for models that have left the node. Not doing so would leave a
        # departed model's stale timestamp as the oldest one forever, defeating the idle
        # reaper's cheap pre-probe guard on every tick. Only prune once we have seen the
        # node serving something — an empty probe during a restart is not evidence.
        if slots and self.model_activity:
            resident = {self.canonical_model(m.model) for m in loaded_models}
            for gone in [k for k in self.model_activity if k not in resident]:
                self.model_activity.pop(gone, None)
        # `loaded` stays the primary occupant for every existing consumer; the list
        # is the additive surface (invariant 8/12).
        loaded: Optional[LoadedModel] = loaded_models[0] if loaded_models else None
        owner = "spark" if loaded_models else ("free" if reachable else "unknown")

        return LaneStatus(
            id=self.cfg.id,
            name=self.cfg.name,
            kind="spark",
            host=self.cfg.host,
            reachable=reachable,
            enabled=self.cfg.enabled,
            owner=owner,
            # A Spark runs neither of the local servers; these stay False so existing
            # clients reading them see "not an Ollama/vLLM unit" rather than garbage.
            ollama_up=False,
            vllm_up=False,
            loaded=loaded,
            loaded_models=loaded_models,
            gpu=GpuOut.from_info(remote_gpu),
            swap_in_progress=self._lock.locked(),
            active_job_id=self._active_job_id,
            idle_s=round(max(0.0, time.time() - self.last_activity), 1),
        )

    # ------------------------------------------------------------------ #
    # Load (Job-based, mirrors Lane.load)
    # ------------------------------------------------------------------ #
    def load(self, req: LoadRequest) -> Job:
        job = self.jobs.create(kind=f"load:{self.cfg.id}:spark:{req.model}")

        async def body(job: Job) -> dict:
            if self._lock.locked():
                self.jobs.log(job, "waiting for an in-progress swap to finish…")
            async with self._lock:
                self._active_job_id = job.id
                try:
                    return await self._load(job, req)
                finally:
                    self._active_job_id = None
                    # Clock the TARGET model, not just the unit: a fresh load must start
                    # its own idle window, or the reaper would fall back to the unit clock
                    # and treat it as busy for as long as any neighbour keeps that fresh.
                    self.touch(model=req.model)

        return self.jobs.start(job, body)

    async def _load(self, job: Job, req: LoadRequest) -> dict:
        entry = self.registry.get(req.model)
        if entry is None:
            raise RuntimeError(
                f"unknown Spark model '{req.model}' on {self.cfg.id} (see GET /api/models?lane={self.cfg.id})"
            )
        if entry.status == "blocked" and not req.force:
            raise RuntimeError(
                f"'{req.model}' is blocked: {entry.notes}. Re-issue with force=true to try anyway."
            )

        target = entry.served_name or entry.alias
        recipe = entry.recipe or entry.alias

        slots = await self.spark.served_slots()          # {port: ServedModel}
        resident = {sm.name: port for port, sm in slots.items() if sm.name}

        # Fast path: already serving what was asked for, on whichever slot.
        if not req.force and target in resident:
            self.jobs.log(
                job, f"{self.cfg.name} already serving {target} on port {resident[target]}"
            )
            return self._result(target, await self.spark.gpu(), port=resident[target])

        # Reloading a model that is already up: free ITS slot only, so co-tenants
        # survive. This replaces the old unconditional `stop --all`.
        if target in resident:
            port = resident[target]
            self.jobs.log(job, f"stopping {target} on port {port} to reload it…")
            stop = await self.spark.stop(recipe=recipe)
            if not stop.ok and stop.rc not in (0, 1):
                self.jobs.log(job, f"stop reported rc={stop.rc}: {stop.text()[:200]}")
            resident.pop(target, None)
            slots.pop(port, None)
        else:
            # A tp>1 recipe spans nodes and cannot share one; it claims everything.
            if entry.tp > 1 and slots:
                self.jobs.log(
                    job,
                    f"{recipe} is a {entry.tp}-node recipe — freeing the whole node first…",
                )
                await self.spark.stop()
                slots, resident = {}, {}

        self._admit(entry, slots)
        port = self._free_slot(slots)
        if port is None:
            raise RuntimeError(
                f"{self.cfg.name} has no free slot: all {self.cfg.max_models} are in use "
                f"({', '.join(sorted(resident)) or 'none'}). Unload one first, or raise "
                f"SPARK_MAX_MODELS."
            )

        self.jobs.log(
            job,
            f"launching {recipe} on {self.cfg.name} as '{target}' "
            f"(tp={entry.tp}, port={port}"
            + (f", mem={entry.mem_fraction}" if entry.mem_fraction else "")
            + ")…",
        )
        r = await self.spark.run_recipe(recipe, tp=entry.tp, extra=entry.extra_args,
                                        served=target, port=port,
                                        mem_fraction=entry.mem_fraction)
        if not r.ok:
            if r.rc == 127:
                raise RuntimeError(
                    "sparkrun not found in WSL — install it on the control node "
                    "(`uvx sparkrun setup install`) or fix SPARK_RUN_CMD"
                )
            if r.rc == 124:
                raise RuntimeError(f"sparkrun timed out launching '{recipe}': {r.text()[:400]}")
            raise RuntimeError(f"sparkrun failed to launch '{recipe}' (rc={r.rc}): {r.text()[:400]}")
        for line in (r.out or "").splitlines()[-5:]:
            if line.strip():
                self.jobs.log(job, line.strip())

        timeout = float(entry.load_timeout_s or self.cfg.load_timeout_s)
        self.jobs.log(
            job,
            f"waiting up to {int(timeout)}s for {target} to answer on "
            f"{self.cfg.api_base_for(port)}…",
        )
        ok = await self.spark.wait_ready(target, timeout,
                                         on_log=lambda l: self.jobs.log(job, l), port=port)
        if not ok:
            tail = await self.spark.logs(n=25)
            raise RuntimeError(
                f"{self.cfg.name} did not serve '{target}' within {int(timeout)}s.\n{tail}"
            )

        gpu = await self.spark.gpu()
        # A successful load proves the node is live — clear the probe breaker so
        # status() stops backing off immediately.
        self._gpu_cached, self._gpu_ts = gpu, time.time()
        self._served_fails = 0
        self.jobs.log(
            job, f"{self.cfg.name} serving {target} on port {port} (VRAM {gpu.vram_pct}% used)"
        )
        return self._result(target, gpu, port=port)

    # ------------------------------------------------------------------ #
    # Unload
    # ------------------------------------------------------------------ #
    async def unload(self, req: UnloadRequest) -> LaneStatus:
        """Free one model, or the whole node.

        `req.model` names a single model to stop, leaving co-residents running —
        that is what per-model reaping and a targeted UI Unload need. Without it
        the whole node is freed, which is what the idle reaper's "free the unit"
        and a lease's `free_on_preempt` still mean.
        """
        async with self._lock:
            self._active_job_id = None
            recipe = None
            if req.model:
                entry = self.registry.get(req.model) or self.registry.find_by_served_name(req.model)
                if entry is None:
                    raise RuntimeError(
                        f"unknown Spark model '{req.model}' on {self.cfg.id} — cannot "
                        f"target it for unload"
                    )
                recipe = entry.recipe or entry.alias
            await self.spark.stop(recipe=recipe)
        return await self.status()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _result(self, served_name: str, gpu: GpuInfo, port: int = 0) -> dict:
        return LoadedModel(
            server="spark",
            model=served_name,
            gpu_vram_pct=gpu.vram_pct,
            fully_on_gpu=True,
            port=port,
        ).model_dump()

    def _free_slot(self, slots: dict) -> Optional[int]:
        """Lowest unoccupied slot port, or None when the node is full."""
        return next((p for p in self.cfg.slot_ports if p not in slots), None)

    def _admit(self, entry, slots: dict) -> None:
        """Refuse a load whose declared budget will not fit beside the residents.

        A Spark has no eviction-wait gate — nothing observes memory before a
        launch — so this declared-budget sum is the only thing standing between
        co-residency and an OOM at load. `mem_fraction: 0.0` means "unset", which
        is read as a whole-node claim: it neither fits beside anything nor lets
        anything fit beside it, which keeps pre-multi-model catalogs behaving as
        they always did.
        """
        want = entry.mem_fraction
        by_name = {sm.name: sm for sm in slots.values() if sm.name}
        if not by_name:
            return
        budgets = {}
        for name in by_name:
            e = self.registry.find_by_served_name(name)
            budgets[name] = e.mem_fraction if e else 0.0

        unbudgeted = [n for n, f in budgets.items() if not f]
        if unbudgeted or not want:
            who = ", ".join(sorted(unbudgeted)) or entry.alias
            raise RuntimeError(
                f"cannot co-locate '{entry.alias}' on {self.cfg.name}: "
                f"{who} has no mem_fraction, so it is treated as claiming the whole "
                f"node. Set mem_fraction on every model that should share a node, "
                f"or unload the other model first."
            )
        used = sum(budgets.values())
        headroom = self.s.spark_mem_headroom
        if used + want > headroom + 1e-9:
            raise RuntimeError(
                f"'{entry.alias}' needs {want:.2f} of {self.cfg.name} but "
                f"{used:.2f}/{headroom:.2f} is already committed to "
                f"{', '.join(sorted(budgets))}. Unload something or lower mem_fraction."
            )

    async def aclose(self) -> None:
        if self._gpu_task is not None and not self._gpu_task.done():
            self._gpu_task.cancel()
        await self.spark.aclose()
