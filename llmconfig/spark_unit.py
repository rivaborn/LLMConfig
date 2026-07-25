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

    def touch(self, ts: float | None = None) -> None:
        """Record activity for the idle reaper. Never moves the clock backwards."""
        self.last_activity = max(self.last_activity, time.time() if ts is None else ts)

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #
    async def _served(self) -> ServedModel:
        """`spark.served_info()` behind the backoff breaker (see `_served_fails`)."""
        now = time.time()
        if (self._served_fails >= self._fails_before_backoff
                and now - self._served_ts < self._probe_backoff_s):
            return ServedModel()  # presumed still down; re-probe once the backoff expires
        self._served_ts = now
        info = await self.spark.served_info()
        self._served_fails = 0 if info.name else self._served_fails + 1
        return info

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
        info = await self._served()
        served = info.name
        self._refresh_gpu_soon()
        remote_gpu = self._gpu_cached or GpuInfo(
            found=False, uuid=self.cfg.gpu_uuid, error="awaiting first telemetry sample"
        )

        # Serving over HTTP proves the node is up. Otherwise fall back to whether
        # the last nvidia-smi sample succeeded — an idle-but-alive node still
        # answers SSH, a powered-off one doesn't.
        reachable = bool(served) or remote_gpu.found

        loaded: Optional[LoadedModel] = None
        if served:
            owner = "spark"
            loaded = LoadedModel(
                server="spark",
                model=served,
                root=info.root,
                context_len=info.context_len,
                gpu_vram_pct=remote_gpu.vram_pct,
                fully_on_gpu=True,
            )
        else:
            owner = "free" if reachable else "unknown"

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
                    self.touch()

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

        # Fast path: already serving what was asked for.
        if not req.force and (await self.spark.served()) == target:
            self.jobs.log(job, f"{self.cfg.name} already serving {target}")
            return self._result(target, await self.spark.gpu())

        # A Spark runs one workload at a time, so a swap is stop-then-run. There is
        # no VRAM gate to wait on — stopping the container releases the whole node.
        self.jobs.log(job, f"stopping any running workload on {self.cfg.name}…")
        stop = await self.spark.stop()
        if not stop.ok and stop.rc not in (0, 1):
            # rc 1 is the common "nothing running" case; anything else is worth showing.
            self.jobs.log(job, f"stop reported rc={stop.rc}: {stop.text()[:200]}")

        recipe = entry.recipe or entry.alias
        self.jobs.log(job, f"launching {recipe} on {self.cfg.name} as '{target}' (tp={entry.tp})…")
        r = await self.spark.run_recipe(recipe, tp=entry.tp, extra=entry.extra_args,
                                        served=target)
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
        self.jobs.log(job, f"waiting up to {int(timeout)}s for {target} to answer on {self.cfg.api_base}…")
        ok = await self.spark.wait_ready(target, timeout, on_log=lambda l: self.jobs.log(job, l))
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
        self.jobs.log(job, f"{self.cfg.name} serving {target} (VRAM {gpu.vram_pct}% used)")
        return self._result(target, gpu)

    # ------------------------------------------------------------------ #
    # Unload
    # ------------------------------------------------------------------ #
    async def unload(self, req: UnloadRequest) -> LaneStatus:
        async with self._lock:
            self._active_job_id = None
            await self.spark.stop()
        return await self.status()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _result(self, served_name: str, gpu: GpuInfo) -> dict:
        return LoadedModel(
            server="spark",
            model=served_name,
            gpu_vram_pct=gpu.vram_pct,
            fully_on_gpu=True,
        ).model_dump()

    async def aclose(self) -> None:
        if self._gpu_task is not None and not self._gpu_task.done():
            self._gpu_task.cancel()
        await self.spark.aclose()
