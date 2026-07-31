"""SlotLane — a GPU lane serving SEVERAL vLLM models at once, one per SLOT.

The classic `Lane` guarantees a sole occupant via the eviction-wait gate; that is
exactly wrong for the daily-driver 3070 Ti, which must hold surya-ocr-2 AND the
opencode relay side by side, where a reload of one must never blip the other.

This class borrows SparkUnit's *shape* (multi-resident, per-model port, additive
`loaded_models`, targeted unload) but none of its dynamic admission: the slot
table is STATIC config (`COMPANION_VLLM_SLOTS`, parsed onto `LaneConfig.vllm_slots`)
and "admission" is simply "the alias must name a slot". Isolation is by
construction, not discipline — every lifecycle call goes through the slot's OWN
`vllm-companion@<alias>` systemd instance (`stop_instance`/`serve_instance`);
nothing here can address a sibling's unit. The eviction-wait gate survives in a
narrowed form: before (re)launching a slot, wait until the card has at least the
slot's declared budget free — an inverted per-slot threshold instead of
drain-to-baseline (baseline would mean "sibling gone", the opposite of the point).

vLLM-only by design. The slot shape exists because both residents declare
`--gpu-memory-utilization`; an Ollama co-resident declares nothing and would
break the budget arithmetic (the dropped doc item 5). The lane's Ollama half is
reported down and refused.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

from .backends.vllm import VllmBackend
from .config import LaneConfig, Settings
from .gpu import GpuInfo, query_gpu
from .jobs import JobManager
from .registry import Registry
from .schemas import (GpuOut, Job, LaneStatus, LoadedModel, LoadRequest,
                      ServedModel, UnloadRequest)
from .wsl import WslKeepalive


class SlotLane:
    kind = "gpu"

    def __init__(
        self,
        settings: Settings,
        cfg: LaneConfig,
        registry: Registry,
        jobs: JobManager,
        keepalive: WslKeepalive,
    ):
        self.s = settings
        self.cfg = cfg
        self.registry = registry
        self.jobs = jobs
        self.keepalive = keepalive
        # alias -> (relay_port, budget_mb); insertion order = display order.
        self.slots: dict[str, tuple[int, int]] = {
            alias: (port, budget) for alias, port, budget in cfg.vllm_slots
        }
        # One backend per slot: its OWN relay URL and its own @<alias> instance.
        self.backends: dict[str, VllmBackend] = {
            alias: VllmBackend(
                settings,
                registry,
                relay_url=f"http://127.0.0.1:{port}",
                serve_script=cfg.vllm_serve_script,
                systemd_unit=cfg.vllm_systemd_unit,
            )
            for alias, (port, _) in self.slots.items()
        }
        # ONE swap lock for the whole lane: slot swaps are serialized on purpose —
        # two concurrent vLLM profile runs against one 8 GB card is how you OOM
        # the one that was innocent. Sibling SERVING is untouched; only SWAPPING
        # queues.
        self._lock = asyncio.Lock()
        self._active_job_id: Optional[str] = None
        self.load_times = None                 # set by attach_load_times()
        self.last_activity: float = time.time()

    def touch(self, ts: float | None = None, model: str | None = None) -> None:
        """Uniform Unit contract; `model` accepted and folded into the unit clock
        (slot lanes are reap-exempt in practice, so per-model clocks buy nothing)."""
        self.last_activity = max(self.last_activity, time.time() if ts is None else ts)

    async def aclose(self) -> None:
        await asyncio.gather(*(b.aclose() for b in self.backends.values()))

    # ------------------------------------------------------------------ #
    # Lookup helpers
    # ------------------------------------------------------------------ #
    def relay_url_for(self, alias: str) -> str:
        """The slot's own relay URL — the gateway's routing hook (duck-typed:
        `resolve()` prefers this over `cfg.vllm_relay_url` when present).

        An alias that is catalogued but has NO slot falls back to the lane's
        configured relay: routing must not 500 on it — the LOAD path is where
        it gets the actionable 'add it to COMPANION_VLLM_SLOTS' refusal."""
        slot = self.slots.get(alias)
        if slot is None:
            return self.cfg.vllm_relay_url
        return f"http://127.0.0.1:{slot[0]}"

    def _slot_of(self, name: str) -> Optional[str]:
        """Resolve an alias OR served name onto a slot alias, else None."""
        if name in self.slots:
            return name
        for alias in self.slots:
            e = self.registry.get(alias)
            if e is not None and (e.served_name or e.alias) == name:
                return alias
        return None

    async def vllm_up(self) -> bool:
        """Any slot serving? (Keepalive accounting — see idle reaper.)"""
        served = await asyncio.gather(*(b.served() for b in self.backends.values()))
        return any(s is not None for s in served)

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #
    async def status(self, gpu: GpuInfo | None = None) -> LaneStatus:
        probes = [b.served_info() for b in self.backends.values()]
        if gpu is None:
            *infos, gpu = await asyncio.gather(*probes, self._gpu())
        else:
            infos = await asyncio.gather(*probes)

        loaded_models: list[LoadedModel] = []
        for (alias, (port, _)), info in zip(self.slots.items(), infos):
            info: ServedModel
            if not info.name:
                continue
            loaded_models.append(LoadedModel(
                server="vllm",
                model=info.name,
                root=info.root,
                context_len=info.context_len,
                gpu_vram_pct=gpu.vram_pct,
                fully_on_gpu=True,
                port=port,
            ))

        if loaded_models:
            owner = "vllm"
        elif not gpu.found or gpu.is_free(self.cfg.vram_free_baseline_mb):
            owner = "free"
        else:
            owner = "unknown"

        return LaneStatus(
            id=self.cfg.id,
            name=self.cfg.name,
            enabled=self.cfg.enabled,
            owner=owner,
            ollama_up=False,               # vLLM-only by design (module docstring)
            vllm_up=bool(loaded_models),
            loaded=loaded_models[0] if loaded_models else None,
            loaded_models=loaded_models,
            gpu=GpuOut.from_info(gpu),
            swap_in_progress=self._lock.locked(),
            active_job_id=self._active_job_id,
            idle_s=round(max(0.0, time.time() - self.last_activity), 1),
        )

    async def _gpu(self) -> GpuInfo:
        return await query_gpu(self.s, uuid=self.cfg.gpu_uuid)

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #
    def load(self, req: LoadRequest) -> Job:
        job = self.jobs.create(kind=f"load:{self.cfg.id}:{req.server}:{req.model}")

        async def body(job: Job) -> dict:
            if req.server == "ollama":
                raise RuntimeError(
                    f"the {self.cfg.id} lane is slot-mode vLLM-only (see "
                    f"COMPANION_VLLM_SLOTS) — Ollama declares no budget and "
                    f"cannot join a budgeted card")
            if self._lock.locked():
                self.jobs.log(job, "waiting for an in-progress slot swap to finish… "
                                   f"(holder: {self._active_job_id or 'unknown'})")
            await self._acquire_swap_lock("load")
            self._active_job_id = job.id
            try:
                return await self._load_slot(job, req)
            finally:
                self._active_job_id = None
                self.touch()
                self._lock.release()

        return self.jobs.start(job, body)

    async def _load_slot(self, job: Job, req: LoadRequest) -> dict:
        alias = req.model
        entry = self.registry.get(alias)
        if entry is None:
            # A served name is a valid way to ask (the gateway resolves aliases,
            # humans paste served names) — fold it back onto the slot's alias.
            slot = self._slot_of(alias)
            entry = self.registry.get(slot) if slot else None
            if entry is None:
                raise RuntimeError(f"unknown vLLM alias '{alias}' (see GET /api/models)")
            alias = slot
        if entry.status == "blocked" and not req.force:
            raise RuntimeError(f"alias '{alias}' is blocked: {entry.notes}. "
                               f"Re-issue with force=true to try anyway.")
        if alias not in self.slots:
            raise RuntimeError(
                f"'{alias}' has no slot on the {self.cfg.id} lane — slots are "
                f"static config ({', '.join(self.slots) or 'none configured'}); "
                f"add it to COMPANION_VLLM_SLOTS and restart")

        backend = self.backends[alias]
        port, budget_mb = self.slots[alias]
        served_target = entry.served_name or alias

        if not req.force and (await backend.served()) == served_target:
            self.jobs.log(job, f"slot {alias} already serving {served_target}")
            return self._result(served_target, await self._gpu(), port)

        if not self.keepalive.ensure():
            self.jobs.log(job, "warning: could not start the WSL keepalive (wsl.exe "
                               "missing?); vLLM may not survive WSL idle-shutdown")
        else:
            self.jobs.log(job, "WSL keepalive active (distro held open)")

        # Tear down THIS slot only, then the narrowed gate: wait until the card
        # has this slot's budget free. The sibling keeps serving throughout —
        # that is the whole point of the class.
        self.jobs.log(job, f"stopping slot instance {self.cfg.vllm_systemd_unit}{alias}…")
        await backend.stop_instance(alias)
        await self._wait_budget_free(job, budget_mb)

        launch_started = time.monotonic()
        self.jobs.log(job, f"starting slot {alias} (relay :{port}, budget {budget_mb} MiB)…")
        r = await backend.serve_instance(alias)
        if not r.ok and ("not found" in r.err.lower() or "not loaded" in r.err.lower()):
            raise RuntimeError(
                f"systemd unit '{self.cfg.vllm_systemd_unit}{alias}' not found — install "
                f"deploy/vllm-companion@.service into ~/.config/systemd/user/ and "
                f"`systemctl --user daemon-reload`. Detail: {r.text()}"
            )

        timeout = float(entry.load_timeout_s or self.s.default_vllm_load_timeout_s)
        self.jobs.log(job, f"waiting up to {int(timeout)}s for {served_target} to be ready…")
        ok = await backend.wait_ready(
            served_target, timeout, on_log=lambda l: self.jobs.log(job, l), alias=alias
        )
        if not ok:
            self.jobs.log(job, f"readiness wait hit {int(timeout)}s; grace re-check "
                               f"({self.s.vllm_ready_grace_s}s)…")
            ok = await backend.wait_ready(served_target,
                                          float(self.s.vllm_ready_grace_s), alias=alias)
        if not ok:
            tail = await backend.journal_tail(alias, n=25)
            if self.load_times is not None:
                from .load_times import fail_key
                self.load_times.record_failure(fail_key(self.cfg.id, "vllm", alias))
            raise RuntimeError(
                f"vLLM did not become ready for '{alias}' within {int(timeout)}s "
                f"(+{self.s.vllm_ready_grace_s}s grace).\n{tail}"
            )

        if self.load_times is not None:
            from .load_times import fail_key, lane_key
            self.load_times.record(lane_key(self.cfg.id, "vllm", alias),
                                   time.monotonic() - launch_started, unit=self.cfg.id)
            self.load_times.clear_failures(fail_key(self.cfg.id, "vllm", alias))
        gpu = await self._gpu()
        self.jobs.log(job, f"slot {alias} serving {served_target} "
                           f"(VRAM {gpu.vram_pct}% used)")
        return self._result(served_target, gpu, port)

    # ------------------------------------------------------------------ #
    # Unload
    # ------------------------------------------------------------------ #
    async def unload(self, req: UnloadRequest) -> LaneStatus:
        await self._acquire_swap_lock("unload")
        try:
            self._active_job_id = None
            if req.model:
                slot = self._slot_of(req.model)
                if slot is None or (await self.backends[slot].served()) is None:
                    # Targeted unload of something not resident: no-op, same
                    # neighbour-must-survive contract as Lane/SparkUnit.
                    return await self.status()
                await self.backends[slot].stop_instance(slot)
            else:
                # Whole-lane free: stop every slot instance, individually.
                for alias, backend in self.backends.items():
                    await backend.stop_instance(alias)
        finally:
            self._lock.release()
        return await self.status()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    async def _acquire_swap_lock(self, what: str) -> None:
        # Same bounded-acquire contract (and bare fast path) as Lane — see
        # Lane._acquire_swap_lock for the invariant-11 reasoning.
        if not self._lock.locked():
            await self._lock.acquire()
            return
        try:
            await asyncio.wait_for(self._lock.acquire(),
                                   timeout=self.s.swap_wait_timeout_s)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"timed out after {self.s.swap_wait_timeout_s:.0f}s waiting for the "
                f"{self.cfg.id} swap lock to {what} (held by job "
                f"{self._active_job_id or 'unknown'}) — the unit is wedged, not busy"
            ) from None

    async def _wait_budget_free(self, job: Job, budget_mb: int) -> bool:
        """The narrowed eviction-wait gate: block until >= `budget_mb` is free on
        the card (NOT drain-to-baseline — the sibling slot is supposed to stay)."""
        deadline = time.monotonic() + self.s.evict_timeout_s
        while time.monotonic() < deadline:
            gpu = await query_gpu(self.s, uuid=self.cfg.gpu_uuid, with_processes=False)
            if not gpu.found:
                self.jobs.log(job, "nvidia-smi unavailable — skipping budget-free wait")
                return True
            if gpu.free_mb >= budget_mb:
                self.jobs.log(job, f"{gpu.free_mb} MiB free (slot budget {budget_mb})")
                return True
            self.jobs.log(job, f"waiting for {budget_mb} MiB free… "
                               f"({gpu.free_mb} MiB free now)")
            await asyncio.sleep(self.s.poll_interval_s)
        self.jobs.log(job, "warning: slot budget did not come free before timeout; "
                           "continuing (vLLM will profile against what exists)")
        return False

    def _result(self, served_name: str, gpu: GpuInfo, port: int) -> dict:
        return LoadedModel(
            server="vllm",
            model=served_name,
            gpu_vram_pct=gpu.vram_pct,
            fully_on_gpu=True,
            port=port,
        ).model_dump()
