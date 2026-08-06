"""One GPU lane — the arbitration state machine for a single card.

Each `Lane` pins one GPU (by UUID) and guarantees the requested model becomes the
*sole* occupant of that card: the other server plus any other Ollama models are
evicted and the VRAM is confirmed freed (via nvidia-smi) **before** the target is
loaded, so it packs 100 % of VRAM before any CPU spill. All swaps on a lane are
serialized behind that lane's own lock.

The `Orchestrator` runs one Lane per `LaneConfig` (primary = RTX 3090, optional
companion = RTX 3070 Ti); the lanes are fully independent — loading on one never
touches the other's card.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

from .backends.ollama import OllamaBackend
from .backends.vllm import VllmBackend
from .config import LaneConfig, Settings
from .gpu import GpuInfo, query_gpu
from .jobs import JobManager
from .registry import Registry
from .schemas import Job, LaneStatus, LoadedModel, LoadRequest, ServedModel, UnloadRequest
from .schemas import GpuOut
from .wsl import WslKeepalive


class Lane:
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
        self.ollama = OllamaBackend(
            settings, base_url=cfg.ollama_url, service_name=cfg.ollama_service_name
        )
        self.vllm = VllmBackend(
            settings,
            registry,
            relay_url=cfg.vllm_relay_url,
            serve_script=cfg.vllm_serve_script,
            systemd_unit=cfg.vllm_systemd_unit,
        )
        self._lock = asyncio.Lock()
        self._active_job_id: Optional[str] = None
        # Set by Orchestrator.attach_load_times() / attach_leases() /
        # attach_stats(); None = don't record / no lease awareness.
        self.load_times = None
        self.leases = None
        self.stats = None
        # Holders whose leases the in-flight load revoked, handed from
        # _preempt_occupant_leases to _note_evictions (the lease is terminal by
        # the time the eviction itself completes).
        self._displaced_holders: list[str] = []
        # Idle-reaper input: wall-clock of the last observed activity (gateway
        # request, load completion, or a Monitor util spike). Construction time =
        # app start, so an autoloaded default gets a full idle window before reaping.
        self.last_activity: float = time.time()

    kind = "gpu"

    def touch(self, ts: float | None = None, model: str | None = None) -> None:
        """Record lane activity for the idle reaper. Never moves the clock backwards,
        so a stale Monitor sample can't shorten the idle window.

        `model` is accepted for a uniform Unit contract and ignored: the eviction-wait
        gate means a lane has at most one occupant, so the unit clock IS that model's
        clock.
        """
        self.last_activity = max(self.last_activity, time.time() if ts is None else ts)

    async def aclose(self) -> None:
        """Close pooled HTTP clients (part of the shared unit contract)."""
        await self.ollama.aclose()
        await self.vllm.aclose()

    async def _gpu(self) -> GpuInfo:
        """This lane's card only (by UUID) — never the other lane's."""
        return await query_gpu(self.s, uuid=self.cfg.gpu_uuid)

    async def vllm_up(self) -> bool:
        """Is this lane serving vLLM right now? Part of the shared unit contract —
        SlotLane answers for all its slots, so callers (the idle reaper's
        keepalive accounting) never reach into `lane.vllm` directly."""
        return self.cfg.vllm_enabled and await self.vllm.up()

    async def _served_info(self) -> ServedModel:
        """The relay's view, or an empty answer on an Ollama-only lane.

        Skipping the probe matters for latency, not just tidiness: a DOWN relay
        blackholes the SYN rather than refusing it (invariant 5), so probing a
        relay that will never exist costs `vllm_probe_timeout_s` on every
        /api/status — which the UI polls every 2.5 s.
        """
        if not self.cfg.vllm_enabled:
            return ServedModel()
        return await self.vllm.served_info()

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #
    async def status(self, gpu: GpuInfo | None = None) -> LaneStatus:
        # `gpu` may be supplied by the Orchestrator (one nvidia-smi shared across
        # lanes); fetch this lane's card only when called standalone.
        if gpu is None:
            served_info, ollama_loaded, ollama_up, gpu = await asyncio.gather(
                self._served_info(),
                self.ollama.loaded(),
                self.ollama.up(),
                self._gpu(),
            )
        else:
            served_info, ollama_loaded, ollama_up = await asyncio.gather(
                self._served_info(),
                self.ollama.loaded(),
                self.ollama.up(),
            )
        served, served_root = served_info.name, served_info.root

        loaded: Optional[LoadedModel] = None
        if served:
            owner = "vllm"
            loaded = LoadedModel(
                server="vllm",
                model=served,
                root=served_root,
                context_len=served_info.context_len,
                gpu_vram_pct=gpu.vram_pct,
                fully_on_gpu=True,
            )
        elif ollama_loaded:
            owner = "ollama"
            m = max(ollama_loaded, key=lambda x: x.size_vram_bytes)
            on_cpu = max(0, m.size_bytes - m.size_vram_bytes)
            loaded = LoadedModel(
                server="ollama",
                model=m.name,
                context_len=m.context_len,
                size_bytes=m.size_bytes,
                on_gpu_bytes=m.size_vram_bytes,
                on_cpu_bytes=on_cpu,
                spilled=on_cpu > 0,
                fully_on_gpu=on_cpu == 0,
                gpu_vram_pct=gpu.vram_pct,
            )
        else:
            owner = "free" if (not gpu.found or gpu.is_free(self.cfg.vram_free_baseline_mb)) else "unknown"

        return LaneStatus(
            id=self.cfg.id,
            name=self.cfg.name,
            enabled=self.cfg.enabled,
            owner=owner,
            ollama_up=ollama_up,
            vllm_up=served is not None,
            loaded=loaded,
            # A GPU lane holds at most one model — the eviction-wait gate guarantees
            # it — so the additive list is just the scalar, wrapped. Populating it
            # here keeps "loaded_models is empty iff loaded is None" true for every
            # unit kind, so consumers can read the list unconditionally.
            loaded_models=[loaded] if loaded else [],
            gpu=GpuOut.from_info(gpu),
            swap_in_progress=self._lock.locked(),
            active_job_id=self._active_job_id,
            idle_s=round(max(0.0, time.time() - self.last_activity), 1),
        )

    # ------------------------------------------------------------------ #
    # Load (returns a Job; the swap runs in the background under the lock)
    # ------------------------------------------------------------------ #
    def load(self, req: LoadRequest) -> Job:
        job = self.jobs.create(kind=f"load:{self.cfg.id}:{req.server}:{req.model}")

        async def body(job: Job) -> dict:
            # BOUNDED acquire. An unbounded `async with self._lock` means one
            # wedged holder blocks every later load forever: on 2026-07-28 a
            # single stuck launch stacked 29 jobs behind it and the unit read as
            # "busy" rather than broken, for six hours. Fail fast instead, and
            # name the holder so the report points at the real culprit.
            if self._lock.locked():
                self.jobs.log(job, "waiting for an in-progress swap to finish… "
                                   f"(holder: {self._active_job_id or 'unknown'})")
            await self._acquire_swap_lock("load")
            self._active_job_id = job.id
            try:
                if req.server == "ollama":
                    return await self._load_ollama(job, req)
                return await self._load_vllm(job, req)
            finally:
                self._active_job_id = None
                self.touch()  # a load (even a failed one) restarts the idle window
                self._lock.release()

        return self.jobs.start(job, body)

    async def _load_ollama(self, job: Job, req: LoadRequest) -> dict:
        # Fast path: already loaded, nothing else on the GPU, not forced.
        # SOLE resident, not membership: a direct-to-Ollama client (ungated by
        # design, invariant 12) can park a second model on the card — the fast
        # path must not bless that state; fall through and evict instead.
        # The vLLM probe is skipped on an Ollama-only lane, same reasoning as
        # `_served_info`: a relay that will never exist blackholes the SYN
        # (invariant 5) and taxed EVERY companion load ~1 s for nothing.
        vllm_served = (await self.vllm.served()) if self.cfg.vllm_enabled else None
        if not req.force and vllm_served is None:
            if set(await self.ollama.loaded_names()) == {req.model}:
                self.jobs.log(job, f"{req.model} already loaded on Ollama")
                return await self._verify_ollama(job, req, remediate=False)

        reason = await self._preempt_occupant_leases(job, req)
        displaced = await self._evict_all(job)
        if vllm_served:
            displaced = [vllm_served, *displaced]
        self._note_evictions(req, reason, displaced)

        self.jobs.log(job, "ensuring Ollama service is running…")
        if not await self.ollama.ensure_running():
            raise RuntimeError("Ollama service is not reachable (check the Windows service / OLLAMA_URL)")

        num_gpu = None  # default: let Ollama auto-fit against the now-empty GPU
        # Load-time clock: after the evict-wait gate (eviction time isn't launch
        # time), through verify — max_pack's reload included, it's honest wall time.
        launch_started = time.monotonic()
        self.jobs.log(job, f"loading {req.model} into Ollama…")
        try:
            await self.ollama.load(req.model, keep_alive=req.keep_alive, num_gpu=num_gpu)
            result = await self._verify_ollama(job, req, remediate=req.max_pack)
        except Exception:
            # Post-eviction launch failure — feeds placement's per-unit blocklist
            # (consecutive count; the next success clears it).
            if self.load_times is not None:
                from .load_times import fail_key
                self.load_times.record_failure(fail_key(self.cfg.id, "ollama", req.model))
            raise
        if self.load_times is not None:  # success only — an exception skipped this
            from .load_times import fail_key, lane_key
            self.load_times.record(lane_key(self.cfg.id, "ollama", req.model),
                                   time.monotonic() - launch_started, unit=self.cfg.id)
            self.load_times.clear_failures(fail_key(self.cfg.id, "ollama", req.model))
        return result

    async def _load_vllm(self, job: Job, req: LoadRequest) -> dict:
        # Refuse FIRST, before the keepalive and before any eviction: this lane's
        # vLLM half is not installed, and the old order unloaded the lane's
        # working Ollama model and drained its VRAM before discovering that the
        # systemd unit was missing — a request for a model that can never run
        # destroyed the model that was running.
        if not self.cfg.vllm_enabled:
            raise RuntimeError(
                f"vLLM is not available on the {self.cfg.id} lane (it has no serve "
                f"script installed — see COMPANION_VLLM_ENABLED). Use Ollama on this "
                f"lane, or another unit."
            )
        alias = req.model
        entry = self.registry.get(alias)
        if entry is None:
            raise RuntimeError(f"unknown vLLM alias '{alias}' (see GET /api/models)")
        if entry.status == "blocked" and not req.force:
            raise RuntimeError(f"alias '{alias}' is blocked: {entry.notes}. Re-issue with force=true to try anyway.")

        served_target = entry.served_name or alias
        cur_served = await self.vllm.served()
        if not req.force and cur_served == served_target:
            self.jobs.log(job, f"vLLM already serving {served_target}")
            return self._vllm_result(served_target, await self._gpu())

        reason = await self._preempt_occupant_leases(job, req)

        # Hold WSL open before starting vLLM: otherwise the distro idle-shuts-down
        # seconds after this load returns and takes the model (and relay) with it.
        if not self.keepalive.ensure():
            self.jobs.log(job, "warning: could not start the WSL keepalive (wsl.exe missing?); "
                               "vLLM may not survive WSL idle-shutdown")
        else:
            self.jobs.log(job, "WSL keepalive active (distro held open)")

        # Stop any OLD vLLM before the drain wait. serve() stops internally, but
        # that runs AFTER _wait_vram_free — on a vLLM→vLLM swap the wait would
        # poll a card the old model still occupies, burn the full evict timeout,
        # and then load with no confirmed drain (the one path that skipped the
        # eviction-wait gate; found in review 2026-07-29). stop() is idempotent
        # and lane-scoped, so this is safe when vLLM is already down.
        await self.vllm.stop()
        self.jobs.log(job, "unloading any Ollama models…")
        names = await self.ollama.unload_all()
        if names:
            self.jobs.log(job, f"unloaded Ollama: {', '.join(names)}")
        await self._wait_vram_free(job)
        self._note_evictions(
            req, reason,
            [n for n in ([cur_served] if cur_served else []) + names
             if n != served_target])

        # Load-time clock: after eviction + VRAM drain, from serve.sh start to ready.
        launch_started = time.monotonic()
        self.jobs.log(job, f"starting vLLM: serve.sh {alias}…")
        r = await self.vllm.serve(alias)
        if not r.ok and ("not found" in r.err.lower() or "not loaded" in r.err.lower()):
            raise RuntimeError(
                f"systemd unit '{self.cfg.vllm_systemd_unit}{alias}' not found — install deploy/vllm@.service "
                f"into ~/.config/systemd/user/ and `systemctl --user daemon-reload`. Detail: {r.text()}"
            )

        timeout = float(entry.load_timeout_s or self.s.default_vllm_load_timeout_s)
        self.jobs.log(job, f"waiting up to {int(timeout)}s for {served_target} to be ready…")
        ok = await self.vllm.wait_ready(
            served_target, timeout, on_log=lambda l: self.jobs.log(job, l), alias=alias
        )
        if not ok:
            # A heavy `mode: compile` alias can report ready a beat after its per-alias
            # deadline; re-check briefly before failing, so we don't fail — and have the
            # load torn down downstream — a vLLM that actually came up.
            self.jobs.log(job, f"readiness wait hit {int(timeout)}s; grace re-check ({self.s.vllm_ready_grace_s}s)…")
            ok = await self.vllm.wait_ready(served_target, float(self.s.vllm_ready_grace_s), alias=alias)
        if not ok:
            tail = await self.vllm.journal_tail(alias, n=25)
            # Feeds placement's per-unit blocklist (consecutive count; a success
            # clears). The systemd-unit-not-found raise above deliberately does
            # NOT count — that's an install fault, not this model failing.
            if self.load_times is not None:
                from .load_times import fail_key
                self.load_times.record_failure(fail_key(self.cfg.id, "vllm", alias))
            raise RuntimeError(
                f"vLLM did not become ready for '{alias}' within {int(timeout)}s "
                f"(+{self.s.vllm_ready_grace_s}s grace).\n{tail}"
            )

        if self.load_times is not None:  # success only — failures raised above
            from .load_times import fail_key, lane_key
            self.load_times.record(lane_key(self.cfg.id, "vllm", alias),
                                   time.monotonic() - launch_started, unit=self.cfg.id)
            self.load_times.clear_failures(fail_key(self.cfg.id, "vllm", alias))
        gpu = await self._gpu()
        self.jobs.log(job, f"vLLM serving {served_target} (VRAM {gpu.vram_pct}% used)")
        return self._vllm_result(served_target, gpu)

    # ------------------------------------------------------------------ #
    # Unload (synchronous eviction)
    # ------------------------------------------------------------------ #
    async def unload(self, req: UnloadRequest) -> LaneStatus:
        # Bounded like load(): unload is the natural "get me out of this" move
        # when a unit is stuck, so it must not itself block on the wedged lock.
        # On 2026-07-28 /api/unload hung with no response — the one call that
        # should have cleared the wedge was queued behind it.
        await self._acquire_swap_lock("unload")
        try:
            self._active_job_id = None
            if req.model and not await self._occupied_by(req.model):
                # Targeted unload of a model that is not resident: no-op. The
                # reaper/sweeper name the victim they saw — a neighbour loaded
                # between their probe and this call must survive, not be
                # collateral (review 2026-07-29; SparkUnit already behaves so).
                # (The finally below releases the lock.)
                return await self.status()
            if req.server in (None, "vllm") and self.cfg.vllm_enabled:
                # Unconditional (given the lane HAS vLLM): gating on the 1 s relay
                # probe skipped eviction whenever the relay was dead while vLLM
                # still held the card. stop() is idempotent and lane-scoped.
                await self.vllm.stop()
            if req.server in (None, "ollama"):
                await self.ollama.unload_all()
            await self._wait_vram_free(None)
        finally:
            self._lock.release()
        return await self.status()

    async def _occupied_by(self, model: str) -> bool:
        """Is `model` (a served name or Ollama tag) resident on this lane now?

        The vLLM probe is skipped on an Ollama-only lane (invariant 5's
        blackholed-SYN tax — see `_served_info`)."""
        served = (await self.vllm.served()) if self.cfg.vllm_enabled else None
        if served is not None:
            return served == model
        return model in await self.ollama.loaded_names()

    async def _acquire_swap_lock(self, what: str) -> None:
        """Acquire the unit's swap lock, or raise rather than wait forever.

        The uncontended path is a BARE acquire on purpose: `asyncio.wait_for`
        wraps the acquire in a task and always yields at least once even when
        the lock is free — which re-opens the check-then-act window the idle
        reaper's and lease sweeper's final sync guard depends on closing
        (invariant 11's reasoning). A bare acquire on a free lock completes
        without yielding."""
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

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    async def _occupied(self) -> bool:
        """Anything resident on this lane? (vLLM probe skipped on an
        Ollama-only lane — invariant 5.)"""
        if self.cfg.vllm_enabled and (await self.vllm.served()) is not None:
            return True
        return bool(await self.ollama.loaded_names())

    async def _preempt_occupant_leases(self, job: Job, req: LoadRequest) -> str:
        """Lease gate + revocation before this lane's eviction.

        Runs under the lane lock, just before the evict that displaces the
        occupant. Two halves:

        * A live NON-preemptible lease held by someone else refuses the load
          (`placement_conflict:`) — the endpoint gate refuses up front, this
          closes the claim-race window under the lock.
        * A placement-driven load (`req.priority is not None`) re-validates the
          occupant the same way `_victim_eligible` ranked it: an ACTIVE
          occupant must be preemptibly held below the incoming priority; an
          idle preemptibly-held one needs the rule-1 switch on. An explicit
          load (priority None — operator REST, boot autoload, cookbook) keeps
          the lane's last-writer-wins semantics and skips this half.

        Then every preemptible lease on the unit is REVOKED (the whole card is
        being taken; a model-scoped lease must not survive its model), sparing
        the requester's own — reason `preempted_by_placement` for placement
        loads, `displaced_by_load` for explicit ones, so displaced holders
        always learn via poll/409. The lease reads, checks, and revocation are
        sync with no await in between (invariant 11); the occupancy probe
        deliberately comes first.

        Returns the eviction-stats reason label for whatever this load
        displaces (the caller records it once the occupants are known).
        """
        reason = ("displaced_idle" if req.priority is not None
                  else "displaced_by_load")
        leases = getattr(self, "leases", None)
        if leases is None or not self.s.lease_enabled:
            return reason
        blocker = leases.blocks_unleased(self.cfg.id)
        if blocker is not None and blocker.holder != req.requested_by:
            raise RuntimeError(
                f"placement_conflict: {self.cfg.id} is held by {blocker.holder}'s "
                f"non-preemptible lease"
            )
        if req.priority is not None and await self._occupied():
            lease = leases.active_for(self.cfg.id)
            active = (time.time() - self.last_activity) \
                <= self.s.usage_active_window_s
            if active:
                if lease is None:
                    raise RuntimeError(
                        f"placement_conflict: {self.cfg.id}'s occupant became "
                        f"active since placement"
                    )
                if (not lease.preemptible or lease.priority >= req.priority
                        or not self.s.placement_preempt_active_enabled):
                    raise RuntimeError(
                        f"placement_conflict: {self.cfg.id}'s occupant is active "
                        f"and held at equal/higher priority"
                    )
                reason = "active_preempt"
            elif lease is not None and lease.preemptible \
                    and lease.holder != req.requested_by:
                if not self.s.placement_preempt_leased_idle_enabled:
                    raise RuntimeError(
                        f"placement_conflict: {self.cfg.id} is held by a lease"
                    )
                reason = "idle_preempt"
        revoked = leases.revoke_for_eviction(
            self.cfg.id, "",
            by=req.requested_by or ("placement" if req.priority is not None
                                    else "load"),
            reason=("preempted_by_placement" if req.priority is not None
                    else "displaced_by_load"),
            spare_holder=req.requested_by,
        )
        for l in revoked:
            self.jobs.log(job, f"revoked {l.holder}'s lease on {self.cfg.id} "
                               f"({l.revoked_reason})")
        # Whose claim this displaced, for the eviction stats. The lane's own
        # revocation is the only place that knows it — by the time _evict_all
        # returns the lease is already terminal.
        self._displaced_holders = [l.holder for l in revoked if l.holder]
        return reason

    def _note_evictions(self, req: LoadRequest, reason: str,
                        displaced: list[str]) -> None:
        """Record what this load displaced (usage stats; reload targets are
        filtered — swapping a model out for itself is not an eviction)."""
        stats = getattr(self, "stats", None)
        holders = getattr(self, "_displaced_holders", []) or []
        self._displaced_holders = []
        if stats is None:
            return
        for name in displaced:
            if not name or name == req.model:
                continue
            stats.note_eviction(
                unit=self.cfg.id, model=name, reason=reason,
                evicted_by=req.requested_by, incoming_model=req.model,
                incoming_priority=req.priority,
                holder=", ".join(holders))

    async def _evict_all(self, job: Job) -> list[str]:
        """Clear this lane's GPU: stop vLLM, unload all Ollama models, confirm
        freed. Returns the Ollama tags it unloaded (for the eviction stats —
        the vLLM occupant, if any, is known only to the caller's probe)."""
        # Unconditional stop — see unload(): the relay probe must not gate it.
        # (Skipped on an Ollama-only lane: two WSL round-trips for a unit that
        # does not exist, on every load.)
        if self.cfg.vllm_enabled:
            self.jobs.log(job, "stopping vLLM to free VRAM…")
            await self.vllm.stop()
        names = await self.ollama.unload_all()
        if names:
            self.jobs.log(job, f"unloaded Ollama: {', '.join(names)}")
        await self._wait_vram_free(job)
        return names

    async def _wait_vram_free(self, job: Optional[Job]) -> bool:
        """Block until this lane's card is back to driver baseline (the 100%-VRAM gate).

        If nvidia-smi can't see the card (off-box), don't block — return True.
        """
        deadline = time.monotonic() + self.s.evict_timeout_s
        while time.monotonic() < deadline:
            # No process list: this loop reads only the memory numbers, and the
            # second nvidia-smi spawn per tick was pure waste (~44 per wait).
            gpu = await query_gpu(self.s, uuid=self.cfg.gpu_uuid, with_processes=False)
            if not gpu.found:
                if job:
                    self.jobs.log(job, "nvidia-smi unavailable — skipping VRAM-free wait")
                return True
            if gpu.is_free(self.cfg.vram_free_baseline_mb):
                if job:
                    self.jobs.log(job, f"VRAM free ({gpu.used_mb} MiB used)")
                return True
            if job:
                self.jobs.log(job, f"waiting for VRAM to free… ({gpu.used_mb} MiB still used)")
            await asyncio.sleep(self.s.poll_interval_s)
        if job:
            self.jobs.log(job, "warning: VRAM did not return to baseline before timeout; continuing")
        return False

    async def _verify_ollama(self, job: Job, req: LoadRequest, remediate: bool) -> dict:
        await asyncio.sleep(0.5)
        gpu = await self._gpu()
        match = next((m for m in await self.ollama.loaded() if m.name == req.model), None)
        if match is None:
            raise RuntimeError(f"{req.model} is not loaded after the request (check Ollama logs)")

        on_cpu = max(0, match.size_bytes - match.size_vram_bytes)
        spilled = on_cpu > 0
        # "Premature spill": spilled while the card still has substantial free VRAM.
        premature = spilled and gpu.found and gpu.free_mb > 2 * self.cfg.vram_free_baseline_mb

        if premature and remediate:
            self.jobs.log(job, f"premature spill detected ({gpu.free_mb} MiB free) — attempting max-pack reload")
            packed = await self._max_pack_reload(job, req, gpu)
            if packed is not None:
                return packed
            # Fall through to report the original load — but re-verify it first:
            # if the max-pack path unloaded the model and its fallback restore
            # ALSO failed, the `match` above is stale and would report success
            # for a model that is no longer resident.
            if not any(m.name == req.model for m in await self.ollama.loaded()):
                raise RuntimeError(
                    f"max-pack reload of {req.model} failed AND the auto-fit "
                    f"restore failed — the model is no longer loaded")

        self.jobs.log(
            job,
            f"loaded {req.model}: {_gib(match.size_vram_bytes)} on GPU / "
            f"{_gib(on_cpu)} on CPU; VRAM {gpu.vram_pct}% used"
            + (" — WARNING premature spill" if premature else ""),
        )
        return LoadedModel(
            server="ollama",
            model=req.model,
            size_bytes=match.size_bytes,
            on_gpu_bytes=match.size_vram_bytes,
            on_cpu_bytes=on_cpu,
            spilled=spilled,
            fully_on_gpu=not spilled,
            gpu_vram_pct=gpu.vram_pct,
        ).model_dump()

    async def _max_pack_reload(self, job: Job, req: LoadRequest, gpu: GpuInfo) -> Optional[dict]:
        """Best-effort: force num_gpu to fill VRAM, then reload once. Falls back on OOM."""
        layers = await self.ollama.block_count(req.model)
        match = next((m for m in await self.ollama.loaded() if m.name == req.model), None)
        if not layers or match is None or match.size_bytes <= 0:
            self.jobs.log(job, "max-pack: layer count unknown; leaving Ollama auto-fit result")
            return None

        usable_mb = max(0, self.cfg.vram_total_mb - self.cfg.vram_free_baseline_mb)
        per_layer_bytes = match.size_bytes / layers
        target_layers = int((usable_mb * 1024 * 1024 * 0.9) / per_layer_bytes)
        target_layers = max(1, min(target_layers, layers))
        self.jobs.log(job, f"max-pack: reloading with num_gpu={target_layers}/{layers}")
        try:
            await self.ollama.unload(req.model)
            await self._wait_vram_free(job)
            await self.ollama.load(req.model, keep_alive=req.keep_alive, num_gpu=target_layers)
        except Exception as e:  # OOM or similar — recover with the plain auto-fit load
            self.jobs.log(job, f"max-pack reload failed ({e}); restoring auto-fit load")
            try:
                await self.ollama.load(req.model, keep_alive=req.keep_alive)
            except Exception:
                pass
            return None
        return await self._verify_ollama(job, req, remediate=False)

    def _vllm_result(self, served_name: str, gpu: GpuInfo) -> dict:
        return LoadedModel(
            server="vllm",
            model=served_name,
            gpu_vram_pct=gpu.vram_pct,
            fully_on_gpu=True,
        ).model_dump()


def _gib(n: int) -> str:
    return f"{n / (1024 ** 3):.1f} GiB"
