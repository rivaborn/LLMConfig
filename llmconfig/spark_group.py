"""A SET of DGX Spark nodes serving ONE tensor-parallel model — the multi-node unit.

A tp-K sparkrun job owns K whole nodes but serves /v1 from a single rank (the
HEAD — this group's first member). That asymmetry drives everything here:

- **Status** probes only the head's group port over HTTP (invariant 9 — never
  SSH on the status path); telemetry reuses the members' own cached samples.
- **Load** must hold EVERY member's swap lock before touching anything, acquired
  in the members' `orch.units` order — a deterministic global order, so two
  overlapping group loads (spark1+spark2 vs spark2+spark3) queue instead of
  deadlocking — each with the same bounded acquire a single unit uses, and
  all-or-nothing: a failed acquire releases the prefix and raises.
- **Residency on the members** is expressed through the shared `GroupState`: a
  live claim makes each member report the model on its card (`LoadedModel.group`)
  and refuse its own loads/unloads — the containers really are on those nodes,
  even though only the head answers HTTP.

The group is SYNTHETIC: it lives in `orch.units` so placement, leases, and the
/v1 gateway treat it like any unit, but it has no UI tab or Home card
(`settings.units()` never emits it) and is filtered out of `/api/status` lanes.
It exists only while `SPARK_FABRIC_ENABLED` is on.

Duck-typed unit surface (see `UNIT_METHODS`): status / load / unload / touch /
aclose, plus the optional `canonical_model` / `idle_for` / `declared_budgets`
the placer and lease manager probe for — `declared_budgets` deliberately, so a
group quacks like a Spark and rides the existing spark-shaped code paths.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

from .config import Settings, SparkGroupConfig
from .gpu import GpuInfo
from .group_state import GroupPlacements, GroupState
from .jobs import JobManager
from .registry import SparkRegistry
from .schemas import (GpuOut, Job, LaneStatus, LoadedModel, LoadRequest,
                      ServedModel, UnloadRequest)
from .spark_unit import SparkUnit


class SparkGroup:
    kind = "spark_group"

    def __init__(
        self,
        settings: Settings,
        cfg: SparkGroupConfig,
        members: list[SparkUnit],
        registry: SparkRegistry,          # the CLUSTER catalog (multi-node recipes)
        group_state: GroupState,
        placements: GroupPlacements,
        jobs: JobManager,
    ):
        self.s = settings
        self.cfg = cfg
        # Members arrive in orch.units order — the lock-acquisition order every
        # group must share (see module docstring). The head is the first one.
        self.members = members
        self.head = members[0]
        self.registry = registry
        self.group_state = group_state
        self.placements = placements
        self.jobs = jobs
        self.leases = None        # attached by Orchestrator.attach_leases
        self.load_times = None    # attached by Orchestrator.attach_load_times
        self._lock = asyncio.Lock()
        self._active_job_id: Optional[str] = None
        self.last_activity: float = time.time()
        self.model_activity: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Shared-with-SparkUnit surface (same semantics, cluster catalog)
    # ------------------------------------------------------------------ #
    def canonical_model(self, model: str) -> str:
        entry = self.registry.get(model) or self.registry.find_by_served_name(model)
        return entry.alias if entry else model

    def touch(self, ts: float | None = None, model: str | None = None) -> None:
        now = time.time() if ts is None else ts
        self.last_activity = max(self.last_activity, now)
        if model:
            key = self.canonical_model(model)
            self.model_activity[key] = max(self.model_activity.get(key, 0.0), now)

    def idle_for(self, model: str) -> float:
        return time.time() - self.model_activity.get(self.canonical_model(model), self.last_activity)

    def declared_budgets(self, names) -> dict[str, float]:
        """Spark-shaped duck-type for the placer. Multi-node entries carry no
        mem_fraction (a tp job claims every member whole), so this reports 0.0 —
        which placement already reads as a whole-node claim. Kept as a method so
        `hasattr(unit, "declared_budgets")` marks the group spark-like."""
        out: dict[str, float] = {}
        for name in names:
            if not name:
                continue
            e = self.registry.find_by_served_name(name)
            out[name] = e.mem_fraction if e else 0.0
        return out

    @property
    def member_hosts(self) -> list[str]:
        return [m.cfg.host for m in self.members]

    # ------------------------------------------------------------------ #
    # Status — HTTP probe of the head's group port ONLY (invariant 9)
    # ------------------------------------------------------------------ #
    async def status(self, gpu: GpuInfo | None = None) -> LaneStatus:
        # Ride the HEAD member's probe breaker instead of keeping a second one:
        # the group's port IS one of the head's slot ports, so "the head's slots
        # all timed out recently" is exactly the evidence that this probe would
        # burn its timeout too. Without this, a powered-off head cost the full
        # connect timeout on every placer sweep even while the member's own
        # status had already backed off (invariant 9's reasoning).
        head_down = (self.head._served_fails >= self.head._fails_before_backoff
                     and time.time() - self.head._served_ts < self.head._probe_backoff_s)
        served = (ServedModel() if head_down
                  else await self.head.spark.served_info(self.cfg.api_port))
        entry = (self.registry.find_by_served_name(served.name)
                 if served.name else None)

        claim = self.group_state.get(self.cfg.id)
        if entry is not None and claim is None:
            # Restart recovery: claims are in-memory, but the deployment outlived
            # the restart — the head is serving a CLUSTER-catalog model on the
            # group port. Re-claim so the members go back to reporting/refusing.
            # Best-effort: an overlap (another group already claimed a member)
            # means the world is inconsistent; status must not raise over it.
            try:
                claim = self.group_state.claim(
                    self.cfg.id, served.name or "", entry.alias,
                    self.cfg.member_ids, self.cfg.api_port,
                )
            except RuntimeError:
                claim = None
        resident = entry is not None and claim is not None

        # No telemetry of its own: reuse the members' cached samples (their own
        # status polls keep them fresh) — worst node-wide occupancy is the
        # honest single figure for a job that spans them all.
        cached = [m._gpu_cached for m in self.members if m._gpu_cached is not None]
        remote_gpu = (max(cached, key=lambda g: g.vram_pct) if cached
                      else GpuInfo(found=False, uuid=self.cfg.gpu_uuid,
                                   error="awaiting members' telemetry"))

        loaded_models: list[LoadedModel] = []
        if resident:
            loaded_models.append(LoadedModel(
                server="spark",
                model=served.name or "",
                root=served.root,
                context_len=served.context_len,
                gpu_vram_pct=remote_gpu.vram_pct,
                fully_on_gpu=True,
                port=self.cfg.api_port,
                group=self.cfg.id,
            ))
            if self.model_activity is not None and entry is not None:
                # Prune clocks for anything that is no longer this group's model.
                for gone in [k for k in self.model_activity if k != entry.alias]:
                    self.model_activity.pop(gone, None)

        loaded = loaded_models[0] if loaded_models else None
        reachable = bool(served.name) or remote_gpu.found
        return LaneStatus(
            id=self.cfg.id,
            name=self.cfg.name,
            kind="spark_group",
            host=",".join(self.member_hosts),
            reachable=reachable,
            enabled=self.cfg.enabled,
            owner="spark" if loaded_models else ("free" if reachable else "unknown"),
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
    # Load — ordered member locks, whole-node eviction, multi-host launch
    # ------------------------------------------------------------------ #
    def load(self, req: LoadRequest) -> Job:
        job = self.jobs.create(kind=f"load:{self.cfg.id}:spark:{req.model}")

        async def body(job: Job) -> dict:
            if self._lock.locked():
                self.jobs.log(job, "waiting for an in-progress group swap to finish… "
                                   f"(holder: {self._active_job_id or 'unknown'})")
            await self._acquire_swap_lock("load")
            self._active_job_id = job.id
            try:
                return await self._load(job, req)
            finally:
                self._active_job_id = None
                self.touch(model=req.model)
                self._lock.release()

        return self.jobs.start(job, body)

    async def _load(self, job: Job, req: LoadRequest) -> dict:
        if not self.s.spark_fabric_enabled:
            raise RuntimeError(
                "multi-node launches are disabled: the 200G fabric is not installed "
                "(SPARK_FABRIC_ENABLED=false). The Cluster tab stays a planner until "
                "the switch lands and `sparkrun setup cx7` has run."
            )
        entry = self.registry.get(req.model)
        if entry is None:
            raise RuntimeError(
                f"unknown multi-node model '{req.model}' (see GET /api/cluster/models — "
                f"multi-node recipes live in the cluster catalog, not per-node ones)"
            )
        if entry.status == "blocked" and not req.force:
            raise RuntimeError(
                f"'{req.model}' is blocked: {entry.notes}. Re-issue with force=true to try anyway."
            )
        k = len(self.members)
        if not (entry.min_nodes <= k <= entry.max_nodes):
            raise RuntimeError(
                f"'{req.model}' supports {entry.min_nodes}-{entry.max_nodes} nodes; "
                f"{self.cfg.id} has {k}"
            )

        target = entry.served_name or entry.alias
        recipe = entry.recipe or entry.alias

        # A member held by ANOTHER group means live containers this load would
        # collide with — even though that member's own slots read empty (only a
        # head answers HTTP). Fast-fail here; re-validated under the locks below.
        for m in self.members:
            c = self.group_state.claim_for(m.cfg.id)
            if c is not None and c.group_id != self.cfg.id:
                raise RuntimeError(
                    f"placement_conflict: {m.cfg.id} is claimed by {c.group_id} "
                    f"({c.model}) — unload that deployment first (/api/cluster/unload)"
                )

        held: list[SparkUnit] = []
        try:
            # Member locks in orch.units order (the members list IS that order) —
            # the global acquisition order that makes overlapping group loads
            # queue rather than deadlock. Bounded like every unit lock
            # (invariant 17); a failed acquire releases the prefix via `held`.
            for m in self.members:
                if m._lock.locked():
                    self.jobs.log(job, f"waiting for {m.cfg.id}'s in-progress swap… "
                                       f"(holder: {m._active_job_id or 'unknown'})")
                await m._acquire_swap_lock("group load")
                held.append(m)
                m._active_job_id = job.id

            # RE-validate the claims now that the locks are held: another group
            # sharing a member may have loaded while we queued on its lock (the
            # pre-check above is a fast-fail, not a guarantee — same snapshot
            # discipline as _evict_victim).
            for m in self.members:
                c = self.group_state.claim_for(m.cfg.id)
                if c is not None and c.group_id != self.cfg.id:
                    raise RuntimeError(
                        f"placement_conflict: {m.cfg.id} was claimed by "
                        f"{c.group_id} ({c.model}) while this load waited"
                    )
            own_claim = self.group_state.get(self.cfg.id)

            # Reloading this group's own model: tear the old job down first —
            # VERIFIED gone, because relaunching over a lying stop would let
            # wait_ready see the OLD ranks and report a reload that never
            # happened (the same trap SparkUnit's reload path guards against).
            if own_claim is not None:
                if own_claim.model == target and not req.force:
                    self.jobs.log(job, f"{self.cfg.name} already serving {target}")
                    return self._result(target, self.cfg.api_port)
                self.jobs.log(job, f"stopping {own_claim.model} to reload…")
                await self._stop_current(job, own_claim.alias, verify_gone=True)

            # Whole-node semantics across every member: everything resident must
            # go, and every eviction is re-validated under the member's lock
            # (still resident / unleased / idle) — a failed re-validation raises
            # `placement_conflict:` so the gateway re-places instead of blaming
            # the model (same protocol as a single Spark).
            requested = list(dict.fromkeys(req.evict or []))
            for m in self.members:
                slots = await m.spark.served_slots()
                resident = {sm.name: port for port, sm in slots.items() if sm.name}
                if not resident:
                    continue
                victims = list(dict.fromkeys(
                    requested + [m.canonical_model(name) for name in resident]
                ))
                self.jobs.log(job, f"freeing {m.cfg.id} "
                                   f"({', '.join(sorted(resident))})…")
                for victim in victims:
                    await m._evict_victim(job, victim, slots, resident)
                if resident:
                    raise RuntimeError(
                        f"placement_conflict: {m.cfg.id} still holds "
                        f"{', '.join(sorted(resident))} after eviction"
                    )
                m.model_activity.clear()

            hosts = self.member_hosts
            self.jobs.log(
                job,
                f"launching {recipe} across {', '.join(self.cfg.member_ids)} as "
                f"'{target}' (tp={k}, head {hosts[0]}:{self.cfg.api_port})…",
            )
            launch_started = time.monotonic()

            def _launch_failed(msg: str) -> RuntimeError:
                # Post-admission launch failures feed placement's blocklist, keyed
                # per GROUP (a 2-node failure says nothing about the 4-node set —
                # and with the fabric down every multi-node launch fails, which is
                # exactly the state the blocklist should latch). rc 127 is an
                # environment fault, not a model fault, and doesn't count.
                if self.load_times is not None:
                    from .load_times import fail_key
                    self.load_times.record_failure(
                        fail_key(self.cfg.id, "spark", entry.alias))
                return RuntimeError(msg)

            timeout = float(entry.load_timeout_s or self.cfg.load_timeout_s)
            r = await self.head.spark.run_recipe(
                recipe, tp=k, extra=entry.extra_args, served=target,
                port=self.cfg.api_port, timeout=timeout, hosts=hosts,
            )
            if not r.ok:
                if r.rc == 127:
                    raise RuntimeError(
                        "sparkrun not found in WSL — install it on the control node "
                        "(`uvx sparkrun setup install`) or fix SPARK_RUN_MULTI_CMD"
                    )
                if r.rc == 124:
                    raise _launch_failed(
                        f"sparkrun timed out launching '{recipe}' after its load_timeout_s "
                    f"budget: {r.text()[:300]} — if the launch was still "
                    f"DISTRIBUTING the model, the upstream repo has likely moved "
                    f"since staging and sparkrun is re-downloading the delta "
                    f"(possibly whole new format dirs); no budget survives that. "
                    f"Re-stage the weights first, then relaunch — and check the "
                    f"head for an ORPHANED `hf download` this kill leaves behind.")
                raise _launch_failed(
                    f"sparkrun failed to launch '{recipe}' (rc={r.rc}): {r.text()[:400]}")
            for line in (r.out or "").splitlines()[-5:]:
                if line.strip():
                    self.jobs.log(job, line.strip())

            self.jobs.log(
                job,
                f"waiting up to {int(timeout)}s for {target} to answer on "
                f"{self.cfg.api_base_for(self.cfg.api_port)}…",
            )
            ok = await self.head.spark.wait_ready(
                target, timeout, on_log=lambda l: self.jobs.log(job, l),
                port=self.cfg.api_port,
            )
            if not ok:
                tail = await self.head.spark.logs(n=25, port=self.cfg.api_port)
                raise _launch_failed(
                    f"{self.cfg.name} did not serve '{target}' — a multi-node launch "
                    f"also fails when the fabric is down or NCCL cannot form the ring "
                    f"(head log tail below).\n{tail}"
                )

            # Success: record the claim (members start reporting/refusing), the
            # placement memory (this node set now feeds auto-placement and
            # restart re-instantiation), and the launch duration under a
            # per-node-count key — a 2-node cold start is not a 4-node one.
            self.group_state.claim(self.cfg.id, target, entry.alias,
                                   self.cfg.member_ids, self.cfg.api_port)
            self.placements.record(entry.alias, self.cfg.member_ids)
            if self.load_times is not None:
                from .load_times import fail_key, spark_key
                # Per NODE COUNT: a 2-node cold start is not a 4-node one, so they
                # must not share a median (placement's tier-3 est_bucket reads it).
                self.load_times.record(f"{spark_key(entry.alias)}:x{k}",
                                       time.monotonic() - launch_started,
                                       unit=self.cfg.id)
                self.load_times.clear_failures(
                    fail_key(self.cfg.id, "spark", entry.alias))
            self.jobs.log(job, f"{self.cfg.name} serving {target} on "
                               f"{hosts[0]}:{self.cfg.api_port} across {k} nodes")
            return self._result(target, self.cfg.api_port)
        finally:
            for m in reversed(held):
                m._active_job_id = None
                m._lock.release()

    # ------------------------------------------------------------------ #
    # Unload
    # ------------------------------------------------------------------ #
    async def unload(self, req: UnloadRequest) -> LaneStatus:
        await self._acquire_swap_lock("unload")
        held: list[SparkUnit] = []
        try:
            self._active_job_id = None
            for m in self.members:
                await m._acquire_swap_lock("group unload")
                held.append(m)
            claim = self.group_state.get(self.cfg.id)
            alias = None
            if claim is not None:
                alias = claim.alias
            elif req.model:
                e = self.registry.get(req.model) or self.registry.find_by_served_name(req.model)
                alias = e.alias if e else None
            else:
                # No claim (restart) and no name given — ask the head what it serves.
                served = (await self.head.spark.served_info(self.cfg.api_port)).name
                if served:
                    e = self.registry.find_by_served_name(served)
                    alias = e.alias if e else None
            await self._stop_current(None, alias)
            self.model_activity.clear()
        finally:
            for m in reversed(held):
                m._lock.release()
            self._lock.release()
        return await self.status()

    async def _stop_current(self, job: Optional[Job], alias: Optional[str],
                            verify_gone: bool = False) -> None:
        """Stop the group's job by sparkrun JOB ID (cluster-wide — one id tears
        down every rank), release the claim, then RE-PROBE the head: `sparkrun
        stop` swallows SSH failures into rc=0, so the probe is the only proof.

        `verify_gone` (the RELOAD path) polls the head until the old server
        actually stops answering and RAISES if it never does — launching over a
        surviving job would let wait_ready see the old ranks and report a reload
        that never happened. A plain unload keeps the single-probe warning:
        status() shows the truth and there is nothing about to be launched over
        it (parity with SparkUnit.unload)."""
        entry = self.registry.get(alias) if alias else None
        recipe = (entry.recipe or entry.alias) if entry else None
        if recipe:
            r = await self.head.spark.stop(recipe=recipe,
                                           any_host_of=set(self.member_hosts))
            if job is not None and not r.ok and r.rc not in (0, 1):
                self.jobs.log(job, f"stop reported rc={r.rc}: {r.text()[:200]}")
        self.group_state.release(self.cfg.id)
        if verify_gone:
            for _ in range(15):
                if not (await self.head.spark.served_info(self.cfg.api_port)).name:
                    return
                await asyncio.sleep(1.0)
            raise RuntimeError(
                f"the previous deployment is still serving on "
                f"{self.cfg.api_base_for(self.cfg.api_port)} 15s after the stop — "
                f"sparkrun stop silently failed; not relaunching over it"
            )
        still = (await self.head.spark.served_info(self.cfg.api_port)).name
        if still and job is not None:
            self.jobs.log(job, f"⚠ head still serving {still} after stop — "
                               f"check `sparkrun status` (stop's rc is not trustworthy)")

    async def _acquire_swap_lock(self, what: str) -> None:
        try:
            await asyncio.wait_for(self._lock.acquire(),
                                   timeout=self.s.swap_wait_timeout_s)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"timed out after {self.s.swap_wait_timeout_s:.0f}s waiting for the "
                f"{self.cfg.id} group lock to {what} (held by job "
                f"{self._active_job_id or 'unknown'}) — the group is wedged, not busy"
            ) from None

    def _result(self, served_name: str, port: int) -> dict:
        cached = [m._gpu_cached for m in self.members if m._gpu_cached is not None]
        vram = max((g.vram_pct for g in cached), default=0.0)
        return LoadedModel(
            server="spark",
            model=served_name,
            gpu_vram_pct=vram,
            fully_on_gpu=True,
            port=port,
            group=self.cfg.id,
        ).model_dump()

    async def aclose(self) -> None:
        # Nothing owned: the HTTP clients belong to the members' backends, which
        # close themselves; the claim deliberately survives (it mirrors reality —
        # the containers keep serving without LLMConfig).
        return None
