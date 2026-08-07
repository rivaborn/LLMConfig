"""Coordinator over every LLM unit.

Three kinds of unit exist and the orchestrator is what makes them interchangeable:

* a **`Lane`** (`lane.py`) — one local card arbitrated Ollama-XOR-vLLM, with the
  eviction-wait gate. The RTX 3090 (`primary`) and RTX 3070 Ti (`companion`).
* a **`SparkUnit`** (`spark_unit.py`) — one remote DGX Spark node driven by
  `sparkrun`. The node is the unit; no local GPU, no keepalive.
* a **`SparkGroup`** (`spark_group.py`) — a SET of Spark nodes serving one
  tensor-parallel model over the 200G fabric. SYNTHETIC: it participates in
  placement/leases/the gateway like any unit but has no UI tab or card
  (`settings.units()` never emits it) and is filtered out of `/api/status`
  lanes. Exists only when `SPARK_FABRIC_ENABLED` is on.

All satisfy the same duck-typed contract (see `UNIT_METHODS`), so routing,
status aggregation, defaults, and autoload are written once against `self.units`.
`self.lanes` still exposes only the GPU lanes, because the WSL keepalive and the
Ollama/vLLM back-compat shims are meaningless for a remote node.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional, Union

from .backends.ollama import OllamaBackend
from .backends.vllm import VllmBackend
from .config import Settings, SparkConfig
from .group_state import GroupPlacements, GroupState
from .idle import classify_usage
from .jobs import JobManager
from .gpu import GpuInfo, query_all_gpus
from .lane import Lane
from .lane_state import LaneDefaults, LanePins
from .slot_lane import SlotLane
from .registry import (DEFAULT_COMPANION_REGISTRY, DEFAULT_SPARK_CLUSTER_REGISTRY,
                       Registry, SparkRegistry)
from .schemas import (Job, LaneStatus, LoadRequest, StatusResponse, UnloadRequest,
                      boot_order_key)
from .spark_group import SparkGroup
from .spark_unit import SparkUnit
from .wsl import WslKeepalive

log = logging.getLogger(__name__)

Unit = Union[Lane, SparkUnit, SparkGroup]

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
        self.pins = LanePins(settings)
        self.units: dict[str, Unit] = {}
        # Set by attach_leases() / attach_monitor(); None = no lease awareness /
        # no util signal. Both are constructed WITH the orchestrator, so neither
        # can be a constructor arg.
        self.leases = None
        self.monitor = None

        for cfg in settings.lanes():
            # The primary lane reuses the registry the app already loaded; the companion
            # lane loads its own (small, 8 GB-friendly) catalog from its configured path.
            if cfg.id == "primary":
                reg = registry
            else:
                reg = Registry(cfg.registry_path, default_path=DEFAULT_COMPANION_REGISTRY)
            # A configured slot table turns the lane multi-model: SlotLane's
            # per-slot lifecycle replaces Lane's whole-card eviction gate.
            cls = SlotLane if cfg.vllm_slots else Lane
            self.units[cfg.id] = cls(settings, cfg, reg, jobs, self.keepalive)

        for scfg in settings.sparks():
            self.units[scfg.id] = SparkUnit(settings, scfg, SparkRegistry(scfg.registry_path), jobs)

        # Multi-node (SparkGroup) plumbing. The shared claim table and the
        # placement memory exist regardless of the fabric flag — the Cluster tab's
        # planner reads placements even while launches are gated — but GROUP UNITS
        # exist only when the fabric is up: with the flag off, orch.units is
        # byte-for-byte what it always was, so placement/status cannot change.
        self.group_state = GroupState()
        self.group_placements = GroupPlacements(settings.spark_group_state_path)
        self.cluster_registry = SparkRegistry(
            settings.spark_cluster_registry_path,
            default_path=DEFAULT_SPARK_CLUSTER_REGISTRY,
        ) if settings.spark_enabled else None
        for u in self.units.values():
            if isinstance(u, SparkUnit):
                u.group_state = self.group_state
        if settings.spark_enabled and settings.spark_fabric_enabled:
            # Re-instantiate a group per RECORDED node set, so a model that has
            # loaded on that set before is a standing auto-placement candidate
            # after a restart without anyone re-launching by hand first.
            for member_ids in self.group_placements.node_sets():
                try:
                    self.get_or_create_group(list(member_ids))
                except ValueError:
                    # A recorded set naming a node that is no longer configured —
                    # stale memory must not stop the app from starting.
                    continue

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
        return {k: v for k, v in self.units.items() if isinstance(v, (Lane, SlotLane))}

    @property
    def sparks(self) -> dict[str, SparkUnit]:
        return {k: v for k, v in self.units.items() if isinstance(v, SparkUnit)}

    @property
    def groups(self) -> dict[str, SparkGroup]:
        return {k: v for k, v in self.units.items() if isinstance(v, SparkGroup)}

    def get_or_create_group(self, member_ids: list[str]) -> SparkGroup:
        """The SparkGroup over `member_ids`, creating it on first use.

        Creation validates through `settings.spark_group_config` (raises
        ValueError on unknown/disabled members or fewer than two). Members are
        passed IN `orch.units` ORDER — that list is the lock-acquisition order
        every group load shares, which is what keeps overlapping group loads
        (spark1+spark2 vs spark2+spark3) from deadlocking.

        Requires the fabric flag: without it no group unit may exist, so the
        placer's candidate set stays byte-for-byte pre-multi-node.
        """
        if not self.s.spark_fabric_enabled:
            raise ValueError(
                "multi-node groups are disabled (SPARK_FABRIC_ENABLED=false)"
            )
        cfg = self.s.spark_group_config(member_ids)
        existing = self.units.get(cfg.id)
        if existing is not None:
            if not isinstance(existing, SparkGroup):  # id collision — impossible
                raise ValueError(f"'{cfg.id}' names a non-group unit")   # pragma: no cover
            return existing
        assert self.cluster_registry is not None  # spark_enabled implied by member validation
        members = [u for u in self.units.values()
                   if isinstance(u, SparkUnit) and u.cfg.id in cfg.member_ids]
        group = SparkGroup(self.s, cfg, members, self.cluster_registry,
                           self.group_state, self.group_placements, self.jobs)
        # Late joiners get the same attachments create_app fanned out at startup.
        if members and members[0].leases is not None:
            group.leases = members[0].leases
        if members and members[0].load_times is not None:
            group.load_times = members[0].load_times
        self.units[cfg.id] = group
        return group

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
            if isinstance(u, (Lane, SlotLane)):
                gpu = gpus.get(u.cfg.gpu_uuid) or GpuInfo(
                    found=False, uuid=u.cfg.gpu_uuid, error=f"GPU {u.cfg.gpu_uuid} not present"
                )
                return await u.status(gpu=gpu)
            return await u.status()

        # SparkGroups are deliberately EXCLUDED from /api/status: off-box consumers
        # switch on this payload (invariant 8) and the UI has no card for a group —
        # the members' own rows carry the residency (LoadedModel.group). The placer
        # is unaffected: it calls each unit's status() directly, never this.
        statuses: list[LaneStatus] = list(
            await asyncio.gather(*(_unit_status(u) for u in self.units.values()
                                   if not isinstance(u, SparkGroup)))
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

    async def autoload_skip_reason(self, unit: "Unit") -> str:
        """Why the boot autoload must leave this unit alone, or "" to proceed.

        The rule: load defaults only into a unit that is FREE or holds an IDLE
        model. Anything else is someone's live work, and the defaults are a
        convenience rather than a claim (see Settings.autoload_skip_busy).

        A failed status probe is treated as "leave it alone" on purpose: if we
        cannot see what a unit is holding we must not load over it, and a load
        would very likely fail anyway.
        """
        # A live lease is an explicit claim by a live holder — stronger than any
        # activity heuristic, and the reason it exists. Same convention as the
        # idle reaper (ANY lease blocks, not just non-preemptible ones): autoload
        # is a restore-my-defaults convenience, not a competing request.
        if self.leases is not None:
            held = self.leases.active_for(unit.cfg.id)
            if held is not None:
                return (f"leased by '{held.holder}' ({held.id}"
                        + (f", model {held.model}" if held.model else "") + ")")
        try:
            st = await unit.status()
        except Exception as e:  # noqa: BLE001 — unseeable unit: never load over it
            return f"status probe failed ({type(e).__name__}: {e})"
        if st.swap_in_progress or st.active_job_id:
            return "a swap is already in flight"
        if not st.loaded_models:
            return ""                       # free — the clean case
        util = (self.monitor.util_for(unit.cfg.gpu_uuid)
                if self.monitor is not None else None)
        if classify_usage(st, util, self.s) == "active":
            return ("holding " + ", ".join(m.model for m in st.loaded_models)
                    + " with recent activity")
        return ""                           # loaded but idle — displaceable

    async def autoload_defaults(self, *, skip_busy: Optional[bool] = None) -> list[Job]:
        """Fire (don't await) a load Job for every default on every enabled unit.

        One job PER MODEL: the unit's own lock serialises them, so a Spark asked for
        an embedder and a reranker loads them back-to-back rather than dropping all
        but the last. Within a unit the jobs are dispatched in `boot_order_key`
        order (needs_empty_node first, biggest budget next — the same rule
        cookbook-apply uses), NOT file order: the reranker's fastsafetensors
        loader kills a resident it launches beside (live, 2026-07-28), so a
        hand-edited lane_defaults.yaml listing it second must still load it first.
        Job-creation order is lock-acquisition order, and SparkUnit's own
        needs_empty_node refusal remains the backstop.

        By default (`Settings.autoload_skip_busy`) a unit that is ACTIVE,
        mid-swap, or leased is SKIPPED — this app restarts far more often than
        the fleet goes idle, and an unconditional restore turns every restart
        into an eviction event for whatever was actually working. `skip_busy`
        overrides the setting per call; False is the old unconditional behaviour.

        Async because the gate has to probe status. Skips are logged with their
        reason, so "my default didn't come back" is answerable from the log.
        """
        if skip_busy is None:
            skip_busy = self.s.autoload_skip_busy
        jobs: list[Job] = []
        for cfg in self.s.units():
            if not cfg.enabled:
                continue
            unit = self.unit(cfg.id)
            reg = getattr(unit, "registry", None)
            wanted = [d for d in self.defaults_for(cfg.id)
                      if d["server"] in ("ollama", "vllm", "spark") and d["model"]]
            if not wanted:
                continue                    # nothing to load — never probe
            if skip_busy:
                reason = await self.autoload_skip_reason(unit)
                if reason:
                    log.info("boot autoload: skipping %s — %s (defaults: %s)",
                             cfg.id, reason,
                             ", ".join(d["model"] for d in wanted))
                    continue
            wanted.sort(key=lambda d: boot_order_key(
                reg.get(d["model"]) if reg is not None else None))
            for d in wanted:
                req = LoadRequest(server=d["server"], model=d["model"], lane=cfg.id)
                jobs.append(unit.load(req))
        return jobs

    def attach_load_times(self, load_times) -> None:
        """Hand every unit the LoadTimes recorder (same fanout as attach_leases)."""
        for u in self.units.values():
            u.load_times = load_times

    def attach_monitor(self, monitor) -> None:
        """Hand the orchestrator the Monitor (same call-after-construction pattern
        as attach_leases). Only the boot autoload gate reads it, and only for the
        util signal that catches a client talking straight to a backend — with no
        Monitor the timestamps still classify (see idle.classify_usage)."""
        self.monitor = monitor

    def attach_leases(self, leases) -> None:
        """Hand every unit the LeaseManager for under-lock victim checks.

        Called from create_app AFTER the LeaseManager exists (it is constructed
        with the orchestrator, so the reference cannot be a constructor arg).
        Every unit kind now re-validates leases inside its own eviction path —
        Sparks in `_evict_victim`, groups in `placement_evict`, lanes in
        `_preempt_occupant_leases` — so the fanout is unconditional.
        """
        self.leases = leases
        for u in self.units.values():
            u.leases = leases

    def attach_stats(self, stats) -> None:
        """Hand every unit the UsageStats recorder (same fanout as attach_leases)."""
        for u in self.units.values():
            u.stats = stats

    async def aclose(self) -> None:
        """Close every unit's pooled HTTP clients (call on app shutdown)."""
        for u in self.units.values():
            await u.aclose()
