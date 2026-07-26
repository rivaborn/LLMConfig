"""Auto-placement — pick the unit for a model when the client names none.

Before this, every `/v1` request carried the scheduler in its headers: the client
chose the unit (`X-LLM-Lane`, defaulting to `primary`) and a naive client 404'd
whenever its model didn't live on the 3090. The fleet now has six units with
budgets, leases, and per-model idle clocks — everything needed to decide
placement server-side.

Two halves, split for testability:

* **`rank()`** — a pure function from per-unit facts to a `Decision`. No I/O, no
  clocks; everything time-dependent arrives pre-computed in `CandidateFacts`.
* **`Placer`** — gathers those facts: one concurrent status sweep (single-flight
  with a short TTL, so a burst of auto requests costs one sweep), sync lease
  reads, sync registry resolution, and the Monitor's latest util sample.

## Placement is ADVISORY — the load-bearing property of this module

The ranking runs on a snapshot; the world moves. Safety never depends on the
choice being right: `SparkUnit._admit` (declared + measured budgets, under the
unit lock), the lane lock, and the gateway's lease gate are the real gates, and
eviction victims are re-validated under the unit's own lock (`_evict_victim`).
A stale snapshot can only mis-rank — the gateway then gets a refusal it can
answer with one re-place (`exclude=` the failed unit).

## The ranking

1. **Sole candidate → `Pin`.** When a model resolves on exactly one unit, auto
   degrades to what an explicit header would do, and that unit's own load
   semantics apply — including a lane's last-writer-wins eviction. This is what
   keeps a primary-only deployment behaving exactly as before auto-placement,
   and it bypasses every predicate below on purpose: with no alternative there
   is no choice to make, so the protections that govern *choices* don't apply.
2. **Resident** (and not lease-refused): prefer `idle` over `active`; stable
   tie-break in unit order, so repeated requests stick to one unit and its
   warm prefix cache.
3. **Fits without displacement**: a Spark whose declared budgets leave room
   (and has a free slot, and no unbudgeted resident), or a free lane. Emptiest
   first.
4. **Fits with displacement** of idle+unleased models only:
   - a lane whose occupant is idle and unleased — victims stay EMPTY, because
     `Lane.load` evicts its occupant itself (that *is* its load semantics);
   - a Spark where stopping the stalest idle unleased co-tenants frees enough
     declared budget — fewest victims, then stalest. A `tp > 1` or
     `mem_fraction 0.0` entry claims the whole node: only an empty node, or one
     where EVERY resident is an eligible victim, qualifies.
   Active models and models with ANY live lease are never victims (matching the
   idle reaper's convention — even a preemptible lease shields, because
   placement is an optimisation, not a competing claimant).
5. Nothing admissible → `NoCapacity`, one reason per unit, so the 503 explains
   itself. A model in no catalog at all → `NotFound` (the gateway's existing
   404 — never collapsed into a 503).

`swap_in_progress` deprioritizes a unit but never disqualifies it — the load
path already coalesces onto identical in-flight loads.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .config import Settings
from .idle import classify_usage
from .schemas import LaneStatus

if TYPE_CHECKING:
    from .leases import LeaseManager
    from .monitor import Monitor
    from .orchestrator import Orchestrator, Unit

AUTO = "auto"


def wants_auto(lane_header: Optional[str]) -> bool:
    """Absent, empty, or the literal 'auto' (any case) → auto-place."""
    return not lane_header or lane_header.strip().lower() == AUTO


# --------------------------------------------------------------------------- #
# Facts
# --------------------------------------------------------------------------- #
@dataclass
class ResidentFact:
    """One resident model on a unit, with everything victim-selection needs."""

    model: str            # served name (as residency reports it)
    alias: str            # canonical catalog alias
    budget: float         # declared mem_fraction (0.0 = unbudgeted/whole-node)
    idle_s: float         # per-model idle clock (unit clock for lanes)
    leased: bool          # ANY live lease (unit-wide or naming this model)


@dataclass
class CandidateFacts:
    """Everything `rank()` may consider about one unit. Pre-computed — no I/O."""

    unit_id: str
    kind: str                              # "gpu" | "spark"
    status: LaneStatus
    usage: str                             # free | idle | active (classify_usage)
    server: str                            # how the model would load here
    load_arg: str                          # alias/tag passed to LoadRequest
    residents: list[ResidentFact] = field(default_factory=list)
    committed: float = 0.0                 # Spark: declared budgets incl. in-flight load
    want: float = 0.0                      # Spark: the target model's mem_fraction
    whole_node: bool = False               # tp>1 or mem_fraction 0.0
    free_slot: bool = True                 # Spark: a port is available
    lease_refused: bool = False            # gateway lease gate would 409 this model
    order: int = 0                         # settings.units() position (tie-break)


@dataclass
class Decision:
    """What placement decided. Exactly one of the shapes below."""

    outcome: str                           # "pin" | "place" | "no_capacity" | "not_found"
    unit_id: str = ""
    server: str = ""
    load_arg: str = ""
    victims: list[str] = field(default_factory=list)   # canonical aliases to evict
    reasons: dict[str, str] = field(default_factory=dict)  # unit -> why not (no_capacity)


def _fits_without_displacement(c: CandidateFacts, headroom: float) -> bool:
    if c.kind != "spark":
        return c.usage == "free"
    if not c.free_slot:
        return False
    if c.whole_node:
        return not c.residents
    if any(r.budget <= 0.0 for r in c.residents):
        return False  # an unbudgeted resident claims the whole node
    return c.committed + c.want <= headroom + 1e-9


def _victims_for(c: CandidateFacts, headroom: float,
                 active_window_s: float) -> Optional[list[ResidentFact]]:
    """Cheapest eligible eviction set that makes the model fit, or None."""
    eligible = sorted(
        (r for r in c.residents
         if not r.leased and r.idle_s > active_window_s),
        key=lambda r: -r.idle_s,           # stalest first
    )
    if c.whole_node:
        # Whole-node claim: every resident must go, and every one must be eligible.
        if c.residents and len(eligible) == len(c.residents):
            return list(eligible)
        return None
    victims: list[ResidentFact] = []
    committed = c.committed
    # An unbudgeted resident poisons the sum regardless of arithmetic — it claims
    # the whole node — so it goes first or the unit is out.
    bad = [r for r in c.residents if r.budget <= 0.0]
    if bad:
        if any(r not in eligible for r in bad):
            return None
        for r in bad:
            victims.append(r)
            committed -= r.budget

    def fits() -> bool:
        return (committed + c.want <= headroom + 1e-9
                and c.free_slot_after(len(victims)))

    for r in (x for x in eligible if x not in victims):
        if fits():
            break
        victims.append(r)
        committed -= r.budget
    return victims if fits() else None


def _lane_evictable(c: CandidateFacts, active_window_s: float) -> bool:
    """A lane candidate whose occupant may be displaced (Lane.load does the evicting)."""
    if c.kind == "spark":
        return False
    if c.usage == "free":
        return True
    occ = c.residents[0] if c.residents else None
    return c.usage == "idle" and occ is not None and not occ.leased


def rank(model: str, candidates: list[CandidateFacts], settings: Settings,
         *, exclude: frozenset[str] = frozenset()) -> Decision:
    """Pure ranking — see the module docstring for the tiers."""
    cands = [c for c in candidates if c.unit_id not in exclude]
    if not cands:
        return Decision(outcome="not_found")

    # Tier 1 — sole candidate pins, bypassing every predicate below.
    if len(cands) == 1:
        c = cands[0]
        return Decision(outcome="pin", unit_id=c.unit_id, server=c.server,
                        load_arg=c.load_arg)

    def order_key(c: CandidateFacts):
        return (c.status.swap_in_progress, c.order)   # mid-swap deprioritized, never out

    # Tier 2 — resident, not lease-refused; idle beats active; stable tie-break.
    resident = [c for c in cands
                if not c.lease_refused
                and any(r.model == model for r in c.residents)]
    if resident:
        resident.sort(key=lambda c: (c.usage != "idle", *order_key(c)))
        c = resident[0]
        return Decision(outcome="place", unit_id=c.unit_id, server=c.server,
                        load_arg=c.load_arg)

    headroom = settings.spark_mem_headroom
    window = settings.usage_active_window_s

    # Tier 3 — fits without displacing anything; emptiest first.
    fits = [c for c in cands if not c.lease_refused
            and _fits_without_displacement(c, headroom)]
    if fits:
        # Emptiest first (mid-swap still deprioritized ahead of everything).
        fits.sort(key=lambda c: (c.status.swap_in_progress,
                                 c.committed if c.kind == "spark"
                                 else (c.status.gpu.vram_pct if c.status.gpu else 0.0),
                                 c.order))
        c = fits[0]
        return Decision(outcome="place", unit_id=c.unit_id, server=c.server,
                        load_arg=c.load_arg)

    # Tier 4 — fits by displacing idle + unleased models only.
    displaceable: list[tuple[CandidateFacts, list[ResidentFact]]] = []
    for c in cands:
        if c.lease_refused:
            continue
        if c.kind == "spark":
            v = _victims_for(c, headroom, window)
            if v is not None:
                displaceable.append((c, v))
        elif _lane_evictable(c, window):
            displaceable.append((c, []))   # Lane.load evicts on its own
    if displaceable:
        displaceable.sort(key=lambda cv: (len(cv[1]),
                                          -(cv[1][0].idle_s if cv[1] else 1e18),
                                          *order_key(cv[0])))
        c, victims = displaceable[0]
        return Decision(outcome="place", unit_id=c.unit_id, server=c.server,
                        load_arg=c.load_arg, victims=[r.alias for r in victims])

    # Tier 5 — nothing admissible: say why, per unit.
    reasons = {}
    for c in cands:
        if c.lease_refused:
            reasons[c.unit_id] = "held by a non-preemptible lease"
        elif c.kind == "spark" and not c.free_slot:
            reasons[c.unit_id] = "all slots in use"
        elif c.kind == "spark" and any(r.budget <= 0.0 for r in c.residents):
            reasons[c.unit_id] = "an unbudgeted resident claims the whole node"
        elif c.kind == "spark":
            reasons[c.unit_id] = (f"committed {c.committed:.2f} + {c.want:.2f} "
                                  f"exceeds {headroom:.2f}; no idle unleased co-tenant frees enough")
        else:
            reasons[c.unit_id] = "occupant is active or leased"
    return Decision(outcome="no_capacity", reasons=reasons)


# Give CandidateFacts a small helper used by _victims_for (defined after the
# dataclass so the dataclass body stays declarative).
def _free_slot_after(self: CandidateFacts, n_victims: int) -> bool:
    """Would a slot be open after evicting n victims?"""
    if self.kind != "spark":
        return True
    return self.free_slot or n_victims > 0


CandidateFacts.free_slot_after = _free_slot_after


# --------------------------------------------------------------------------- #
# Placer — gathers facts, delegates the decision to rank()
# --------------------------------------------------------------------------- #
class Placer:
    def __init__(self, settings: Settings, orch: "Orchestrator",
                 leases: "LeaseManager", monitor: "Monitor"):
        self.s = settings
        self.orch = orch
        self.leases = leases
        self.monitor = monitor
        # Single-flight status sweep: one in-flight task shared by every awaiter,
        # then cached for placement_cache_ttl_s. Staleness only mis-ranks (see
        # module docstring); freshness costs a full-fleet probe per request.
        self._sweep_task: Optional[asyncio.Task] = None
        self._sweep_ts: float = 0.0
        self._sweep_result: dict[str, LaneStatus] = {}

    # ---- status sweep ----
    async def _statuses(self) -> dict[str, LaneStatus]:
        now = time.monotonic()
        if self._sweep_result and now - self._sweep_ts < self.s.placement_cache_ttl_s:
            return self._sweep_result
        if self._sweep_task is None or self._sweep_task.done():
            self._sweep_task = asyncio.create_task(self._sweep())
        return await asyncio.shield(self._sweep_task)

    async def _sweep(self) -> dict[str, LaneStatus]:
        units = list(self.orch.units.values())
        results = await asyncio.gather(*(u.status() for u in units),
                                       return_exceptions=True)
        out: dict[str, LaneStatus] = {}
        for u, res in zip(units, results):
            if isinstance(res, LaneStatus):
                out[u.cfg.id] = res
            # a raising unit simply isn't a candidate this round
        self._sweep_result, self._sweep_ts = out, time.monotonic()
        return out

    # ---- resolution (mirrors gateway.resolve, minus the URL) ----
    async def _resolve(self, unit: "Unit", model: str) -> Optional[tuple[str, str, object]]:
        """(server, load_arg, spark_entry|None) if `model` can run on `unit`."""
        if hasattr(unit, "declared_budgets"):          # SparkUnit
            for e in unit.registry.entries():
                if (e.served_name or e.alias) == model and e.status != "blocked":
                    return ("spark", e.alias, e)
            return None
        for e in unit.registry.entries():
            if (e.served_name or e.alias) == model and e.status != "blocked":
                return ("vllm", e.alias, None)
        if ":" in model:
            try:
                names = {m.name for m in await unit.ollama.list_models()}
            except Exception:  # noqa: BLE001 — a down Ollama just isn't a candidate
                names = set()
            if model in names:
                return ("ollama", model, None)
        return None

    def _inflight_budget(self, unit: "Unit", st: LaneStatus) -> float:
        """Declared budget of a load already in flight on a Spark (job kind
        `load:{unit}:spark:{alias}`), so a burst doesn't over-place."""
        if not st.active_job_id or not hasattr(unit, "declared_budgets"):
            return 0.0
        job = self.orch.jobs.get(st.active_job_id)
        if job is None or not job.kind.startswith(f"load:{unit.cfg.id}:spark:"):
            return 0.0
        alias = job.kind.rsplit(":", 1)[-1]
        entry = unit.registry.get(alias)
        return entry.mem_fraction if entry else 0.0

    def _facts_for(self, unit: "Unit", st: LaneStatus, server: str, load_arg: str,
                   entry, order: int, model: str) -> CandidateFacts:
        uid = unit.cfg.id
        usage = classify_usage(st, self.monitor.util_for(unit.cfg.gpu_uuid), self.s)
        is_spark = hasattr(unit, "declared_budgets")
        residents: list[ResidentFact] = []
        if is_spark:
            budgets = unit.declared_budgets([m.model for m in st.loaded_models])
            for m in st.loaded_models:
                alias = unit.canonical_model(m.model)
                residents.append(ResidentFact(
                    model=m.model,
                    alias=alias,
                    budget=budgets.get(m.model, 0.0),
                    idle_s=unit.idle_for(m.model),
                    leased=self.leases.active_for(uid, alias) is not None,
                ))
        else:
            for m in st.loaded_models:
                residents.append(ResidentFact(
                    model=m.model, alias=m.model, budget=0.0,
                    idle_s=st.idle_s or 0.0,
                    leased=self.leases.active_for(uid) is not None,
                ))
        committed = sum(r.budget for r in residents) + self._inflight_budget(unit, st)
        want = entry.mem_fraction if entry is not None else 0.0
        whole_node = bool(entry is not None and (entry.tp > 1 or want <= 0.0)) if is_spark else False
        free_slot = (len(st.loaded_models) < getattr(unit.cfg, "max_models", 1)) if is_spark else True
        return CandidateFacts(
            unit_id=uid, kind="spark" if is_spark else "gpu", status=st, usage=usage,
            server=server, load_arg=load_arg, residents=residents,
            committed=committed, want=want, whole_node=whole_node,
            free_slot=free_slot,
            lease_refused=self.leases.blocks_unleased(uid, model) is not None,
            order=order,
        )

    # ---- the public entry point ----
    async def place(self, model: str, *, server: Optional[str] = None,
                    exclude: frozenset[str] = frozenset()) -> Decision:
        """Decide where `model` should run. `server` constrains candidates (the
        /api/load path, whose request already names a server kind)."""
        if not model:
            return Decision(outcome="not_found")
        statuses = await self._statuses()
        candidates: list[CandidateFacts] = []
        for order, unit in enumerate(self.orch.units.values()):
            uid = unit.cfg.id
            st = statuses.get(uid)
            if st is None or not unit.cfg.enabled:
                continue
            resolved = await self._resolve(unit, model)
            if resolved is None:
                continue
            srv, load_arg, entry = resolved
            if server is not None and srv != server:
                continue
            candidates.append(self._facts_for(unit, st, srv, load_arg, entry, order, model))
        return rank(model, candidates, self.s, exclude=exclude)
