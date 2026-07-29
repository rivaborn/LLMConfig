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
   (and has a free slot, and no unbudgeted resident), or a free lane. Fastest
   measured load first (bucketed to the minute, from `LoadTimes`), then
   emptiest — a fresh load should go where it comes up quickest, but minute-
   level ties fall back to balance.
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

## The proven-load gate (tiers 3 and 4 only)

A fresh load is only *placed* on a unit where this model has **launched
successfully before** (`PLACEMENT_REQUIRE_PROVEN`, default on). Proof is a
`LoadTimes` sample — recorded iff a real launch succeeded — or current
residency: `spark:{alias}` samples are shared across the four identical GB10s
(loaded on one Spark ⇒ provable on any), lanes prove per-unit. The tiers that
don't CHOOSE are exempt on purpose: the sole-candidate pin (tier 1) behaves as
an explicit header, which is how a first-ever load gets seeded at all, and a
resident model (tier 2) is its own proof. Note the limit: proof means the
LAUNCH succeeds, not that inference is stable.

Alongside it, a **consecutive-failure blocklist**: N failed launches in a row
(`PLACEMENT_FAIL_BLOCK_AFTER`) blocks that unit for
`PLACEMENT_FAIL_BLOCK_COOLDOWN_S`, after which ONE probe attempt is allowed.
Failure counters are per-unit even for Sparks — launch failures are usually
node-state-dependent (a co-resident's quantize transient), so spark1 failing
must not block spark2.

## Workload-aware tiering (2026-07-29)

The fleet has two performance tiers with opposite strengths: the RTX 3090 is
the SPEED tier (highest single-stream decode, ~24 GB — best time-to-answer for
one caller) and the Spark cluster is the CAPACITY/CONCURRENCY tier (121 GB
unified pools, continuous batching that amortizes decode across requests —
measured 43:1 prefill:decode on batch work without breaking a sweat). When a
model resolves on BOTH tiers, the request's shape decides:

* **interactive** — small prompt, bounded generation, a human waiting. Prefer
  the GPU lane: tier 2 keeps idle-first (never queue a latency request behind
  an active model) and breaks ties toward the 3090; tier 3 keeps
  fastest-load-first and breaks est-bucket ties toward the 3090.
* **batch** — long prompt, bulk generation, or a pooling body. Prefer a Spark
  BEFORE idleness: an active Spark absorbs one more request into its batch by
  design, and it keeps the 3090 free for interactive traffic.

Classification (`classify_workload`) is heuristic-from-the-body — prompt chars
vs `PLACEMENT_INTERACTIVE_MAX_PROMPT_CHARS`, `max_tokens` vs
`PLACEMENT_INTERACTIVE_MAX_NEW_TOKENS`, pooling bodies always batch — with an
`X-LLM-Workload: interactive|batch` header override. No workload (REST
`/api/load`, `PLACEMENT_WORKLOAD_ENABLED=false`) means the neutral ordering,
byte-for-byte as before. Like everything here it is a PREFERENCE, not a gate:
it reorders candidates and never disqualifies one — capacity, leases, the
proven gate and admission still govern.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .config import Settings
from .idle import classify_usage
from .schemas import LaneStatus

if TYPE_CHECKING:
    from .leases import LeaseManager
    from .load_times import LoadTimes
    from .monitor import Monitor
    from .orchestrator import Orchestrator, Unit

AUTO = "auto"
DECISION_LOG_SIZE = 50


def wants_auto(lane_header: Optional[str]) -> bool:
    """Absent, empty, or the literal 'auto' (any case) → auto-place."""
    return not lane_header or lane_header.strip().lower() == AUTO


# --------------------------------------------------------------------------- #
# Workload classification — interactive (speed tier) vs batch (capacity tier)
# --------------------------------------------------------------------------- #
@dataclass
class Workload:
    """What kind of request this is, for tier preference in rank()."""

    cls: str                        # "interactive" | "batch"
    prompt_chars: int = 0
    max_tokens: Optional[int] = None
    source: str = "heuristic"       # "header" | "heuristic" | "pooling"

    @property
    def preferred_kind(self) -> str:
        return "gpu" if self.cls == "interactive" else "spark"


_WORKLOAD_HEADER_VALUES = {
    "interactive": "interactive", "latency": "interactive",
    "batch": "batch", "throughput": "batch", "bulk": "batch",
}


def _prompt_chars(body: dict) -> int:
    """Rough request size without tokenizing (~4 chars/token). Good enough to
    split 'a question' from 'a document' — the only distinction we need."""
    msgs = body.get("messages")
    if isinstance(msgs, list):
        total = 0
        for m in msgs:
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):        # multi-part content blocks
                total += sum(len(txt) for p in content if isinstance(p, dict)
                             and isinstance(txt := p.get("text"), str))
        return total
    prompt = body.get("prompt")
    if isinstance(prompt, str):
        return len(prompt)
    if isinstance(prompt, list):
        return sum(len(p) for p in prompt if isinstance(p, str))
    return 0


def classify_workload(body: dict, header: Optional[str],
                      settings: Settings) -> Optional[Workload]:
    """Classify one /v1 request, or None (= neutral ranking, the pre-workload
    ordering). An explicit `X-LLM-Workload` header always wins; otherwise a
    heuristic on the body: pooling bodies (no messages/prompt) are batch — bulk
    data paths by construction — and a chat/completion is interactive iff the
    prompt is small AND the requested generation is bounded small."""
    if not settings.placement_workload_enabled:
        return None
    hdr = _WORKLOAD_HEADER_VALUES.get((header or "").strip().lower())
    chars = _prompt_chars(body)
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
    max_tokens = int(max_tokens) if isinstance(max_tokens, (int, float)) else None
    if hdr:
        return Workload(cls=hdr, prompt_chars=chars, max_tokens=max_tokens,
                        source="header")
    if "messages" not in body and "prompt" not in body:
        return Workload(cls="batch", prompt_chars=chars, source="pooling")
    interactive = (chars <= settings.placement_interactive_max_prompt_chars
                   and (max_tokens is None
                        or max_tokens <= settings.placement_interactive_max_new_tokens))
    return Workload(cls="interactive" if interactive else "batch",
                    prompt_chars=chars, max_tokens=max_tokens)


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
    proven: bool = True                    # launched successfully here before (see gate)
    fail_blocked: bool = False             # consecutive-failure blocklist tripped
    fail_count: int = 0                    # for the tier-5 reason text
    est_s: Optional[float] = None          # measured load estimate (tier-3 tie-break)


@dataclass
class Decision:
    """What placement decided. Exactly one of the shapes below."""

    outcome: str                           # "pin" | "place" | "no_capacity" | "not_found"
    unit_id: str = ""
    server: str = ""
    load_arg: str = ""
    victims: list[str] = field(default_factory=list)   # canonical aliases to evict
    reasons: dict[str, str] = field(default_factory=dict)  # unit -> why not (no_capacity)
    tier: str = ""                         # which tier fired: pin|resident|fits|displace
                                           # ("" on refusals) — for the decision log


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

    if not fits():
        # Prefer a SINGLE sufficient victim (stalest among those that suffice)
        # over greedy stalest-first accumulation — greedy could evict two
        # models where the second alone was enough (review 2026-07-29).
        lone = next((r for r in eligible if r not in victims
                     and committed - r.budget + c.want <= headroom + 1e-9
                     and c.free_slot_after(len(victims) + 1)), None)
        if lone is not None:
            victims.append(lone)
            committed -= lone.budget
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
         *, exclude: frozenset[str] = frozenset(),
         workload: Optional[Workload] = None) -> Decision:
    """Pure ranking — see the module docstring for the tiers. `workload` biases
    tier-2/tier-3 ordering toward the speed tier (gpu) or the capacity tier
    (spark); None keeps the neutral ordering byte-for-byte."""
    cands = [c for c in candidates if c.unit_id not in exclude]
    if not cands:
        if candidates:
            # The model DOES resolve somewhere — every unit was excluded after a
            # conflict. A 404 here would be a lie; report busy, not missing.
            return Decision(outcome="no_capacity", reasons={
                c.unit_id: "excluded after a load conflict" for c in candidates})
        return Decision(outcome="not_found")

    # Tier 1 — sole candidate pins, bypassing every predicate below.
    if len(cands) == 1:
        c = cands[0]
        return Decision(outcome="pin", unit_id=c.unit_id, server=c.server,
                        load_arg=c.load_arg, tier="pin")

    def order_key(c: CandidateFacts):
        return (c.status.swap_in_progress, c.order)   # mid-swap deprioritized, never out

    # Tier 2 — resident, not lease-refused; idle beats active; stable tie-break.
    # Matched by served name OR canonical alias: /api/load passes aliases, and an
    # alias request missing its warm resident would place a second copy elsewhere.
    resident = [c for c in cands
                if not c.lease_refused
                and any(r.model == model or r.alias == model for r in c.residents)]
    if resident:
        if workload is not None and workload.cls == "batch":
            # Capacity tier BEFORE idleness: an active Spark absorbs one more
            # request into its continuous batch by design, and bulk work on the
            # 3090 would evict the speed tier's availability.
            resident.sort(key=lambda c: (c.kind != "spark", c.usage != "idle",
                                         *order_key(c)))
        elif workload is not None:   # interactive
            # Idle-first still (never queue a latency request behind an active
            # model), speed tier breaks the tie.
            resident.sort(key=lambda c: (c.usage != "idle", c.kind != "gpu",
                                         *order_key(c)))
        else:
            resident.sort(key=lambda c: (c.usage != "idle", *order_key(c)))
        c = resident[0]
        return Decision(outcome="place", unit_id=c.unit_id, server=c.server,
                        load_arg=c.load_arg, tier="resident")

    headroom = settings.spark_mem_headroom
    window = settings.usage_active_window_s

    def fresh_ok(c: CandidateFacts) -> bool:
        """May this unit take a FRESH load? (The gate — tiers 3/4 only.)"""
        if c.fail_blocked:
            return False
        return c.proven or not settings.placement_require_proven

    # Tier 3 — fits without displacing anything; fastest measured load first
    # (bucketed to the minute — sub-minute differences are noise), then emptiest.
    fits = [c for c in cands if not c.lease_refused and fresh_ok(c)
            and _fits_without_displacement(c, headroom)]
    if fits:
        def est_bucket(c: CandidateFacts) -> float:
            # No estimate (proven by residency alone) sorts last among proven —
            # a measured launch beats an unmeasured one when both fit.
            return round(c.est_s / 60.0) if c.est_s is not None else float("inf")
        def emptiness(c: CandidateFacts) -> float:
            return (c.committed if c.kind == "spark"
                    else (c.status.gpu.vram_pct if c.status.gpu else 0.0))

        if workload is not None and workload.cls == "batch":
            # Capacity tier first even if the lane loads faster — bulk work
            # belongs on the Sparks, and nobody is watching a batch job's TTFB.
            fits.sort(key=lambda c: (c.status.swap_in_progress, c.kind != "spark",
                                     est_bucket(c), emptiness(c), c.order))
        elif workload is not None:   # interactive
            # Load time dominates (a human is blocked on the cold load); the
            # speed tier breaks est-bucket ties.
            fits.sort(key=lambda c: (c.status.swap_in_progress, est_bucket(c),
                                     c.kind != "gpu", emptiness(c), c.order))
        else:
            fits.sort(key=lambda c: (c.status.swap_in_progress, est_bucket(c),
                                     emptiness(c), c.order))
        c = fits[0]
        return Decision(outcome="place", unit_id=c.unit_id, server=c.server,
                        load_arg=c.load_arg, tier="fits")

    # Tier 4 — fits by displacing idle + unleased models only.
    displaceable: list[tuple[CandidateFacts, list[ResidentFact]]] = []
    for c in cands:
        if c.lease_refused or not fresh_ok(c):
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
                        load_arg=c.load_arg, victims=[r.alias for r in victims],
                        tier="displace")

    # Tier 5 — nothing admissible: say why, per unit.
    reasons = {}
    for c in cands:
        if c.lease_refused:
            reasons[c.unit_id] = "held by a non-preemptible lease"
        elif c.fail_blocked:
            reasons[c.unit_id] = (f"{c.fail_count} consecutive launch failures — "
                                  f"blocked until the cooldown lapses "
                                  f"(or load it explicitly)")
        elif not c.proven and settings.placement_require_proven:
            reasons[c.unit_id] = ("never loaded successfully here — load it once "
                                  "explicitly to prove it")
        elif c.kind == "spark" and not c.free_slot:
            reasons[c.unit_id] = "all slots in use"
        elif c.kind == "spark" and any(r.budget <= 0.0 for r in c.residents):
            reasons[c.unit_id] = "an unbudgeted resident claims the whole node"
        elif c.kind == "spark" and c.whole_node:
            reasons[c.unit_id] = ("whole-node recipe (tp>1 or unbudgeted) — "
                                  "residents are active or leased")
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
                 leases: "LeaseManager", monitor: "Monitor",
                 load_times: "Optional[LoadTimes]" = None):
        self.s = settings
        self.orch = orch
        self.leases = leases
        self.monitor = monitor
        self.load_times = load_times   # proven-load gate + blocklist + estimates
        # Single-flight status sweep: one in-flight task shared by every awaiter,
        # then cached for placement_cache_ttl_s. Staleness only mis-ranks (see
        # module docstring); freshness costs a full-fleet probe per request.
        self._sweep_task: Optional[asyncio.Task] = None
        self._sweep_ts: float = 0.0
        self._sweep_result: dict[str, LaneStatus] = {}
        self._sweep_gen: int = 0     # bumped by invalidate(); stale sweeps don't publish
        # Ollama tag lists, per lane, with their own TTL. The registry halves of
        # _resolve are in-memory; this is its ONLY real I/O, and it sat on the
        # hot path serialized per lane — a wedged Ollama stacked its client
        # timeout onto every auto request for a tag. Failures cache too
        # (negative cache): a down Ollama answers "no tags" instantly for the
        # TTL instead of being re-probed per request.
        self._tags_cache: dict[str, tuple[float, frozenset[str]]] = {}
        # The decision log: last N placements, newest last. Consecutive
        # identical boring decisions (same model→unit→tier, nothing displaced
        # or excluded) collapse into one entry with a count, so a chatty client
        # can't wash the interesting entries out of the window.
        self._decisions: deque[dict] = deque(maxlen=DECISION_LOG_SIZE)

    def invalidate(self) -> None:
        """Drop the cached sweep. Called when a load is COMMITTED off a placement
        decision: within the TTL a burst's second request would otherwise rank on
        a snapshot predating the first request's load and double-place onto the
        same unit — admission still refuses (the gates hold), but it burns the
        gateway's single re-place. A forced re-sweep sees the in-flight job.

        The generation bump + task drop matter: a sweep that STARTED before the
        load would otherwise complete after this call and re-publish the
        pre-load snapshot with a fresh timestamp (review 2026-07-29)."""
        self._sweep_gen += 1
        self._sweep_task = None
        self._sweep_result, self._sweep_ts = {}, 0.0

    # ---- status sweep ----
    async def _statuses(self) -> dict[str, LaneStatus]:
        now = time.monotonic()
        if self._sweep_result and now - self._sweep_ts < self.s.placement_cache_ttl_s:
            return self._sweep_result
        if self._sweep_task is None or self._sweep_task.done():
            self._sweep_task = asyncio.create_task(self._sweep())
        return await asyncio.shield(self._sweep_task)

    async def _sweep(self) -> dict[str, LaneStatus]:
        gen = self._sweep_gen
        units = [u for u in self.orch.units.values() if u.cfg.enabled]
        results = await asyncio.gather(*(u.status() for u in units),
                                       return_exceptions=True)
        out: dict[str, LaneStatus] = {}
        for u, res in zip(units, results):
            if isinstance(res, LaneStatus):
                out[u.cfg.id] = res
            # a raising unit simply isn't a candidate this round
        if gen == self._sweep_gen:   # stale sweeps (invalidated mid-flight) don't publish
            self._sweep_result, self._sweep_ts = out, time.monotonic()
        return out

    # ---- resolution (mirrors gateway.resolve, minus the URL) ----
    async def _resolve(self, unit: "Unit", model: str) -> Optional[tuple[str, str, object]]:
        """(server, load_arg, spark_entry|None) if `model` can run on `unit`.

        Matches the served name OR the catalog alias — `/api/load {lane: auto}`
        passes aliases, and served-name-only matching 404'd every alias whose
        served name differs (found in review, 2026-07-28)."""
        def hit(e) -> bool:
            return (e.alias == model or (e.served_name or e.alias) == model) \
                and e.status != "blocked"

        if hasattr(unit, "declared_budgets"):          # SparkUnit
            for e in unit.registry.entries():
                if hit(e):
                    return ("spark", e.alias, e)
            return None
        # Ollama-only lane: its vLLM catalog is not a candidate source at all.
        for e in (unit.registry.entries() if unit.cfg.vllm_enabled else []):
            if hit(e):
                return ("vllm", e.alias, None)
        if ":" in model:
            if model in await self._ollama_tags(unit):
                return ("ollama", model, None)
        return None

    async def _ollama_tags(self, unit: "Unit") -> frozenset[str]:
        """This lane's Ollama tag set, cached for placement_tags_ttl_s.

        Tags change on the timescale of pulls, not requests, and staleness only
        mis-ranks (placement is advisory) — so even a generous TTL is safe.
        Failures cache as an empty set on purpose: without that, a lane whose
        Ollama is down-but-not-refusing re-pays its client timeout on EVERY auto
        request for a tag, serialized per lane."""
        uid = unit.cfg.id
        cached = self._tags_cache.get(uid)
        if cached and time.monotonic() - cached[0] < self.s.placement_tags_ttl_s:
            return cached[1]
        try:
            names = frozenset(m.name for m in await unit.ollama.list_models())
        except Exception:  # noqa: BLE001 — a down Ollama just isn't a candidate
            names = frozenset()
        self._tags_cache[uid] = (time.monotonic(), names)
        return names

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

    def _gate_facts(self, unit: "Unit", server: str, alias: str, is_spark: bool,
                    spark_resident: frozenset[str],
                    residents: list[ResidentFact]) -> tuple[bool, bool, int, Optional[float]]:
        """(proven, fail_blocked, fail_count, est_s) for the proven-load gate.

        Proof = a LoadTimes sample (recorded iff a real launch succeeded), or
        current residency — a running model is live proof even when it predates
        the recorder. Spark proof is fleet-wide (identical GB10s: the success key
        and residency on ANY spark both count); lanes prove per-unit. Failure
        counters are per-unit for everyone (see load_times.fail_key).
        """
        if self.load_times is None:
            return True, False, 0, None
        from .load_times import fail_key, lane_key, spark_key
        key = spark_key(alias) if is_spark else lane_key(unit.cfg.id, server, alias)
        est = self.load_times.estimate(key)
        proven = est is not None or (
            alias in spark_resident if is_spark
            else any(r.alias == alias or r.model == alias for r in residents))
        fk = fail_key(unit.cfg.id, "spark" if is_spark else server, alias)
        count, _ts = self.load_times.failures_for(fk)
        threshold = self.s.placement_fail_block_after
        blocked = threshold > 0 and self.load_times.blocked(
            fk, threshold=threshold, cooldown_s=self.s.placement_fail_block_cooldown_s)
        return proven, blocked, count, est

    def _facts_for(self, unit: "Unit", st: LaneStatus, server: str, load_arg: str,
                   entry, order: int, model: str,
                   spark_resident: frozenset[str] = frozenset()) -> CandidateFacts:
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
                # Resolve the catalog alias — vLLM residency reports the SERVED
                # name, and alias=m.model left tier-2 alias matching and the
                # gate's residency-proof broken for GPU lanes (the 2026-07-28
                # alias fix was only half-applied; review 2026-07-29).
                e = unit.registry.find_by_served_name(m.model)
                residents.append(ResidentFact(
                    model=m.model, alias=(e.alias if e else m.model), budget=0.0,
                    idle_s=st.idle_s or 0.0,
                    leased=self.leases.active_for(uid) is not None,
                ))
        committed = sum(r.budget for r in residents) + self._inflight_budget(unit, st)
        want = entry.mem_fraction if entry is not None else 0.0
        # `needs_empty_node` gets the same PLACEMENT semantics as a whole-node
        # claim — only an empty node (or one whose residents are all evictable)
        # qualifies. Without this, placement would keep choosing a populated node
        # that `_load` is now guaranteed to refuse, burning the single re-place.
        whole_node = bool(entry is not None
                          and (entry.tp > 1 or want <= 0.0
                               or getattr(entry, "needs_empty_node", False))) if is_spark else False
        free_slot = (len(st.loaded_models) < getattr(unit.cfg, "max_models", 1)) if is_spark else True
        proven, fail_blocked, fail_count, est_s = self._gate_facts(
            unit, server, load_arg, is_spark, spark_resident, residents)
        return CandidateFacts(
            unit_id=uid, kind="spark" if is_spark else "gpu", status=st, usage=usage,
            server=server, load_arg=load_arg, residents=residents,
            committed=committed, want=want, whole_node=whole_node,
            free_slot=free_slot,
            lease_refused=self.leases.blocks_unleased(uid, model) is not None,
            order=order,
            proven=proven, fail_blocked=fail_blocked, fail_count=fail_count,
            est_s=est_s,
        )

    # ---- the public entry point ----
    async def place(self, model: str, *, server: Optional[str] = None,
                    exclude: frozenset[str] = frozenset(),
                    workload: Optional[Workload] = None) -> Decision:
        """Decide where `model` should run. `server` constrains candidates (the
        /api/load path, whose request already names a server kind); `workload`
        biases the ranking toward the speed or capacity tier (None = neutral)."""
        if not model:
            return Decision(outcome="not_found")
        statuses = await self._statuses()
        # Aliases resident on ANY spark right now — live proof for the gate
        # (identical hardware, same rule as the shared spark:{alias} sample key).
        spark_resident = frozenset(
            u.canonical_model(m.model)
            for u in self.orch.units.values()
            if hasattr(u, "declared_budgets") and u.cfg.id in statuses
            for m in statuses[u.cfg.id].loaded_models
        )
        eligible = [(order, unit, statuses[unit.cfg.id])
                    for order, unit in enumerate(self.orch.units.values())
                    if unit.cfg.enabled and unit.cfg.id in statuses]
        # Resolve concurrently: registry halves are in-memory, but a cold Ollama
        # tag fetch is real I/O and was paid back-to-back per lane.
        resolutions = await asyncio.gather(
            *(self._resolve(unit, model) for _, unit, _ in eligible))
        candidates: list[CandidateFacts] = []
        for (order, unit, st), resolved in zip(eligible, resolutions):
            if resolved is None:
                continue
            srv, load_arg, entry = resolved
            if server is not None and srv != server:
                continue
            candidates.append(self._facts_for(unit, st, srv, load_arg, entry, order,
                                              model, spark_resident))
        decision = rank(model, candidates, self.s, exclude=exclude,
                        workload=workload)
        self._record(model, server, exclude, candidates, decision, workload)
        return decision

    # ---- the decision log ----
    def _record(self, model: str, server: Optional[str],
                exclude: frozenset[str], candidates: list[CandidateFacts],
                decision: Decision, workload: Optional[Workload] = None) -> None:
        """Append to the ring buffer (see __init__ for the dedupe rationale)."""
        boring = (decision.outcome in ("place", "pin")
                  and not decision.victims and not exclude)
        if boring and self._decisions:
            last = self._decisions[-1]
            if (last.get("_boring")
                    and last["model"] == model
                    and last["unit"] == decision.unit_id
                    and last["tier"] == decision.tier
                    and last["server_constraint"] == (server or "")
                    and last["workload"] == (workload.cls if workload else "")):
                last["count"] += 1
                last["ts"] = round(time.time(), 1)
                return
        self._decisions.append({
            "ts": round(time.time(), 1),
            "count": 1,
            "model": model,
            "outcome": decision.outcome,
            "unit": decision.unit_id,
            "tier": decision.tier,
            "workload": workload.cls if workload else "",
            "victims": list(decision.victims),
            "server_constraint": server or "",
            "exclude": sorted(exclude),
            "reasons": dict(decision.reasons),
            "sweep_age_s": round(max(time.monotonic() - self._sweep_ts, 0.0), 2)
                           if self._sweep_ts else None,
            "candidates": [{
                "unit": c.unit_id,
                "kind": c.kind,
                "usage": c.usage,
                "resident": any(r.model == model or r.alias == model
                                for r in c.residents),
                "proven": c.proven,
                "fail_blocked": c.fail_blocked,
                "lease_refused": c.lease_refused,
                "swap": c.status.swap_in_progress,
                "committed": round(c.committed, 2),
                "want": round(c.want, 2),
                "est_s": c.est_s,
            } for c in candidates],
            "_boring": boring,
        })

    def decisions(self) -> list[dict]:
        """Newest-first copy of the log, without the internal dedupe marker."""
        return [{k: v for k, v in d.items() if not k.startswith("_")}
                for d in reversed(self._decisions)]
