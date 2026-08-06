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
4. **Fits with displacement** of models that are not shielded:
   - a lane whose occupant is displaceable — victims stay EMPTY, because
     `Lane.load` evicts its occupant itself (that *is* its load semantics);
   - a Spark where stopping the stalest displaceable co-tenants frees enough
     declared budget — single-node victims before groups (ascending node
     count), then fewest victims, then stalest. A `tp > 1` or
     `mem_fraction 0.0` entry claims the whole node: only an empty node, or one
     where EVERY resident is an eligible victim, qualifies.
   Who is displaceable (`_victim_eligible`): a NON-preemptible lease shields
   fully; an UNLEASED model is a victim iff idle (active-and-unleased stays
   protected — there is no holder to notify); a PREEMPTIBLE lease yields once
   its model is idle, and while active it yields to strictly higher-priority
   traffic (`workload_priority`: interactive > neutral > batch — REST paths
   rank at neutral). Evicting a leased victim REVOKES the lease
   (`preempted_by_placement`), so the displaced holder learns via poll/409.
   A multi-node group model is a victim too — ranked LAST via its node span —
   and its eviction is a whole-group teardown carried in
   `Decision.group_victims`, executed by the caller through
   `SparkGroup.placement_evict` BEFORE the target unit's load.
   The idle reaper keeps its stricter convention (ANY lease shields it):
   the reaper is a power optimisation, placement is a competing request.
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


def workload_priority(workload: Optional[Workload], settings: Settings) -> int:
    """Placement priority of an incoming request (PLACEMENT_PRIORITY_*).

    None (REST paths, workload kill switch) is NEUTRAL — this does not
    fabricate a Workload (invariant 15), it maps its absence onto the middle of
    the scale so unclassified traffic neither bullies batch holders nor yields
    to them."""
    if workload is None:
        return settings.placement_priority_neutral
    if workload.cls == "interactive":
        return settings.placement_priority_interactive
    if workload.cls == "batch":
        return settings.placement_priority_batch
    return settings.placement_priority_neutral


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
    # The lease's disposition, for `_victim_eligible`. None = no live lease;
    # False here is the full shield (also used as a sentinel for rows placement
    # must never touch, e.g. a group row whose group unit can't be resolved).
    lease_preemptible: Optional[bool] = None
    lease_priority: int = 0
    # Nodes this ROW spans: 1 for a normal resident, K for a group-claimed row.
    # Victim ordering prefers the smallest span — tearing down a 2-node model
    # to place a 1-node one is a last resort, a 4-node one more so.
    span: int = 1
    group_id: str = ""    # owning SparkGroup unit id for a group row ("" otherwise)


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
    # Nodes this candidate spans: 1 for every lane/Spark, K for a SparkGroup.
    # Sorted ascending in tiers 3/4 — "run each model on the fewest nodes it
    # fits in" (the cluster's operating rule 1): extra nodes buy capacity, not
    # speed, and the fabric pays a cross-node tax per rank. Groups of different
    # sizes are the ONLY candidates that ever differ here (a model resolves
    # either on groups or on single units, never both), so single-unit ranking
    # is untouched.
    member_count: int = 1


@dataclass
class Decision:
    """What placement decided. Exactly one of the shapes below."""

    outcome: str                           # "pin" | "place" | "no_capacity" | "not_found"
    unit_id: str = ""
    server: str = ""
    load_arg: str = ""
    victims: list[str] = field(default_factory=list)   # canonical aliases to evict
    # SparkGroup unit ids whose model must be torn down first. Separate from
    # `victims` because a group teardown is a cross-unit operation the CALLER
    # executes (Placer.evict_group_victims → SparkGroup.placement_evict) before
    # the target unit's load job — the target's own lock can't do it.
    group_victims: list[str] = field(default_factory=list)
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


def _victim_eligible(r: ResidentFact, incoming_priority: int,
                     active_window_s: float, settings: Settings) -> bool:
    """May placement displace this resident? (See the module docstring, tier 4.)

    The asymmetries are deliberate: an UNLEASED active model is never a victim
    (there is no holder to notify), while a PREEMPTIBLE lease buys its holder a
    *notification*, not tenure — idle it yields to anyone, active it yields to
    strictly higher priority. A non-preemptible lease shields fully."""
    if r.lease_preemptible is False:                  # non-preemptible: full shield
        return False
    if r.group_id and not settings.placement_group_eviction_enabled:
        return False
    idle = r.idle_s > active_window_s
    if r.lease_preemptible is None:                   # unleased: today's rule
        return idle
    if idle:                                          # preemptible + idle
        return settings.placement_preempt_leased_idle_enabled
    return (settings.placement_preempt_active_enabled  # preemptible + active
            and r.lease_priority < incoming_priority)


def _victims_for(c: CandidateFacts, headroom: float, active_window_s: float,
                 incoming_priority: int, settings: Settings) -> Optional[list[ResidentFact]]:
    """Cheapest eligible eviction set that makes the model fit, or None."""
    eligible = sorted(
        (r for r in c.residents
         if _victim_eligible(r, incoming_priority, active_window_s, settings)),
        key=lambda r: (r.span, -r.idle_s),  # smallest span, then stalest first
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


def _lane_evictable(c: CandidateFacts, active_window_s: float,
                    incoming_priority: int, settings: Settings) -> bool:
    """A lane candidate whose occupant may be displaced (Lane.load does the evicting).

    The lane flavour of `_victim_eligible`, keyed off `c.usage` rather than the
    occupant's `idle_s` — usage folds in the Monitor's util signal (a client
    hitting Ollama or the relay directly) and a swap in progress, which the
    unit-level idle clock alone can miss."""
    if c.kind == "spark":
        return False
    if c.usage == "free":
        return True
    occ = c.residents[0] if c.residents else None
    if occ is None or occ.lease_preemptible is False:
        return False
    if c.usage == "idle":
        # Idle: unleased yields to anyone (today's rule); a preemptible lease
        # yields unless the rule-1 kill switch restored the old shield.
        return (occ.lease_preemptible is None
                or settings.placement_preempt_leased_idle_enabled)
    # Active: only a preemptible lease held below the incoming priority yields;
    # an unleased active occupant has no holder to notify and stays protected.
    return (occ.lease_preemptible is True
            and settings.placement_preempt_active_enabled
            and occ.lease_priority < incoming_priority)


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
    prio = workload_priority(workload, settings)

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

        # `member_count` leads every branch: "run each model on the fewest nodes
        # it fits in" is a hardware fact (extra ranks buy capacity, not speed,
        # and each one pays the fabric tax), so it outranks even a faster
        # measured load. It is 1 for every lane/Spark, so single-unit ordering is
        # byte-for-byte what it was — only differently-sized GROUPS ever differ.
        if workload is not None and workload.cls == "batch":
            # Capacity tier first even if the lane loads faster — bulk work
            # belongs on the Sparks, and nobody is watching a batch job's TTFB.
            fits.sort(key=lambda c: (c.status.swap_in_progress, c.member_count,
                                     c.kind != "spark",
                                     est_bucket(c), emptiness(c), c.order))
        elif workload is not None:   # interactive
            # Load time dominates (a human is blocked on the cold load); the
            # speed tier breaks est-bucket ties.
            fits.sort(key=lambda c: (c.status.swap_in_progress, c.member_count,
                                     est_bucket(c),
                                     c.kind != "gpu", emptiness(c), c.order))
        else:
            fits.sort(key=lambda c: (c.status.swap_in_progress, c.member_count,
                                     est_bucket(c), emptiness(c), c.order))
        c = fits[0]
        return Decision(outcome="place", unit_id=c.unit_id, server=c.server,
                        load_arg=c.load_arg, tier="fits")

    # Tier 4 — fits by displacing eligible victims (see _victim_eligible).
    displaceable: list[tuple[CandidateFacts, list[ResidentFact]]] = []
    for c in cands:
        if c.lease_refused or not fresh_ok(c):
            continue
        if c.kind == "spark":
            v = _victims_for(c, headroom, window, prio, settings)
            if v is not None:
                displaceable.append((c, v))
        elif _lane_evictable(c, window, prio, settings):
            displaceable.append((c, []))   # Lane.load evicts on its own
    if displaceable:
        # Fewest nodes on the TARGET leads (see CandidateFacts.member_count —
        # only groups differ), then the cheapest VICTIMS: single-node victims
        # before groups, ascending by node span, then fewest, then stalest.
        displaceable.sort(key=lambda cv: (cv[0].member_count,
                                          max((r.span for r in cv[1]), default=1),
                                          len(cv[1]),
                                          -(cv[1][0].idle_s if cv[1] else 1e18),
                                          *order_key(cv[0])))
        c, victims = displaceable[0]
        return Decision(outcome="place", unit_id=c.unit_id, server=c.server,
                        load_arg=c.load_arg,
                        victims=[r.alias for r in victims if not r.group_id],
                        group_victims=list(dict.fromkeys(
                            r.group_id for r in victims if r.group_id)),
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
        elif c.kind == "spark" and any(
                r.group_id and not _victim_eligible(r, prio, window, settings)
                for r in c.residents):
            # Before the unbudgeted-resident reason: a group row is unbudgeted
            # by construction, and "groups are evicted last" is the useful part.
            reasons[c.unit_id] = ("a multi-node group model holds this node and "
                                  "is active, shielded, or held at equal/higher "
                                  "priority — groups are evicted last")
        elif c.kind == "spark" and any(r.budget <= 0.0 for r in c.residents):
            reasons[c.unit_id] = "an unbudgeted resident claims the whole node"
        elif c.kind == "spark" and c.whole_node:
            reasons[c.unit_id] = ("whole-node recipe (tp>1 or unbudgeted) — "
                                  "a resident is active, non-preemptibly leased, "
                                  "or held at equal/higher priority")
        elif c.kind == "spark":
            reasons[c.unit_id] = (f"committed {c.committed:.2f} + {c.want:.2f} "
                                  f"exceeds {headroom:.2f}; no displaceable "
                                  f"co-tenant frees enough")
        else:
            reasons[c.unit_id] = ("occupant is active, non-preemptibly leased, "
                                  "or held at equal/higher priority")
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

        if hasattr(unit, "declared_budgets"):          # SparkUnit or SparkGroup
            # A group's registry is the CLUSTER catalog and a node's is its own —
            # the lists are disjoint, so a model resolves on groups XOR single
            # units and rank() never has to arbitrate between the two. A group
            # additionally must SIZE-match: a 4-node group is no candidate for a
            # recipe capped at 2.
            k = len(getattr(unit.cfg, "member_ids", ())) or 1
            for e in unit.registry.entries():
                if hit(e):
                    if k > 1 and not (e.min_nodes <= k <= e.max_nodes):
                        return None
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
                if m.group:
                    # A multi-node claim: the row belongs to its GROUP, not this
                    # member — clocks and leases live on the group unit, and
                    # eviction is a whole-group teardown (Decision.group_victims
                    # → SparkGroup.placement_evict), never a slot stop here. An
                    # unresolvable group (stale claim, fabric flipped off) stays
                    # fully shielded — never evict what we can't tear down.
                    grp = self.orch.units.get(m.group)
                    if grp is None or not hasattr(grp, "placement_evict"):
                        residents.append(ResidentFact(
                            model=m.model, alias=alias, budget=0.0,
                            idle_s=0.0, leased=True, lease_preemptible=False,
                            span=1, group_id=m.group))
                        continue
                    g_alias = grp.canonical_model(m.model)
                    g_lease = self.leases.active_for(m.group, g_alias)
                    residents.append(ResidentFact(
                        model=m.model,
                        alias=g_alias,
                        budget=0.0,
                        idle_s=grp.idle_for(m.model),
                        leased=True,
                        lease_preemptible=(g_lease.preemptible
                                           if g_lease is not None else None),
                        lease_priority=g_lease.priority if g_lease is not None else 0,
                        span=len(grp.cfg.member_ids),
                        group_id=m.group,
                    ))
                    continue
                lease = self.leases.active_for(uid, alias)
                residents.append(ResidentFact(
                    model=m.model,
                    alias=alias,
                    budget=budgets.get(m.model, 0.0),
                    idle_s=unit.idle_for(m.model),
                    leased=lease is not None,
                    lease_preemptible=lease.preemptible if lease is not None else None,
                    lease_priority=lease.priority if lease is not None else 0,
                ))
        else:
            for m in st.loaded_models:
                # Resolve the catalog alias — vLLM residency reports the SERVED
                # name, and alias=m.model left tier-2 alias matching and the
                # gate's residency-proof broken for GPU lanes (the 2026-07-28
                # alias fix was only half-applied; review 2026-07-29).
                e = unit.registry.find_by_served_name(m.model)
                lease = self.leases.active_for(uid)
                residents.append(ResidentFact(
                    model=m.model, alias=(e.alias if e else m.model), budget=0.0,
                    idle_s=st.idle_s or 0.0,
                    leased=lease is not None,
                    lease_preemptible=lease.preemptible if lease is not None else None,
                    lease_priority=lease.priority if lease is not None else 0,
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

    def _group_facts(self, group: "Unit", st: LaneStatus, server: str, load_arg: str,
                     entry, order: int, model: str,
                     statuses: dict[str, LaneStatus]) -> Optional[CandidateFacts]:
        """CandidateFacts for a SparkGroup: the residents are the UNION of the
        members' residents — a tp launch claims every member whole, so tier 3
        needs all of them empty and tier 4 needs every one of them evictable
        (exactly the whole_node semantics rank() already enforces). Victim
        eligibility (idle/leased) is judged against the OWNING member, which is
        where the clocks and leases live and where `_evict_victim` will
        re-validate under the lock. None ⇒ a member's status is missing this
        sweep, and a group that cannot see all its members is not a candidate.
        """
        residents: list[ResidentFact] = []
        for m_unit in group.members:
            mid = m_unit.cfg.id
            m_st = statuses.get(mid)
            if m_st is None or not m_unit.cfg.enabled:
                return None
            budgets = m_unit.declared_budgets([lm.model for lm in m_st.loaded_models])
            for lm in m_st.loaded_models:
                alias = m_unit.canonical_model(lm.model)
                if lm.group:
                    # ANY group's claim on a member is untouchable from a group
                    # candidate — group-evicts-group is out of scope (the load
                    # path fast-fails such overlaps with placement_conflict),
                    # so the row keeps the full shield here.
                    residents.append(ResidentFact(
                        model=lm.model, alias=alias, budget=0.0,
                        idle_s=m_unit.idle_for(lm.model),
                        leased=True, lease_preemptible=False,
                        group_id=lm.group))
                    continue
                lease = self.leases.active_for(mid, alias)
                residents.append(ResidentFact(
                    model=lm.model,
                    alias=alias,
                    budget=budgets.get(lm.model, 0.0),
                    idle_s=m_unit.idle_for(lm.model),
                    leased=lease is not None,
                    lease_preemptible=lease.preemptible if lease is not None else None,
                    lease_priority=lease.priority if lease is not None else 0,
                ))
        # The group's own residency (its model on the head) rides on top so
        # tier 2 can route a request for the RESIDENT group model here.
        k = len(group.cfg.member_ids)
        for lm in st.loaded_models:
            g_lease = self.leases.active_for(group.cfg.id)
            residents.append(ResidentFact(
                model=lm.model,
                alias=group.canonical_model(lm.model),
                budget=0.0,
                idle_s=group.idle_for(lm.model),
                leased=g_lease is not None,
                lease_preemptible=g_lease.preemptible if g_lease is not None else None,
                lease_priority=g_lease.priority if g_lease is not None else 0,
                span=k,
                group_id=group.cfg.id,
            ))
        usage = classify_usage(st, self.monitor.util_for(group.cfg.gpu_uuid), self.s)
        # Gate facts, group flavour. `proven` cannot come from residency here (a
        # multi-node cold start is the expensive case the gate exists for), so the
        # proof is the PERSISTED RECORD: this alias has launched on THIS node set
        # before. That is exactly what makes a recorded placement an
        # auto-placement candidate and an unrecorded node set a manual, deliberate
        # first launch from the Cluster tab. Estimates and failure counters are
        # keyed per node count / per group id — a 2-node launch is not a 4-node
        # one, and a group's failures are its own.
        proven, fail_blocked, fail_count, est_s = True, False, 0, None
        if self.load_times is not None:
            from .load_times import fail_key, spark_key
            est_s = self.load_times.estimate(f"{spark_key(load_arg)}:x{k}")
            recorded = tuple(group.cfg.member_ids) in group.placements.sets_for(load_arg)
            proven = est_s is not None or recorded
            fk = fail_key(group.cfg.id, "spark", load_arg)
            fail_count, _ts = self.load_times.failures_for(fk)
            threshold = self.s.placement_fail_block_after
            fail_blocked = threshold > 0 and self.load_times.blocked(
                fk, threshold=threshold,
                cooldown_s=self.s.placement_fail_block_cooldown_s)
        return CandidateFacts(
            unit_id=group.cfg.id, kind="spark", status=st, usage=usage,
            server=server, load_arg=load_arg, residents=residents,
            committed=sum(r.budget for r in residents),
            want=0.0, whole_node=True,
            free_slot=not st.loaded_models,
            lease_refused=self.leases.blocks_unleased(group.cfg.id, model) is not None,
            order=order,
            proven=proven, fail_blocked=fail_blocked, fail_count=fail_count,
            est_s=est_s,
            member_count=k,
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
            if getattr(unit, "kind", "") == "spark_group":
                facts = self._group_facts(unit, st, srv, load_arg, entry, order,
                                          model, statuses)
                if facts is not None:
                    candidates.append(facts)
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
                  and not decision.victims and not decision.group_victims
                  and not exclude)
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
            "priority": workload_priority(workload, self.s),
            "victims": list(decision.victims),
            "group_victims": list(decision.group_victims),
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

    # ---- group-victim execution ----
    async def evict_group_victims(self, group_victims: list[str], *,
                                  priority: Optional[int],
                                  requested_by: str) -> None:
        """Execute a Decision's `group_victims` — tear each group's model down
        via `SparkGroup.placement_evict` BEFORE the target unit's load job runs.

        The single funnel for every call site (gateway chat/pooling, the one
        re-place, REST /api/load auto): the teardown holds the group's and every
        member's lock, so it can never run inside the target member's own load
        (deadlock) — it happens here, first. Raises
        `RuntimeError("placement_conflict: …")` when re-validation under those
        locks refuses (the world moved); callers route that into the existing
        single-re-place machinery. No-op for an empty list."""
        if not group_victims:
            return
        for gid in group_victims:
            unit = self.orch.units.get(gid)
            if unit is None or not hasattr(unit, "placement_evict"):
                raise RuntimeError(
                    f"placement_conflict: group victim '{gid}' no longer exists")
            await unit.placement_evict(priority=priority, requested_by=requested_by)
        self.invalidate()
