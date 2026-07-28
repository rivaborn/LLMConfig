"""Auto-placement (`llmconfig/placement.py`) — the pure ranking and the Placer's
fact sweep. No HTTP servers; units and leases are minimal fakes."""
import asyncio
import time
from types import SimpleNamespace

from llmconfig.config import Settings
from llmconfig.placement import (
    CandidateFacts,
    Placer,
    ResidentFact,
    rank,
    wants_auto,
)
from llmconfig.schemas import GpuOut, LaneStatus, LoadedModel

S = Settings(_env_file=None)
WINDOW = S.usage_active_window_s          # 60s — the idle/active boundary
IDLE = WINDOW + 600


def st(unit="u1", *, models=(), swap=False, vram=0.0, owner=None):
    lm = [LoadedModel(server="spark", model=m) for m in models]
    return LaneStatus(
        id=unit, name=unit, owner=owner or ("spark" if lm else "free"),
        ollama_up=False, vllm_up=False,
        loaded=lm[0] if lm else None, loaded_models=lm,
        swap_in_progress=swap,
        gpu=GpuOut(found=True, vram_pct=vram),
    )


def spark(unit, *, residents=(), committed=None, want=0.3, whole=False,
          free_slot=True, refused=False, order=0, swap=False, usage="idle"):
    res = list(residents)
    return CandidateFacts(
        unit_id=unit, kind="spark",
        status=st(unit, models=[r.model for r in res], swap=swap),
        usage=usage, server="spark", load_arg="target",
        residents=res,
        committed=committed if committed is not None else sum(r.budget for r in res),
        want=want, whole_node=whole, free_slot=free_slot,
        lease_refused=refused, order=order,
    )


def gpu_lane(unit, *, usage, occupant=None, leased=False, refused=False,
             order=0, vram=0.0):
    res = ([ResidentFact(model=occupant, alias=occupant, budget=0.0,
                         idle_s=IDLE if usage == "idle" else 1.0, leased=leased)]
           if occupant else [])
    return CandidateFacts(
        unit_id=unit, kind="gpu",
        status=st(unit, models=[r.model for r in res], vram=vram,
                  owner="vllm" if res else "free"),
        usage=usage, server="vllm", load_arg="target",
        residents=res, lease_refused=refused, order=order,
    )


def R(model, *, budget=0.3, idle=IDLE, leased=False, alias=None):
    return ResidentFact(model=model, alias=alias or model, budget=budget,
                        idle_s=idle, leased=leased)


# --------------------------------------------------------------------------- #
# wants_auto
# --------------------------------------------------------------------------- #
def test_wants_auto_on_absent_empty_or_sentinel():
    assert wants_auto(None) and wants_auto("") and wants_auto("auto") and wants_auto(" AUTO ")
    assert not wants_auto("primary") and not wants_auto("spark1")


# --------------------------------------------------------------------------- #
# Tier 1 — sole candidate pins
# --------------------------------------------------------------------------- #
def test_sole_candidate_pins_even_when_active_and_leased():
    """A single-unit deployment must behave exactly as an explicit header would:
    the unit's own load semantics apply, protections govern only CHOICES."""
    c = gpu_lane("primary", usage="active", occupant="other", leased=True, refused=False)
    d = rank("target", [c], S)
    assert d.outcome == "pin" and d.unit_id == "primary"
    assert d.server == "vllm" and d.load_arg == "target"


def test_no_candidates_is_not_found_never_503():
    assert rank("target", [], S).outcome == "not_found"


def test_exclude_exhausted_is_no_capacity_not_a_404():
    # The model DOES resolve on s1 — it was excluded after a load conflict. A
    # 404 ("model not found") there is a lie; the truthful answer is busy.
    c = spark("s1")
    d = rank("target", [c], S, exclude=frozenset({"s1"}))
    assert d.outcome == "no_capacity"
    assert "excluded" in d.reasons["s1"]


# --------------------------------------------------------------------------- #
# Tier 2 — resident first
# --------------------------------------------------------------------------- #
def test_resident_beats_every_empty_unit():
    resident = spark("s1", residents=[R("target-served")], order=3, usage="active")
    resident.residents[0].model = "target"          # residency matches the request
    empty = spark("s2", order=0)
    d = rank("target", [resident, empty], S)
    assert d.outcome == "place" and d.unit_id == "s1" and not d.victims


def test_idle_resident_beats_active_resident_then_stable_order():
    a = spark("s1", residents=[R("target")], usage="active", order=0)
    b = spark("s2", residents=[R("target")], usage="idle", order=1)
    d = rank("target", [a, b], S)
    assert d.unit_id == "s2", "idle beats active regardless of unit order"

    c1 = spark("s1", residents=[R("target")], usage="idle", order=0)
    c2 = spark("s2", residents=[R("target")], usage="idle", order=1)
    assert rank("target", [c1, c2], S).unit_id == "s1", "stable tie-break = affinity"
    assert rank("target", [c2, c1], S).unit_id == "s1", "order field, not list position"


def test_lease_refused_resident_is_skipped():
    held = spark("s1", residents=[R("target")], refused=True, order=0)
    other = spark("s2", order=1)
    d = rank("target", [held, other], S)
    assert d.unit_id == "s2"


# --------------------------------------------------------------------------- #
# Tier 3 — fits without displacement
# --------------------------------------------------------------------------- #
def test_spark_fit_by_declared_budget_and_emptiest_wins():
    fuller = spark("s1", residents=[R("a", budget=0.5)], want=0.3, order=0)
    emptier = spark("s2", residents=[R("b", budget=0.2)], want=0.3, order=1)
    d = rank("target", [fuller, emptier], S)
    assert d.unit_id == "s2" and not d.victims


def test_inflight_budget_counts_and_unbudgeted_resident_blocks():
    # committed already includes the in-flight load (Placer adds it); over headroom → no fit
    over = spark("s1", residents=[R("a", budget=0.5)], committed=0.9, want=0.3)
    ok = spark("s2", residents=[R("b", budget=0.2)], want=0.3, order=1)
    assert rank("target", [over, ok], S).unit_id == "s2"

    poisoned = spark("s1", residents=[R("a", budget=0.0, idle=1.0)], want=0.3)
    ok2 = spark("s2", order=1)
    assert rank("target", [poisoned, ok2], S).unit_id == "s2"


def test_free_lane_fits_and_mid_swap_deprioritized():
    free = gpu_lane("companion", usage="free", order=1)
    swapping = spark("s1", order=0, swap=True)
    d = rank("target", [swapping, free], S)
    assert d.unit_id == "companion", "a mid-swap unit is deprioritized, not banned"


# --------------------------------------------------------------------------- #
# Tier 4 — displacement
# --------------------------------------------------------------------------- #
def test_spark_eviction_picks_fewest_then_stalest_victims():
    c = spark("s1",
              residents=[R("a", budget=0.4, idle=IDLE + 100),
                         R("b", budget=0.4, idle=IDLE + 999)],
              committed=0.8, want=0.3, free_slot=False)
    d = rank("target", [c, spark("s2", committed=0.9, want=0.3, order=1,
                                 residents=[R("x", budget=0.9, idle=1.0)])], S)
    assert d.outcome == "place" and d.unit_id == "s1"
    assert d.victims == ["b"], "one victim suffices; the stalest goes"


def test_leased_and_active_residents_are_never_victims():
    leased = spark("s1", residents=[R("a", budget=0.8, leased=True)],
                   committed=0.8, want=0.3)
    active = spark("s2", residents=[R("b", budget=0.8, idle=1.0)],
                   committed=0.8, want=0.3, order=1)
    d = rank("target", [leased, active], S)
    assert d.outcome == "no_capacity"
    assert "lease" in d.reasons["s1"] or "no idle" in d.reasons["s1"]


def test_lane_displacement_idle_unleased_only():
    idle_lane = gpu_lane("primary", usage="idle", occupant="old", order=0)
    active_lane = gpu_lane("companion", usage="active", occupant="busy", order=1)
    d = rank("target", [idle_lane, active_lane], S)
    assert d.outcome == "place" and d.unit_id == "primary"
    assert d.victims == [], "Lane.load evicts on its own — no explicit victims"

    leased_lane = gpu_lane("primary", usage="idle", occupant="old", leased=True)
    d2 = rank("target", [leased_lane, active_lane], S)
    assert d2.outcome == "no_capacity"


def test_whole_node_claim_needs_every_resident_evictable():
    all_idle = spark("s1", residents=[R("a", budget=0.3), R("b", budget=0.3)],
                     want=0.0, whole=True)
    d = rank("big", [all_idle, spark("s2", order=1, want=0.0, whole=True,
                                     residents=[R("x", budget=0.3, idle=1.0)])], S)
    assert d.unit_id == "s1" and sorted(d.victims) == ["a", "b"]

    one_active = spark("s1", residents=[R("a", budget=0.3), R("b", budget=0.3, idle=1.0)],
                       want=0.0, whole=True)
    blocked = rank("big", [one_active, spark("s2", order=1, want=0.0, whole=True,
                                             residents=[R("x", budget=0.3, idle=1.0)])], S)
    assert blocked.outcome == "no_capacity"


def test_exclude_reroutes_to_the_runner_up():
    a = spark("s1", residents=[R("target")], order=0)
    b = spark("s2", residents=[R("target")], order=1)
    assert rank("target", [a, b], S).unit_id == "s1"
    assert rank("target", [a, b], S, exclude=frozenset({"s1"})).unit_id == "s2"


def test_no_capacity_names_every_unit():
    full = spark("s1", residents=[R("a", budget=0.9, idle=1.0)], want=0.3)
    busy = gpu_lane("primary", usage="active", occupant="x", order=1)
    d = rank("target", [full, busy], S)
    assert d.outcome == "no_capacity"
    assert set(d.reasons) == {"s1", "primary"}


# --------------------------------------------------------------------------- #
# Placer — fact sweep
# --------------------------------------------------------------------------- #
class FakeSparkUnit:
    """Duck-typed just enough: declared_budgets marks it a Spark to the Placer."""

    def __init__(self, uid, entries, served=(), fail=False):
        self.cfg = SimpleNamespace(id=uid, gpu_uuid=f"spark:{uid}", enabled=True,
                                   max_models=4)
        self._entries = entries
        self._served = list(served)
        self._fail = fail
        self.registry = SimpleNamespace(
            entries=lambda: entries,
            get=lambda a: next((e for e in entries if e.alias == a), None),
            find_by_served_name=lambda n: next(
                (e for e in entries if (e.served_name or e.alias) == n), None),
        )
        self.model_activity = {}

    async def status(self):
        if self._fail:
            raise RuntimeError("node is down")
        return st(self.cfg.id, models=self._served)

    def declared_budgets(self, names):
        out = {}
        for n in names:
            e = self.registry.find_by_served_name(n)
            out[n] = e.mem_fraction if e else 0.0
        return out

    def canonical_model(self, m):
        e = self.registry.find_by_served_name(m) or self.registry.get(m)
        return e.alias if e else m

    def idle_for(self, m):
        return IDLE


def entry(alias, served=None, frac=0.3, tp=1):
    return SimpleNamespace(alias=alias, served_name=served or alias,
                           mem_fraction=frac, tp=tp, status="ok")


def make_placer(units, load_times=None):
    orch = SimpleNamespace(units={u.cfg.id: u for u in units},
                           jobs=SimpleNamespace(get=lambda _id: None))
    leases = SimpleNamespace(active_for=lambda *a, **k: None,
                             blocks_unleased=lambda *a, **k: None)
    monitor = SimpleNamespace(util_for=lambda uuid: None)
    return Placer(S, orch, leases, monitor, load_times)


async def test_placer_places_on_the_resident_unit():
    u1 = FakeSparkUnit("s1", [entry("m")], served=[])
    u2 = FakeSparkUnit("s2", [entry("m")], served=["m"])
    d = await make_placer([u1, u2]).place("m")
    assert d.outcome == "place" and d.unit_id == "s2"


async def test_placer_not_found_when_no_catalog_has_it():
    d = await make_placer([FakeSparkUnit("s1", [entry("m")])]).place("nope")
    assert d.outcome == "not_found"


async def test_placer_single_flight_sweep():
    """Two concurrent placements inside the TTL cost ONE status sweep."""
    u = FakeSparkUnit("s1", [entry("m")], served=["m"])
    calls = {"n": 0}
    real_status = u.status

    async def counted():
        calls["n"] += 1
        await asyncio.sleep(0.02)
        return await real_status()

    u.status = counted
    placer = make_placer([u])
    d1, d2 = await asyncio.gather(placer.place("m"), placer.place("m"))
    assert d1.unit_id == d2.unit_id == "s1"
    assert calls["n"] == 1, "concurrent placements must share one sweep"


async def test_placer_excludes_a_unit_whose_status_raises():
    dead = FakeSparkUnit("s1", [entry("m")], fail=True)
    alive = FakeSparkUnit("s2", [entry("m")], served=["m"])
    d = await make_placer([dead, alive]).place("m")
    assert d.unit_id == "s2", "a dead unit self-excludes instead of sinking placement"


async def test_placer_server_constraint_filters_candidates():
    u = FakeSparkUnit("s1", [entry("m")], served=["m"])
    d = await make_placer([u]).place("m", server="vllm")
    assert d.outcome == "not_found", "the /api/load server kind constrains candidates"


async def test_placer_resolves_by_alias_when_served_name_differs():
    """/api/load {lane: auto} passes ALIASES; served-name-only matching 404'd
    them (review find, 2026-07-28)."""
    u1 = FakeSparkUnit("s1", [entry("m", served="m-served")], served=[])
    u2 = FakeSparkUnit("s2", [entry("m", served="m-served")], served=["m-served"])
    p = make_placer([u1, u2])
    d = await p.place("m")                       # by alias
    assert d.outcome == "place" and d.unit_id == "s2", \
        "alias resolves AND finds the warm resident (tier 2, not a second copy)"
    d2 = await p.place("m-served")               # by served name still works
    assert d2.outcome == "place" and d2.unit_id == "s2"


async def test_invalidate_forces_a_resweep():
    u = FakeSparkUnit("s1", [entry("m")], served=["m"])
    calls = {"n": 0}
    real_status = u.status

    async def counted():
        calls["n"] += 1
        return await real_status()

    u.status = counted
    placer = make_placer([u])
    await placer.place("m")
    await placer.place("m")
    assert calls["n"] == 1, "second placement inside the TTL rides the cache"
    placer.invalidate()      # what a committed load calls
    await placer.place("m")
    assert calls["n"] == 2, "invalidate() drops the cache — the next place re-sweeps"


# --------------------------------------------------------------------------- #
# The proven-load gate + failure blocklist
# --------------------------------------------------------------------------- #
def test_unproven_unit_is_not_chosen_for_a_fresh_load():
    unproven = spark("s1", order=0)
    unproven.proven = False
    proven = spark("s2", order=1)
    d = rank("target", [unproven, proven], S)
    assert d.unit_id == "s2", "tier 3 skips the unproven unit"


def test_all_unproven_is_no_capacity_with_the_seeding_hint():
    a = spark("s1")
    a.proven = False
    b = spark("s2", order=1)
    b.proven = False
    d = rank("target", [a, b], S)
    assert d.outcome == "no_capacity"
    assert "never loaded successfully" in d.reasons["s1"]
    assert "load it once explicitly" in d.reasons["s2"]


def test_gate_off_restores_the_old_ranking():
    s_off = Settings(_env_file=None, placement_require_proven=False)
    a = spark("s1")
    a.proven = False
    b = spark("s2", order=1)
    b.proven = False
    d = rank("target", [a, b], s_off)
    assert d.outcome == "place" and d.unit_id == "s1"


def test_sole_candidate_pin_bypasses_the_gate():
    """The pin behaves as an explicit header — that's how a first-ever load gets
    seeded at all (user decision, 2026-07-28)."""
    c = spark("s1")
    c.proven = False
    d = rank("target", [c], S)
    assert d.outcome == "pin" and d.unit_id == "s1"


def test_resident_tier_is_exempt_from_the_gate():
    # proven=False can coexist with residency (samples predate the recorder);
    # a resident model is its own proof and must keep placing.
    c = spark("s1", residents=[R("target")])
    c.proven = False
    other = spark("s2", order=1)
    d = rank("target", [c, other], S)
    assert d.outcome == "place" and d.unit_id == "s1"


def test_fail_blocked_unit_is_skipped_even_when_proven():
    blocked = spark("s1", order=0)
    blocked.fail_blocked, blocked.fail_count = True, 2
    ok = spark("s2", order=1)
    d = rank("target", [blocked, ok], S)
    assert d.unit_id == "s2"

    lone_ok = spark("s2", order=1)
    lone_ok.fail_blocked, lone_ok.fail_count = True, 3
    d2 = rank("target", [blocked, lone_ok], S)
    assert d2.outcome == "no_capacity"
    assert "consecutive launch failures" in d2.reasons["s1"]


def test_gate_applies_to_displacement_tier_too():
    unproven = spark("s1", residents=[R("a", budget=0.8)], committed=0.8, want=0.3)
    unproven.proven = False
    active_only = spark("s2", residents=[R("b", budget=0.8, idle=1.0)],
                        committed=0.8, want=0.3, order=1)
    d = rank("target", [unproven, active_only], S)
    assert d.outcome == "no_capacity", "no eviction on a unit that never ran the model"


def test_tier2_resident_matches_by_alias_as_well_as_served_name():
    c = spark("s1", residents=[R("target-served", alias="target")], usage="idle")
    empty = spark("s2", order=1)
    d = rank("target", [c, empty], S)
    assert d.outcome == "place" and d.unit_id == "s1", \
        "an alias request must find its warm resident, not start a second copy"


def test_gate_facts_proof_scopes(tmp_path):
    """Spark proof is fleet-wide (shared sample key / residency on ANY spark);
    lane proof is strictly per-unit."""
    from llmconfig.load_times import LoadTimes, fail_key, spark_key

    lt = LoadTimes(path=tmp_path / "lt.yaml")
    u = FakeSparkUnit("s1", [entry("m")])
    lane = SimpleNamespace(cfg=SimpleNamespace(id="primary"))
    p = make_placer([u], load_times=lt)

    # No sample, not resident anywhere → unproven.
    proven, blocked, count, est = p._gate_facts(u, "spark", "m", True, frozenset(), [])
    assert not proven and not blocked and est is None

    # Resident on ANOTHER spark → proven (identical hardware), no estimate yet.
    proven, *_ = p._gate_facts(u, "spark", "m", True, frozenset({"m"}), [])
    assert proven

    # A sample recorded from any node → proven with an estimate.
    lt.record(spark_key("m"), 300.0, unit="spark3")
    proven, _b, _c, est = p._gate_facts(u, "spark", "m", True, frozenset(), [])
    assert proven and est == 300.0

    # The spark-wide proof does NOT leak onto a lane (per-unit key, no sample).
    proven, *_ = p._gate_facts(lane, "vllm", "m", False, frozenset({"m"}), [])
    assert not proven

    # Failure blocklist: 2 consecutive per-unit failures trip it despite proof.
    lt.record_failure(fail_key("s1", "spark", "m"))
    lt.record_failure(fail_key("s1", "spark", "m"))
    _p, blocked, count, _e = p._gate_facts(u, "spark", "m", True, frozenset(), [])
    assert blocked and count == 2
    # …but only THAT unit: s2 is clean.
    u2 = FakeSparkUnit("s2", [entry("m")])
    _p, blocked2, *_ = p._gate_facts(u2, "spark", "m", True, frozenset(), [])
    assert not blocked2


async def test_placer_end_to_end_gate_blocks_then_residency_proves(tmp_path):
    from llmconfig.load_times import LoadTimes

    lt = LoadTimes(path=tmp_path / "lt.yaml")
    empty1 = FakeSparkUnit("s1", [entry("m")])
    empty2 = FakeSparkUnit("s2", [entry("m")])
    d = await make_placer([empty1, empty2], load_times=lt).place("m")
    assert d.outcome == "no_capacity", "two candidates, neither proven — refused"
    assert "never loaded successfully" in d.reasons["s1"]

    # The same fleet with the model resident on s2: s2 wins tier 2, and if s2 is
    # excluded the residency itself proves s1 (shared GB10 proof).
    resident2 = FakeSparkUnit("s2", [entry("m")], served=["m"])
    p = make_placer([empty1, resident2], load_times=lt)
    assert (await p.place("m")).unit_id == "s2"
    d2 = await p.place("m", exclude=frozenset({"s2"}))
    # One remaining candidate → the sole-candidate pin (same effect as a place).
    assert d2.outcome in ("place", "pin") and d2.unit_id == "s1"


def test_tier3_prefers_the_faster_measured_load_bucketed():
    slow_empty = spark("s1", order=0)                 # emptier but ~8 min
    slow_empty.est_s = 480.0
    fast_fuller = spark("s2", residents=[R("x", budget=0.3)], want=0.3, order=1)
    fast_fuller.est_s = 120.0                         # ~2 min
    d = rank("target", [slow_empty, fast_fuller], S)
    assert d.unit_id == "s2", "minutes-level estimate difference beats emptiness"

    near_a = spark("s1", residents=[R("x", budget=0.4)], want=0.3, order=0)
    near_a.est_s = 130.0
    near_b = spark("s2", order=1)
    near_b.est_s = 125.0                              # same minute bucket
    d2 = rank("target", [near_a, near_b], S)
    assert d2.unit_id == "s2", "inside one bucket, emptiest-first still decides"
