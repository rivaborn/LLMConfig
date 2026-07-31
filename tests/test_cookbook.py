"""Cookbook (`llmconfig/cookbook.py`) — snapshot, apply, default sync.

Units are minimal duck-typed fakes (the pattern from test_leases.FakeUnit) so
the apply logic is exercised without HTTP or hardware.
"""
import asyncio
import time
from types import SimpleNamespace

import pytest

from llmconfig.config import Settings
from llmconfig.cookbook import Cookbook
from llmconfig.group_state import GroupState
from llmconfig.jobs import JobManager
from llmconfig.lane_state import LaneDefaults
from llmconfig.leases import LeaseManager
from llmconfig.schemas import (Job, LaneStatus, LeaseClaimRequest, LoadedModel,
                               LoadRequest, UnloadRequest)


class Entry(SimpleNamespace):
    pass


def entry(alias, served=None, frac=0.3, needs_empty=False):
    return Entry(alias=alias, served_name=served or alias, mem_fraction=frac,
                 needs_empty_node=needs_empty, tp=1)


class FakeSpark:
    """Duck-typed SparkUnit: registry, canonical_model, multi-model residency."""

    def __init__(self, uid, entries, resident=(), jobs=None, enabled=True,
                 fail_loads=(), load_delay=0.0):
        self.cfg = SimpleNamespace(id=uid, enabled=enabled, gpu_uuid=f"spark:{uid}",
                                   max_models=4)
        self._entries = {e.alias: e for e in entries}
        self.resident = list(resident)              # aliases
        self.jobs = jobs or JobManager()
        self._lock = asyncio.Lock()
        self._active_job_id = None
        self.unloads: list[str | None] = []
        self.loads: list[str] = []
        self._fail = set(fail_loads)
        self._delay = load_delay
        self.registry = SimpleNamespace(
            get=lambda a: self._entries.get(a),
            find_by_served_name=lambda n: next(
                (e for e in self._entries.values() if e.served_name == n), None))

    def canonical_model(self, name):
        e = self._entries.get(name) or self.registry.find_by_served_name(name)
        return e.alias if e else name

    async def status(self, gpu=None):
        models = [LoadedModel(server="spark", model=self._entries[a].served_name)
                  for a in self.resident]
        return LaneStatus(id=self.cfg.id, name=self.cfg.id,
                          owner="spark" if models else "free",
                          ollama_up=False, vllm_up=False,
                          loaded=models[0] if models else None, loaded_models=models)

    async def unload(self, req: UnloadRequest):
        self.unloads.append(req.model)
        if req.model:
            self.resident = [a for a in self.resident
                             if a != self.canonical_model(req.model)]
        else:
            self.resident = []
        return await self.status()

    def load(self, req: LoadRequest) -> Job:
        job = self.jobs.create(kind=f"load:{self.cfg.id}:spark:{req.model}")

        async def body(j: Job) -> dict:
            if self._delay:
                await asyncio.sleep(self._delay)
            self.loads.append(req.model)
            if req.model in self._fail:
                raise RuntimeError(f"boom loading {req.model}")
            self.resident.append(req.model)
            return {}

        return self.jobs.start(job, body)


class FakeGroup:
    """Duck-typed SparkGroup: claims/releases through a real GroupState, so the
    cookbook's claim reads are exercised against the genuine table."""

    kind = "spark_group"

    def __init__(self, member_ids, gstate, jobs, serving=None, fail_loads=()):
        gid = "_".join(sorted(member_ids))
        self.cfg = SimpleNamespace(id=gid, enabled=True,
                                   member_ids=tuple(sorted(member_ids)))
        self.gstate = gstate
        self.jobs = jobs
        self._lock = asyncio.Lock()
        self._active_job_id = None
        self.unloads: list[str] = []
        self.loads: list[str] = []
        self._fail = set(fail_loads)
        if serving:
            gstate.claim(gid, serving, serving, self.cfg.member_ids, 8000)

    async def unload(self, req: UnloadRequest):
        self.unloads.append(req.lane)
        self.gstate.release(self.cfg.id)
        return None

    def load(self, req: LoadRequest) -> Job:
        job = self.jobs.create(kind=f"load:{self.cfg.id}:spark:{req.model}")

        async def body(j: Job) -> dict:
            self.loads.append(req.model)
            if req.model in self._fail:
                raise RuntimeError(f"boom loading {req.model}")
            self.gstate.claim(self.cfg.id, req.model, req.model,
                              self.cfg.member_ids, 8000)
            return {}

        return self.jobs.start(job, body)


def make(units, tmp_path, groups=None):
    jobs = units[0].jobs
    orch = SimpleNamespace(units={u.cfg.id: u for u in units},
                           defaults=LaneDefaults(Settings(_env_file=None),
                                                 path=tmp_path / "ld.yaml"))
    if groups is not None:
        # Mirror the real orchestrator: groups live IN orch.units too, plus the
        # groups/group_state/get_or_create_group surface the cookbook reaches for.
        gstate = groups[0].gstate
        for g in groups:
            orch.units[g.cfg.id] = g
        orch.groups = {g.cfg.id: g for g in groups}
        orch.group_state = gstate

        def get_or_create(member_ids):
            gid = "_".join(sorted(member_ids))
            if gid not in orch.groups:
                g = FakeGroup(member_ids, gstate, jobs)
                orch.groups[gid] = g
                orch.units[gid] = g
            return orch.groups[gid]

        orch.get_or_create_group = get_or_create
    leases = LeaseManager(Settings(_env_file=None), orch)
    cb = Cookbook(Settings(_env_file=None), orch, jobs, leases,
                  path=tmp_path / "cookbook.yaml")
    return cb, orch, leases


async def wait(job: Job, timeout=10.0):
    deadline = time.monotonic() + timeout
    while job.state in ("pending", "running") and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    return job


# --------------------------------------------------------------------------- #
# Snapshot
# --------------------------------------------------------------------------- #
async def test_snapshot_folds_served_names_and_records_empty_units(tmp_path):
    jobs = JobManager()
    u1 = FakeSpark("spark1", [entry("m1", served="served-1")], resident=["m1"], jobs=jobs)
    u2 = FakeSpark("spark2", [entry("m2")], resident=[], jobs=jobs)
    cb, orch, leases = make([u1, u2], tmp_path)

    st = await cb.snapshot("mine")
    assert st["units"]["spark1"] == [{"server": "spark", "model": "m1"}], \
        "served-1 folded to the loadable alias"
    assert st["units"]["spark2"] == [], "empty units recorded — apply frees them"

    # round-trips through the YAML
    again = Cookbook(Settings(_env_file=None), orch, jobs, leases,
                     path=tmp_path / "cookbook.yaml")
    assert again.get("mine")["units"]["spark2"] == []


async def test_snapshot_refused_mid_swap(tmp_path):
    u = FakeSpark("spark1", [entry("m1")], jobs=JobManager())
    cb, orch, leases = make([u], tmp_path)
    async with u._lock:
        with pytest.raises(RuntimeError, match="swap in progress"):
            await cb.snapshot("nope")


# --------------------------------------------------------------------------- #
# Apply
# --------------------------------------------------------------------------- #
async def test_apply_unloads_extras_then_loads_missing(tmp_path):
    u = FakeSpark("spark1", [entry("keep"), entry("extra"), entry("new")],
                  resident=["keep", "extra"], jobs=JobManager())
    cb, orch, leases = make([u], tmp_path)
    cb._states["s"] = {"saved_at": 0.0, "units": {"spark1": [
        {"server": "spark", "model": "keep"}, {"server": "spark", "model": "new"}]}}

    meta = await wait(cb.apply("s"))
    assert meta.state == "succeeded", meta.error
    assert sorted(u.resident) == ["keep", "new"]
    assert u.unloads == ["extra"], "only the extra was unloaded"
    assert u.loads == ["new"], "the kept resident was not reloaded"
    assert meta.result["unloaded"] == [{"unit": "spark1", "model": "extra"}]


async def test_apply_frees_an_explicitly_empty_unit(tmp_path):
    u = FakeSpark("spark1", [entry("m1")], resident=["m1"], jobs=JobManager())
    cb, orch, leases = make([u], tmp_path)
    cb._states["empty"] = {"saved_at": 0.0, "units": {"spark1": []}}

    meta = await wait(cb.apply("empty"))
    assert meta.state == "succeeded"
    assert u.resident == []


async def test_needs_empty_node_forces_full_rebuild_and_loads_it_first(tmp_path):
    u = FakeSpark("spark1", [entry("gem", frac=0.4, needs_empty=True),
                             entry("emb", frac=0.33)],
                  resident=["emb"], jobs=JobManager())
    cb, orch, leases = make([u], tmp_path)
    cb._states["s"] = {"saved_at": 0.0, "units": {"spark1": [
        {"server": "spark", "model": "emb"}, {"server": "spark", "model": "gem"}]}}

    meta = await wait(cb.apply("s"))
    assert meta.state == "succeeded", meta.error
    assert u.unloads == ["emb"], "the kept resident was rebuilt away first"
    assert u.loads == ["gem", "emb"], "needs_empty_node loads FIRST"
    assert sorted(u.resident) == ["emb", "gem"]


async def test_nonpreemptible_lease_blocks_unload_and_is_reported(tmp_path):
    u = FakeSpark("spark1", [entry("held"), entry("new", frac=0.3)],
                  resident=["held"], jobs=JobManager())
    cb, orch, leases = make([u], tmp_path)
    leases.claim(LeaseClaimRequest(unit="spark1", holder="alice", model="held",
                                   preemptible=False))
    cb._states["s"] = {"saved_at": 0.0, "units": {"spark1": [
        {"server": "spark", "model": "new"}]}}

    meta = await wait(cb.apply("s"))
    assert meta.state == "succeeded"
    assert "held" in u.resident, "the leased model survives"
    assert any(s.get("model") == "held" for s in meta.result["skipped"])


async def test_preemptible_hold_is_displaced_and_reported(tmp_path):
    u = FakeSpark("spark1", [entry("held"), entry("new")],
                  resident=["held"], jobs=JobManager())
    cb, orch, leases = make([u], tmp_path)
    leases.claim(LeaseClaimRequest(unit="spark1", holder="opencode", model="held",
                                   preemptible=True))
    cb._states["s"] = {"saved_at": 0.0, "units": {"spark1": [
        {"server": "spark", "model": "new"}]}}

    meta = await wait(cb.apply("s"))
    assert "held" not in u.resident
    assert meta.result["displaced_holds"] == [
        {"unit": "spark1", "model": "held", "holder": "opencode"}]


async def test_one_unit_failure_does_not_cancel_others(tmp_path):
    jobs = JobManager()
    bad = FakeSpark("spark1", [entry("x")], jobs=jobs, fail_loads={"x"})
    good = FakeSpark("spark2", [entry("y")], jobs=jobs)
    cb, orch, leases = make([bad, good], tmp_path)
    cb._states["s"] = {"saved_at": 0.0, "units": {
        "spark1": [{"server": "spark", "model": "x"}],
        "spark2": [{"server": "spark", "model": "y"}]}}

    meta = await wait(cb.apply("s"))
    assert meta.state == "succeeded", "the META job reports failures, not raises"
    assert good.resident == ["y"], "the healthy unit completed"
    assert any(f.get("model") == "x" for f in meta.result["failed"])


async def test_concurrent_apply_refused(tmp_path):
    u = FakeSpark("spark1", [entry("slow")], jobs=JobManager(), load_delay=0.3)
    cb, orch, leases = make([u], tmp_path)
    cb._states["s"] = {"saved_at": 0.0, "units": {"spark1": [
        {"server": "spark", "model": "slow"}]}}

    first = cb.apply("s")
    with pytest.raises(RuntimeError, match="another apply"):
        cb.apply("s")
    await wait(first)


# --------------------------------------------------------------------------- #
# Default
# --------------------------------------------------------------------------- #
async def test_set_default_syncs_lane_defaults_with_tombstones(tmp_path):
    jobs = JobManager()
    u1 = FakeSpark("spark1", [entry("m1"), entry("m2")], resident=["m1", "m2"], jobs=jobs)
    u2 = FakeSpark("spark2", [entry("m3")], resident=[], jobs=jobs)
    cb, orch, leases = make([u1, u2], tmp_path)
    await cb.snapshot("base")

    cb.set_default("base")
    assert cb.default == "base"
    assert orch.defaults.entries_or_none("spark1") == [
        {"server": "spark", "model": "m1"}, {"server": "spark", "model": "m2"}]
    assert orch.defaults.entries_or_none("spark2") == [], "empty unit -> tombstone"
    assert cb.default_in_sync() is True

    orch.defaults.add("spark2", "spark", "m3")        # user stars a model later
    assert cb.default_in_sync() is False, "drift is reported, not fought"


async def test_set_default_writes_boot_order(tmp_path):
    """LaneDefaults should READ in true boot order: needs_empty_node first, even
    when the state's stored list has it last (autoload re-sorts anyway — this is
    for the human reading lane_defaults.yaml)."""
    u = FakeSpark("spark1", [entry("emb", frac=0.33),
                             entry("rr", frac=0.35, needs_empty=True)],
                  jobs=JobManager())
    cb, orch, leases = make([u], tmp_path)
    cb._states["s"] = {"saved_at": 0.0, "units": {"spark1": [
        {"server": "spark", "model": "emb"}, {"server": "spark", "model": "rr"}]}}

    cb.set_default("s")
    assert orch.defaults.entries_or_none("spark1") == [
        {"server": "spark", "model": "rr"}, {"server": "spark", "model": "emb"}]


async def test_deleting_the_default_clears_the_marker(tmp_path):
    u = FakeSpark("spark1", [entry("m1")], resident=["m1"], jobs=JobManager())
    cb, orch, leases = make([u], tmp_path)
    await cb.snapshot("base")
    cb.set_default("base")

    assert cb.delete("base") is True
    assert cb.default == ""
    assert cb.default_in_sync() is None


# --------------------------------------------------------------------------- #
# Multi-node groups
# --------------------------------------------------------------------------- #
async def test_snapshot_records_live_group_and_old_states_still_load(tmp_path):
    jobs = JobManager()
    u1 = FakeSpark("spark1", [], jobs=jobs)
    u2 = FakeSpark("spark2", [], jobs=jobs)
    g = FakeGroup(["spark1", "spark2"], GroupState(), jobs, serving="big-model")
    cb, orch, leases = make([u1, u2], tmp_path, groups=[g])

    st = await cb.snapshot("mine")
    assert st["groups"] == {"spark1_spark2": {"model": "big-model"}}
    assert "spark1_spark2" not in st["units"], \
        "the group is its own section, never a unit row"

    # round-trips through the YAML
    again = Cookbook(Settings(_env_file=None), orch, jobs, leases,
                     path=tmp_path / "cookbook.yaml")
    assert again.get("mine")["groups"] == {"spark1_spark2": {"model": "big-model"}}

    # a pre-groups state file (no `groups:` key) still loads, as groups={}
    (tmp_path / "old.yaml").write_text(
        "default: ''\nstates:\n  legacy:\n    saved_at: 1.0\n    units:\n"
        "      spark1: []\n", encoding="utf-8")
    old = Cookbook(Settings(_env_file=None), orch, jobs, leases,
                   path=tmp_path / "old.yaml")
    assert old.get("legacy")["groups"] == {}


async def test_snapshot_refused_while_group_swaps(tmp_path):
    jobs = JobManager()
    u1 = FakeSpark("spark1", [], jobs=jobs)
    g = FakeGroup(["spark1", "spark2"], GroupState(), jobs)
    cb, orch, leases = make([u1], tmp_path, groups=[g])
    async with g._lock:
        with pytest.raises(RuntimeError, match="swap in progress"):
            await cb.snapshot("nope")


async def test_apply_tears_down_undesired_group_and_skips_member_units(tmp_path):
    jobs = JobManager()
    u1 = FakeSpark("spark1", [], jobs=jobs)
    u2 = FakeSpark("spark2", [], jobs=jobs)
    u3 = FakeSpark("spark3", [entry("m3")], jobs=jobs)
    g = FakeGroup(["spark1", "spark2"], GroupState(), jobs, serving="old-big")
    cb, orch, leases = make([u1, u2, u3], tmp_path, groups=[g])
    cb._states["s"] = {"saved_at": 0.0, "units": {
        "spark1": [], "spark2": [],
        "spark3": [{"server": "spark", "model": "m3"}]},
        "groups": {"spark1_spark2": {"model": "new-big"}}}

    meta = await wait(cb.apply("s"))
    assert meta.state == "succeeded", meta.error
    assert g.unloads == ["spark1_spark2"], "old-big torn down first"
    assert g.loads == ["new-big"], "desired group loaded last"
    assert orch.group_state.get("spark1_spark2").alias == "new-big"
    assert u3.resident == ["m3"], "single-node unit applied alongside"
    assert not u1.unloads and not u1.loads, \
        "a desired group's member is the group load's to manage"
    assert {"unit": "spark1_spark2", "model": "new-big"} in meta.result["loaded"]
    assert {"unit": "spark1_spark2", "model": "old-big"} in meta.result["unloaded"]


async def test_apply_leaves_a_group_already_serving_the_desired_model(tmp_path):
    jobs = JobManager()
    u1 = FakeSpark("spark1", [], jobs=jobs)
    g = FakeGroup(["spark1", "spark2"], GroupState(), jobs, serving="big")
    cb, orch, leases = make([u1], tmp_path, groups=[g])
    cb._states["s"] = {"saved_at": 0.0, "units": {"spark1": []},
                       "groups": {"spark1_spark2": {"model": "big"}}}

    meta = await wait(cb.apply("s"))
    assert meta.state == "succeeded", meta.error
    assert g.unloads == [] and g.loads == [], "resident group untouched"
    assert any(s.get("unit") == "spark1_spark2" and s.get("reason") == "already resident"
               for s in meta.result["skipped"])


async def test_apply_tears_down_a_group_the_state_omits(tmp_path):
    jobs = JobManager()
    u1 = FakeSpark("spark1", [entry("m1")], jobs=jobs)
    g = FakeGroup(["spark1", "spark2"], GroupState(), jobs, serving="big")
    cb, orch, leases = make([u1], tmp_path, groups=[g])
    cb._states["s"] = {"saved_at": 0.0,
                       "units": {"spark1": [{"server": "spark", "model": "m1"}]}}

    meta = await wait(cb.apply("s"))
    assert meta.state == "succeeded", meta.error
    assert g.unloads == ["spark1_spark2"], "stale group freed"
    assert orch.group_state.get("spark1_spark2") is None
    assert u1.resident == ["m1"], "freed member is loadable again in the same apply"


async def test_apply_refuses_overlapping_desired_groups(tmp_path):
    jobs = JobManager()
    u1 = FakeSpark("spark1", [], jobs=jobs)
    g = FakeGroup(["spark1", "spark2"], GroupState(), jobs)
    cb, orch, leases = make([u1], tmp_path, groups=[g])
    cb._states["s"] = {"saved_at": 0.0, "units": {},
                       "groups": {"spark1_spark2": {"model": "a"},
                                  "spark2_spark3": {"model": "b"}}}

    with pytest.raises(RuntimeError, match="both claim member 'spark2'"):
        cb.apply("s")


async def test_apply_group_load_failure_is_reported_not_raised(tmp_path):
    jobs = JobManager()
    u1 = FakeSpark("spark1", [], jobs=jobs)
    g = FakeGroup(["spark1", "spark2"], GroupState(), jobs, fail_loads={"big"})
    cb, orch, leases = make([u1], tmp_path, groups=[g])
    cb._states["s"] = {"saved_at": 0.0, "units": {"spark1": []},
                       "groups": {"spark1_spark2": {"model": "big"}}}

    meta = await wait(cb.apply("s"))
    assert meta.state == "succeeded", "the META job reports failures, not raises"
    assert any(f.get("unit") == "spark1_spark2" for f in meta.result["failed"])


async def test_apply_creates_the_group_on_first_use(tmp_path):
    jobs = JobManager()
    u1 = FakeSpark("spark1", [], jobs=jobs)
    seed = FakeGroup(["spark3", "spark4"], GroupState(), jobs)   # unrelated pair
    cb, orch, leases = make([u1], tmp_path, groups=[seed])
    cb._states["s"] = {"saved_at": 0.0, "units": {},
                       "groups": {"spark1_spark2": {"model": "big"}}}

    meta = await wait(cb.apply("s"))
    assert meta.state == "succeeded", meta.error
    assert orch.group_state.get("spark1_spark2").alias == "big", \
        "get_or_create_group instantiated the pair on demand"


async def test_apply_without_group_support_reports_not_crashes(tmp_path):
    u = FakeSpark("spark1", [entry("m1")], jobs=JobManager())
    cb, orch, leases = make([u], tmp_path)          # plain orch — no groups surface
    cb._states["s"] = {"saved_at": 0.0,
                       "units": {"spark1": [{"server": "spark", "model": "m1"}]},
                       "groups": {"spark1_spark2": {"model": "big"}}}

    meta = await wait(cb.apply("s"))
    assert meta.state == "succeeded", meta.error
    assert u.resident == ["m1"], "unit half of the state still applied"
    assert any("group support" in (f.get("error") or "")
               for f in meta.result["failed"])


async def test_set_default_ignores_groups(tmp_path):
    jobs = JobManager()
    u1 = FakeSpark("spark1", [], jobs=jobs)
    g = FakeGroup(["spark1", "spark2"], GroupState(), jobs, serving="big")
    cb, orch, leases = make([u1], tmp_path, groups=[g])
    await cb.snapshot("base")

    cb.set_default("base")
    assert orch.defaults.entries_or_none("spark1_spark2") is None, \
        "groups never reach LaneDefaults — boot autoload cannot launch them"
