"""Multi-node SparkGroups — group state, the group unit, member claims, placement.

Everything is faked, mirroring test_spark.py: `run_wsl` (no wsl.exe / sparkrun /
ssh) and each node's OpenAI endpoint via respx. The stateful fake here spans
SEVERAL hosts — a `sparkrun run --hosts h1,h2` fills the HEAD's port, stop by
job id clears every rank — so group loads exercise the real multi-host paths.
"""
import asyncio
import time
from types import SimpleNamespace

import httpx
import pytest
import respx

import llmconfig.backends.spark as spark_mod
from llmconfig.config import Settings, SparkConfig, group_id_for
from llmconfig.group_state import GroupPlacements, GroupState
from llmconfig.jobs import JobManager
from llmconfig.placement import Placer, rank, CandidateFacts, ResidentFact
from llmconfig.proc import CmdResult
from llmconfig.registry import SparkRegistry
from llmconfig.schemas import (LaneStatus, LoadedModel, LoadRequest,
                               SparkModelEntry, UnloadRequest)
from llmconfig.spark_group import SparkGroup
from llmconfig.spark_unit import SparkUnit

HOSTS = {"spark1": "10.9.9.1", "spark2": "10.9.9.2",
         "spark3": "10.9.9.3", "spark4": "10.9.9.4"}
PORT = 8000

# served name -> recipe, mirroring the seeded catalogs below. The fake's
# `sparkrun status` derives its job listing from this so stop-by-recipe (a
# member freeing one model) and stop-by-group both resolve their job ids.
RECIPES = {"ds4-served": "@t/ds4",
           **{f"{nid}-served": f"@t/{nid}-m" for nid in HOSTS}}


def job_id(served: str) -> str:
    """Deterministic 12-hex job id per served name (matches _JOB_RE)."""
    return f"{abs(hash(served)) % 16**12:012x}"


S = Settings(_env_file=None, spark_fabric_enabled=True, swap_wait_timeout_s=2.0)


# --------------------------------------------------------------------------- #
# Fixtures — a stateful multi-host fake
# --------------------------------------------------------------------------- #
@pytest.fixture
def state():
    """{(host, port): served_name} — the fake cluster's residency."""
    return {}


@pytest.fixture
def calls(monkeypatch, state):
    """Fake run_wsl that MUTATES `state` the way sparkrun would: `run` fills the
    head's port, `status` lists one Job block per resident (with its hosts),
    `stop <id>` clears every rank of that job, `stop --all --hosts h` one host."""
    recorded = []

    def _flag(cmd: str, flag: str) -> str:
        parts = cmd.split()
        return parts[parts.index(flag) + 1] if flag in parts else ""

    async def fake_run_wsl(command, *, login=True, timeout=30.0, settings=None):
        recorded.append(command)
        if "sparkrun run" in command:
            hosts = (_flag(command, "--hosts") or "").split(",")
            port = int(_flag(command, "--port") or PORT)
            served = _flag(command, "--served-model-name")
            state[(hosts[0], port)] = served      # a tp job serves from the head
            return CmdResult(0, f"Job: launched [{job_id(served)}]", "")
        if "sparkrun status" in command:
            by_served: dict[str, list[str]] = {}
            for (h, _p), name in state.items():
                by_served.setdefault(name, []).append(h)
            lines = []
            for name, hosts_ in sorted(by_served.items()):
                rec = RECIPES.get(name, "@t/unknown")
                lines.append(f"Job: {rec}  (tp=1)  [{job_id(name)}]  "
                             f"({len(hosts_)} container(s))")
                for h in sorted(hosts_):
                    lines.append(f"  solo       {h}   Up 1 hour   img")
            return CmdResult(0, "\n".join(lines), "")
        if "sparkrun stop --all" in command:      # free one whole host
            host = _flag(command, "--hosts")
            for k in [k for k in state if k[0] == host]:
                state.pop(k)
            return CmdResult(0, "stopped", "")
        if "sparkrun stop" in command:            # by job id — cluster-wide
            sid = command.split("sparkrun stop", 1)[1].split()[0]
            for k in [k for k, v in state.items() if job_id(v) == sid]:
                state.pop(k)
            return CmdResult(0, "stopped", "")
        if "nvidia-smi" in command:
            return CmdResult(0, "GPU-abc, 122880, 40960, 81920, 17\n#MEM#\n"
                                "MemTotal: 126950000 kB\nMemAvailable: 100000000 kB\n", "")
        return CmdResult(0, "ok", "")

    monkeypatch.setattr(spark_mod, "run_wsl", fake_run_wsl)
    return recorded


def _routes(state, node_ids):
    """One respx route per (host, slot port), answering from `state`."""
    for nid in node_ids:
        host = HOSTS[nid]

        def responder(request, _h=host):
            port = request.url.port
            name = state.get((_h, port))
            data = [{"id": name}] if name else []
            return httpx.Response(200, json={"data": data})

        respx.get(f"http://{host}:{PORT}/v1/models").mock(side_effect=responder)


def make_member(tmp_path, nid, settings=S) -> SparkUnit:
    cfg = SparkConfig(id=nid, name=nid, host=HOSTS[nid], ssh_user="u",
                      api_port=PORT, max_models=1,
                      registry_path=tmp_path / f"spark_models_{nid}.yaml",
                      load_timeout_s=5)
    reg = SparkRegistry(cfg.registry_path)
    reg.upsert(SparkModelEntry(alias=f"{nid}-m", recipe=f"@t/{nid}-m",
                               served_name=f"{nid}-served", mem_fraction=0.3,
                               load_timeout_s=5))
    return SparkUnit(settings, cfg, reg, JobManager())


def make_group(tmp_path, node_ids, settings=S, gs=None, placements=None,
               jobs=None, min_nodes=2, max_nodes=4):
    """A SparkGroup + its members, wired the way the Orchestrator wires them."""
    gs = gs or GroupState()
    placements = placements or GroupPlacements(tmp_path / "spark_group_state.yaml")
    jobs = jobs or JobManager()
    members = [make_member(tmp_path, nid, settings) for nid in sorted(node_ids)]
    for m in members:
        m.group_state = gs
        m.jobs = jobs
    creg = SparkRegistry(tmp_path / "cluster.yaml")
    creg.upsert(SparkModelEntry(alias="ds4", recipe="@t/ds4", served_name="ds4-served",
                                tp=2, min_nodes=min_nodes, max_nodes=max_nodes,
                                load_timeout_s=5))
    cfg = settings.spark_group_config([m.cfg.id for m in members]) \
        if settings.spark_enabled else _group_cfg(members)
    group = SparkGroup(settings, cfg, members, creg, gs, placements, jobs)
    return group, members, gs, placements


def _group_cfg(members):
    """spark_group_config needs SPARK_ENABLED; tests build the cfg directly."""
    from llmconfig.config import SparkGroupConfig
    ids = tuple(sorted(m.cfg.id for m in members))
    return SparkGroupConfig(id=group_id_for(ids), name="Sparks " + "+".join(ids),
                            member_ids=ids, head_host=members[0].cfg.host,
                            api_port=PORT, load_timeout_s=5)


async def wait_job(job, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while job.state in ("pending", "running") and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    return job


# --------------------------------------------------------------------------- #
# GroupState / GroupPlacements
# --------------------------------------------------------------------------- #
def test_group_state_claims_and_reverse_index():
    gs = GroupState()
    gs.claim("spark1_spark2", "ds4-served", "ds4", ("spark1", "spark2"), PORT)
    assert gs.claim_for("spark1").group_id == "spark1_spark2"
    assert gs.claim_for("spark3") is None
    # A member can belong to ONE live deployment.
    with pytest.raises(RuntimeError, match="already claimed"):
        gs.claim("spark2_spark3", "x", "x", ("spark2", "spark3"), PORT)
    assert gs.release("spark1_spark2") is True
    assert gs.claim_for("spark2") is None
    assert gs.release("spark1_spark2") is False


def test_group_placements_persist_and_roundtrip(tmp_path):
    p = GroupPlacements(tmp_path / "gs.yaml")
    p.record("ds4", ["spark2", "spark1"])          # order-insensitive
    p.record("ds4", ["spark1", "spark2"])          # idempotent upsert
    p.record("ds4", ["spark1", "spark2", "spark3", "spark4"])
    fresh = GroupPlacements(tmp_path / "gs.yaml")   # reload from disk
    rows = fresh.for_alias("ds4")
    assert [r["members"] for r in rows] == [["spark1", "spark2"],
                                            ["spark1", "spark2", "spark3", "spark4"]]
    assert rows[0]["loads"] == 2
    assert set(fresh.node_sets()) == {("spark1", "spark2"),
                                      ("spark1", "spark2", "spark3", "spark4")}
    # Most-recently-used first — the cold-start order.
    assert fresh.sets_for("ds4")[0] == ("spark1", "spark2", "spark3", "spark4")


def test_group_config_validation():
    s = Settings(_env_file=None, spark_enabled=True)
    cfg = s.spark_group_config(["spark2", "spark1"])
    assert cfg.id == "spark1_spark2"
    assert cfg.member_ids == ("spark1", "spark2")
    assert cfg.head_host == "192.168.1.50"          # spark1 is the head
    assert cfg.api_base_for(8000) == "http://192.168.1.50:8000"
    with pytest.raises(ValueError, match="at least 2"):
        s.spark_group_config(["spark1"])
    with pytest.raises(ValueError, match="unknown spark node"):
        s.spark_group_config(["spark1", "nope"])


def test_fabric_links_parse_and_membership():
    """SPARK_FABRIC_LINKS describes which nodes are physically cabled."""
    s = Settings(_env_file=None, spark_enabled=True,
                 spark_fabric_links="spark1+spark2,spark3+spark4")
    assert s.fabric_links() == [frozenset({"spark1", "spark2"}),
                                frozenset({"spark3", "spark4"})]
    assert s.fabric_link_ok(["spark1", "spark2"])
    assert s.fabric_link_ok(["spark4", "spark3"])          # order-insensitive
    assert not s.fabric_link_ok(["spark1", "spark3"])      # different pairs
    assert not s.fabric_link_ok(["spark1", "spark2", "spark3"])   # spans pairs
    assert "spark1+spark2" in s.fabric_links_describe()

    # A switched fabric is ONE group naming every node, and must still admit
    # smaller jobs — hence subset rather than equality.
    sw = Settings(_env_file=None, spark_enabled=True,
                  spark_fabric_links="spark1+spark2+spark3+spark4")
    assert sw.fabric_link_ok(["spark1", "spark3"])
    assert sw.fabric_link_ok(["spark1", "spark2", "spark3", "spark4"])

    # Unset = unconstrained, i.e. exactly the pre-setting behaviour.
    un = Settings(_env_file=None, spark_enabled=True)
    assert un.fabric_links() == []
    assert un.fabric_link_ok(["spark1", "spark3"])
    # Malformed / single-member groups are dropped, never raised on.
    junk = Settings(_env_file=None, spark_enabled=True,
                    spark_fabric_links="spark1,,  ,spark2+spark3")
    assert junk.fabric_links() == [frozenset({"spark2", "spark3"})]


def test_group_config_refuses_uncabled_node_set():
    """The topology check sits in spark_group_config — the ONE chokepoint that
    both POST /api/cluster/load and the startup re-instantiation go through."""
    s = Settings(_env_file=None, spark_enabled=True,
                 spark_fabric_links="spark1+spark2,spark3+spark4")
    assert s.spark_group_config(["spark1", "spark2"]).id == "spark1_spark2"
    assert s.spark_group_config(["spark3", "spark4"]).id == "spark3_spark4"
    with pytest.raises(ValueError, match="not cabled together"):
        s.spark_group_config(["spark1", "spark3"])
    with pytest.raises(ValueError, match="not cabled together"):
        s.spark_group_config(["spark1", "spark2", "spark3", "spark4"])


# --------------------------------------------------------------------------- #
# Group load — the multi-host launch
# --------------------------------------------------------------------------- #
@respx.mock
async def test_group_load_happy_path(tmp_path, calls, state):
    group, members, gs, placements = make_group(tmp_path, ["spark1", "spark2"])
    _routes(state, ["spark1", "spark2"])

    job = group.load(LoadRequest(server="spark", model="ds4", lane=group.cfg.id))
    await wait_job(job)
    assert job.state == "succeeded", job.error

    # The launch went through the MULTI-host template with the verified flags —
    # this is the multi-host extension of
    # test_launch_command_matches_verified_sparkrun_flags, same discipline.
    run = next(c for c in calls if "sparkrun run" in c)
    assert f"--hosts {HOSTS['spark1']},{HOSTS['spark2']}" in run
    assert "--tp 2" in run
    assert "--cluster" in run, "cluster supplies the ssh user"
    assert "--no-follow" in run, "without this the launch never returns"
    assert f"--port {PORT}" in run
    assert "--served-model-name ds4-served" in run

    # Claim set, placement recorded, member locks released.
    assert gs.get("spark1_spark2").model == "ds4-served"
    assert placements.for_alias("ds4")[0]["members"] == ["spark1", "spark2"]
    for m in members:
        assert not m._lock.locked()
        assert m._active_job_id is None
    assert job.result["group"] == "spark1_spark2"
    assert job.result["port"] == PORT


@respx.mock
async def test_group_load_refused_when_fabric_off(tmp_path, calls, state):
    s_off = Settings(_env_file=None, spark_fabric_enabled=False, swap_wait_timeout_s=2.0)
    group, *_ = make_group(tmp_path, ["spark1", "spark2"], settings=s_off)
    job = group.load(LoadRequest(server="spark", model="ds4", lane=group.cfg.id))
    await wait_job(job)
    assert job.state == "failed"
    assert "SPARK_FABRIC_ENABLED" in job.error
    assert not any("sparkrun run" in c for c in calls), "nothing may launch"


@respx.mock
async def test_group_load_enforces_node_range(tmp_path, calls, state):
    group, *_ = make_group(tmp_path, ["spark1", "spark2"], min_nodes=3, max_nodes=4)
    job = group.load(LoadRequest(server="spark", model="ds4", lane=group.cfg.id))
    await wait_job(job)
    assert job.state == "failed"
    assert "supports 3-4 nodes" in job.error


@respx.mock
async def test_group_load_frees_idle_members_first(tmp_path, calls, state):
    """A member holding an idle single-node model is freed (re-validated the
    _evict_victim way) before the multi-host launch."""
    group, members, gs, _ = make_group(tmp_path, ["spark1", "spark2"])
    _routes(state, ["spark1", "spark2"])
    state[(HOSTS["spark2"], PORT)] = "spark2-served"     # idle resident on a member
    for m in members:                                     # make it look long idle
        m.last_activity = time.time() - 3600
        m.model_activity.clear()

    job = group.load(LoadRequest(server="spark", model="ds4", lane=group.cfg.id))
    await wait_job(job)
    assert job.state == "succeeded", job.error
    stop_i = next(i for i, c in enumerate(calls) if "sparkrun stop" in c)
    run_i = next(i for i, c in enumerate(calls) if "sparkrun run" in c)
    assert stop_i < run_i, "member must be freed before the launch"


@respx.mock
async def test_group_load_conflicts_on_active_member_resident(tmp_path, calls, state):
    group, members, *_ = make_group(tmp_path, ["spark1", "spark2"])
    _routes(state, ["spark1", "spark2"])
    state[(HOSTS["spark2"], PORT)] = "spark2-served"
    members[1].touch(model="spark2-m")                    # ACTIVE right now

    job = group.load(LoadRequest(server="spark", model="ds4", lane=group.cfg.id))
    await wait_job(job)
    assert job.state == "failed"
    assert "placement_conflict" in job.error
    assert not any("sparkrun run" in c for c in calls)


@respx.mock
async def test_overlapping_group_loads_do_not_deadlock(tmp_path, calls, state):
    """g12 and g23 share spark2. Ordered bounded acquisition means the loser
    queues, then fails on the under-lock claim re-check — never a deadlock."""
    gs = GroupState()
    jobs = JobManager()
    placements = GroupPlacements(tmp_path / "p.yaml")
    g12, m12, _, _ = make_group(tmp_path, ["spark1", "spark2"], gs=gs,
                                placements=placements, jobs=jobs)
    # g23 reuses spark2's unit so the LOCK is genuinely shared.
    shared = m12[1]
    m3 = make_member(tmp_path, "spark3")
    m3.group_state = gs
    creg = g12.registry
    from llmconfig.config import SparkGroupConfig
    cfg23 = SparkGroupConfig(id="spark2_spark3", name="Sparks 2+3",
                             member_ids=("spark2", "spark3"),
                             head_host=HOSTS["spark2"], api_port=PORT,
                             load_timeout_s=5)
    g23 = SparkGroup(S, cfg23, [shared, m3], creg, gs, placements, jobs)
    _routes(state, ["spark1", "spark2", "spark3"])

    j1 = g12.load(LoadRequest(server="spark", model="ds4", lane=g12.cfg.id))
    j2 = g23.load(LoadRequest(server="spark", model="ds4", lane=g23.cfg.id))
    await asyncio.gather(wait_job(j1), wait_job(j2))

    states = sorted([j1.state, j2.state])
    assert states == ["failed", "succeeded"], (j1.error, j2.error)
    loser = j1 if j1.state == "failed" else j2
    assert "placement_conflict" in loser.error or "claimed by" in loser.error
    # Every lock is free again — the definition of "no deadlock".
    for m in (*m12, m3):
        assert not m._lock.locked()


@respx.mock
async def test_disjoint_group_loads_run_concurrently(tmp_path, calls, state):
    gs = GroupState()
    jobs = JobManager()
    placements = GroupPlacements(tmp_path / "p.yaml")
    g12, *_ = make_group(tmp_path, ["spark1", "spark2"], gs=gs,
                         placements=placements, jobs=jobs)
    g34, *_ = make_group(tmp_path, ["spark3", "spark4"], gs=gs,
                         placements=placements, jobs=jobs)
    _routes(state, ["spark1", "spark2", "spark3", "spark4"])
    j1 = g12.load(LoadRequest(server="spark", model="ds4", lane=g12.cfg.id))
    j2 = g34.load(LoadRequest(server="spark", model="ds4", lane=g34.cfg.id))
    await asyncio.gather(wait_job(j1), wait_job(j2))
    assert j1.state == j2.state == "succeeded", (j1.error, j2.error)
    assert gs.get("spark1_spark2") and gs.get("spark3_spark4")


# --------------------------------------------------------------------------- #
# Members under a claim
# --------------------------------------------------------------------------- #
@respx.mock
async def test_claimed_member_status_and_refusals(tmp_path, calls, state):
    group, members, gs, _ = make_group(tmp_path, ["spark1", "spark2"])
    _routes(state, ["spark1", "spark2"])
    await wait_job(group.load(LoadRequest(server="spark", model="ds4",
                                          lane=group.cfg.id)))

    head, worker = members
    # The HEAD discovers the model on its own port — tagged, not duplicated.
    st_head = await head.status()
    assert [m.model for m in st_head.loaded_models] == ["ds4-served"]
    assert st_head.loaded_models[0].group == "spark1_spark2"
    # The WORKER serves no HTTP — the claim appears as a synthetic resident.
    st_worker = await worker.status()
    assert [m.model for m in st_worker.loaded_models] == ["ds4-served"]
    assert st_worker.loaded_models[0].group == "spark1_spark2"
    assert st_worker.loaded_models[0].port == 0
    assert st_worker.owner == "spark"

    # _admit refuses on BOTH ranks (worker slots read empty — the claim check
    # must fire before the empty-slots early return).
    for m in members:
        job = m.load(LoadRequest(server="spark", model=f"{m.cfg.id}-m", lane=m.cfg.id))
        await wait_job(job)
        assert job.state == "failed"
        assert "claimed by the multi-node deployment" in job.error

    # Member-level unload refuses too — stop --all on one rank wedges the others.
    with pytest.raises(RuntimeError, match="one rank of the multi-node"):
        await worker.unload(UnloadRequest(server="spark", lane=worker.cfg.id))

    # list_models grays everything when handed the claim (invariant 14).
    listed = await worker.spark.list_models(claim=gs.claim_for(worker.cfg.id))
    assert listed and all(not m.addable for m in listed)
    assert "Cluster tab" in listed[0].add_note


@respx.mock
async def test_group_unload_stops_by_job_id_and_releases(tmp_path, calls, state):
    group, members, gs, _ = make_group(tmp_path, ["spark1", "spark2"])
    _routes(state, ["spark1", "spark2"])
    await wait_job(group.load(LoadRequest(server="spark", model="ds4",
                                          lane=group.cfg.id)))
    assert gs.get("spark1_spark2") is not None

    st = await group.unload(UnloadRequest(server="spark", lane=group.cfg.id))
    assert gs.get("spark1_spark2") is None
    assert st.loaded is None
    stop = next(c for c in calls if f"sparkrun stop {job_id('ds4-served')}" in c)
    assert "--cluster" in stop
    # And the members report free again.
    for m in members:
        assert (await m.status()).loaded is None


@respx.mock
async def test_group_status_never_awaits_ssh(tmp_path, calls, state):
    """Invariant 9, group edition: status() costs one HTTP probe, zero run_wsl."""
    group, *_ = make_group(tmp_path, ["spark1", "spark2"])
    _routes(state, ["spark1", "spark2"])
    before = len(calls)
    st = await group.status()
    assert len(calls) == before, "group status must not shell out (SSH/sparkrun)"
    assert st.kind == "spark_group"
    assert st.loaded is None


@respx.mock
async def test_group_status_reclaims_after_restart(tmp_path, calls, state):
    """Claims are in-memory; a deployment that outlives a restart is re-claimed
    from the head probe so the members go back to reporting/refusing."""
    group, members, gs, _ = make_group(tmp_path, ["spark1", "spark2"])
    _routes(state, ["spark1", "spark2"])
    state[(HOSTS["spark1"], PORT)] = "ds4-served"   # deployment survived; no claim
    assert gs.get("spark1_spark2") is None
    st = await group.status()
    assert st.loaded is not None and st.loaded.group == "spark1_spark2"
    assert gs.claim_for("spark2").model == "ds4-served"


@respx.mock
async def test_group_reload_verifies_the_stop(tmp_path, calls, state, monkeypatch):
    """Relaunching over a lying `sparkrun stop` would let wait_ready see the OLD
    ranks — the reload path must poll the head until the old server is gone and
    refuse if it never is (parity with SparkUnit's reload guard)."""
    group, *_ = make_group(tmp_path, ["spark1", "spark2"])
    _routes(state, ["spark1", "spark2"])
    await wait_job(group.load(LoadRequest(server="spark", model="ds4",
                                          lane=group.cfg.id)))

    # An HONEST stop: force-reload tears down, verifies gone, relaunches.
    j = group.load(LoadRequest(server="spark", model="ds4", lane=group.cfg.id,
                               force=True))
    await wait_job(j)
    assert j.state == "succeeded", j.error
    assert sum(1 for c in calls if "sparkrun run" in c) == 2

    # A LYING stop: rc=0, nothing actually stopped → the reload must refuse.
    async def lying_stop(recipe=None, any_host_of=None):
        return CmdResult(0, "Workload stopped", "")
    monkeypatch.setattr(group.head.spark, "stop", lying_stop)
    fast = asyncio.sleep
    monkeypatch.setattr("llmconfig.spark_group.asyncio.sleep",
                        lambda _s: fast(0))          # 15×1s poll → instant
    j2 = group.load(LoadRequest(server="spark", model="ds4", lane=group.cfg.id,
                                force=True))
    await wait_job(j2)
    assert j2.state == "failed"
    assert "not relaunching over it" in j2.error
    assert sum(1 for c in calls if "sparkrun run" in c) == 2, "must not relaunch"


@respx.mock
async def test_reaper_never_picks_a_group_claimed_resident(tmp_path, calls, state):
    """The claimed row is one rank of a multi-node job — unload() refuses it, so
    a reaper that chose it would raise on every tick AND shadow real victims."""
    from dataclasses import replace
    from llmconfig.idle import IdleReaper
    group, members, gs, _ = make_group(tmp_path, ["spark1", "spark2"])
    _routes(state, ["spark1", "spark2"])
    await wait_job(group.load(LoadRequest(server="spark", model="ds4",
                                          lane=group.cfg.id)))

    worker = members[1]
    worker.cfg = replace(worker.cfg, idle_unload_enabled=True)
    worker.last_activity = time.time() - 7200        # far past any timeout
    worker.model_activity.clear()
    unloads: list = []

    async def recording_unload(req):
        unloads.append(req)
        raise AssertionError("a group-claimed resident must never be reaped")
    worker.unload = recording_unload

    s_on = Settings(_env_file=None, idle_unload_enabled=True,
                    idle_unload_after_min=1.0)
    reaper = IdleReaper(
        s_on,
        SimpleNamespace(units={m.cfg.id: m for m in members}, lanes={},
                        keepalive=None),
        SimpleNamespace(last_util_activity=lambda *a, **k: None),
        SimpleNamespace(blocks_idle_unload=lambda *a, **k: None),
    )
    assert await reaper._check_lane(worker) is False
    assert unloads == []


@respx.mock
async def test_snapshot_skips_group_claimed_rows(tmp_path, calls, state):
    """A cookbook state recording the group's model under a MEMBER id could never
    be applied (the model lives in the cluster catalog, not the node's)."""
    from llmconfig.cookbook import Cookbook
    group, members, gs, _ = make_group(tmp_path, ["spark1", "spark2"])
    _routes(state, ["spark1", "spark2"])
    await wait_job(group.load(LoadRequest(server="spark", model="ds4",
                                          lane=group.cfg.id)))

    orch = SimpleNamespace(units={m.cfg.id: m for m in members}
                           | {group.cfg.id: group})
    cb = Cookbook(S, orch, JobManager(),
                  SimpleNamespace(active_for=lambda *a, **k: None),
                  path=tmp_path / "cb.yaml")
    st = await cb.snapshot("mid-deployment")
    assert st["units"] == {"spark1": [], "spark2": []}, \
        "claimed rows and the group itself must both stay out of the state"


@respx.mock
async def test_group_status_rides_the_heads_probe_breaker(tmp_path, calls, state):
    """A powered-off head must not cost the connect timeout on every placer
    sweep: when the head member's breaker is open, the group presumes down
    instead of probing (respx has NO route here — a probe would blow up)."""
    group, members, *_ = make_group(tmp_path, ["spark1", "spark2"])
    head = members[0]
    head._served_fails = head._fails_before_backoff
    head._served_ts = time.time()                    # breaker freshly open
    st = await group.status()
    assert st.loaded is None
    assert st.kind == "spark_group"


# --------------------------------------------------------------------------- #
# Placement — ranking + facts
# --------------------------------------------------------------------------- #
def _spark_facts(uid, order=0, member_count=1, committed=0.0, residents=(),
                 whole_node=False, want=0.1):
    return CandidateFacts(
        unit_id=uid, kind="spark",
        status=LaneStatus(id=uid, name=uid, kind="spark", owner="free",
                          ollama_up=False, vllm_up=False),
        usage="free", server="spark", load_arg="m",
        residents=list(residents), committed=committed, want=want,
        whole_node=whole_node, free_slot=True, order=order,
        member_count=member_count,
    )


def test_rank_prefers_fewest_nodes_in_tier3():
    small = _spark_facts("spark1_spark2", order=5, member_count=2, whole_node=True, want=0.0)
    big = _spark_facts("spark1_spark2_spark3_spark4", order=1, member_count=4,
                       whole_node=True, want=0.0)
    d = rank("m", [big, small], S)
    assert d.unit_id == "spark1_spark2", "2 nodes beat 4 despite worse order"


def test_rank_prefers_fewest_nodes_in_tier4():
    idle = ResidentFact(model="x", alias="x", budget=0.0, idle_s=9999, leased=False)
    small = _spark_facts("g2", order=5, member_count=2, whole_node=True, want=0.0,
                         residents=[idle])
    big = _spark_facts("g4", order=1, member_count=4, whole_node=True, want=0.0,
                       residents=[idle])
    d = rank("m", [big, small], S)
    assert d.unit_id == "g2"
    assert d.victims == ["x"]


def test_rank_single_unit_ordering_unchanged_by_member_count_default():
    # All member_count=1 → identical ordering to the pre-multi-node sort.
    a = _spark_facts("s1", order=0, committed=0.5)
    b = _spark_facts("s2", order=1, committed=0.1)
    d = rank("m", [a, b], S)
    assert d.unit_id == "s2", "emptiest still wins among single units"


@respx.mock
async def test_placer_group_facts_union_and_claim_shielding(tmp_path, calls, state):
    """The group candidate sees its MEMBERS' residents; a claimed member's row is
    shielded like a lease, so a foreign group is never rank()-evictable."""
    gs = GroupState()
    group, members, _, _ = make_group(tmp_path, ["spark1", "spark2"], gs=gs)
    _routes(state, ["spark1", "spark2"])
    # spark2 is claimed by ANOTHER group (its model spans spark2+spark3).
    gs.claim("spark2_spark3", "other-served", "other", ("spark2", "spark3"), PORT)

    orch = SimpleNamespace(units={m.cfg.id: m for m in members} | {group.cfg.id: group},
                           jobs=SimpleNamespace(get=lambda _id: None))
    leases = SimpleNamespace(active_for=lambda *a, **k: None,
                             blocks_unleased=lambda *a, **k: None)
    monitor = SimpleNamespace(util_for=lambda uuid: None)
    placer = Placer(S, orch, leases, monitor)

    d = await placer.place("ds4-served")
    # The group is the ONLY unit resolving ds4 (disjoint catalogs) → tier-1 pin,
    # which bypasses every predicate — the real gates are under the group's
    # locks. So assert the pin AND the facts that guard the ranked tiers.
    assert d.outcome == "pin" and d.unit_id == "spark1_spark2"
    st = await placer._statuses()
    facts = placer._group_facts(group, st[group.cfg.id], "spark", "ds4",
                                group.registry.get("ds4"), 0, "ds4-served", st)
    assert facts is not None
    claimed = [r for r in facts.residents if r.model == "other-served"]
    assert claimed and claimed[0].leased is True
    assert facts.member_count == 2 and facts.whole_node is True


@respx.mock
async def test_gateway_treats_a_group_as_spark_shaped(tmp_path, calls, state):
    """A group must not fall down the LANE branch of the gateway: it has no
    cfg.vllm_relay_url / ollama, so `isinstance(lane, SparkUnit)` alone raised
    AttributeError on /v1/models and on resolve() the moment a group existed."""
    from llmconfig.openai_gateway import OpenAIGateway, _spark_shaped
    group, members, *_ = make_group(tmp_path, ["spark1", "spark2"])
    _routes(state, ["spark1", "spark2"])
    assert _spark_shaped(group) and _spark_shaped(members[0])
    assert not _spark_shaped(SimpleNamespace(kind="gpu"))

    orch = SimpleNamespace(units={m.cfg.id: m for m in members} | {group.cfg.id: group},
                           jobs=SimpleNamespace(get=lambda _id: None),
                           lane=lambda uid: group)
    gw = OpenAIGateway(orch, JobManager(), S)
    # resolve() finds the CLUSTER catalog entry and hands back the head's base URL.
    resolved = await gw.resolve(group, "ds4-served")
    assert resolved == ("spark", "ds4", f"http://{HOSTS['spark1']}:{PORT}")
    # _route() corrects to the group's port from residency.
    st = await group.status()
    st.loaded_models.append(LoadedModel(server="spark", model="ds4-served",
                                        port=PORT, group=group.cfg.id))
    assert gw._route(group, "spark", "ds4-served", st,
                     "http://unused:1") == f"http://{HOSTS['spark1']}:{PORT}"


@respx.mock
async def test_placer_resolves_multinode_model_only_on_sized_groups(tmp_path, calls, state):
    group, members, *_ = make_group(tmp_path, ["spark1", "spark2"],
                                    min_nodes=3, max_nodes=4)
    _routes(state, ["spark1", "spark2"])
    orch = SimpleNamespace(units={m.cfg.id: m for m in members} | {group.cfg.id: group},
                           jobs=SimpleNamespace(get=lambda _id: None))
    leases = SimpleNamespace(active_for=lambda *a, **k: None,
                             blocks_unleased=lambda *a, **k: None)
    monitor = SimpleNamespace(util_for=lambda uuid: None)
    placer = Placer(S, orch, leases, monitor)
    # The 2-node group is size-inadequate for a 3-4-node model; members' own
    # catalogs never contain it — so nothing resolves it.
    d = await placer.place("ds4-served")
    assert d.outcome == "not_found"


def test_api_load_accepts_a_group_lane(tmp_path, monkeypatch):
    """/api/load's kind check treated only SparkUnit as spark-shaped, so a
    `lane: auto` placement answering a multi-node model with a GROUP id — or an
    explicit group lane — got 400 "takes 'ollama' or 'vllm'"."""
    import time as _time
    from fastapi.testclient import TestClient
    import llmconfig.main as main_mod
    from llmconfig.schemas import Job

    # Keep the per-node registry seeds out of the repo's data/.
    monkeypatch.setattr("llmconfig.config.REPO_ROOT", tmp_path)
    s = Settings(
        _env_file=None, monitor_enabled=False, idle_unload_enabled=False,
        registry_path=tmp_path / "reg.yaml",
        spark_enabled=True, spark_fabric_enabled=True,
        spark_group_state_path=tmp_path / "gs.yaml",
        spark_cluster_registry_path=tmp_path / "cluster.yaml",
    )
    # A recorded placement makes the group exist at startup.
    GroupPlacements(tmp_path / "gs.yaml").record("ds4", ["spark1", "spark2"])
    monkeypatch.setattr(main_mod, "get_settings", lambda: s)
    app = main_mod.create_app()
    assert "spark1_spark2" in app.state.orch.units

    def fake_load(req):
        return Job(id="fake", kind=f"load:{req.lane}:{req.server}:{req.model}",
                   state="running", created_at=_time.time())

    monkeypatch.setattr(app.state.orch, "load", fake_load)
    with TestClient(app) as c:
        r = c.post("/api/load", json={"server": "spark", "model": "deepseek-v4-flash",
                                      "lane": "spark1_spark2"})
        assert r.status_code == 200, r.text
        # And the mismatch arm still refuses honestly.
        r = c.post("/api/load", json={"server": "vllm", "model": "x",
                                      "lane": "spark1_spark2"})
        assert r.status_code == 400
        assert "takes server 'spark'" in r.text
