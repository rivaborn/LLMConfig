"""DGX Spark units — config parsing, the curated catalog, status, and loads.

Everything is faked: `run_wsl` (so no wsl.exe / sparkrun / ssh) and the node's
OpenAI endpoint via respx. No cluster required.
"""
import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
import respx

import llmconfig.backends.spark as spark_mod
from llmconfig.config import Settings, SparkConfig, _parse_spark_nodes
from llmconfig.idle import IdleReaper, classify_usage
from llmconfig.jobs import JobManager
from llmconfig.leases import LeaseManager
from llmconfig.proc import CmdResult
from llmconfig.registry import SparkRegistry
from llmconfig.schemas import (LaneStatus, LeaseClaimRequest, LoadRequest,
                               SparkModelEntry, UnloadRequest)
from llmconfig.spark_unit import SparkUnit

HOST = "10.9.9.9"
PORT = 8000
BASE = f"http://{HOST}:{PORT}"

SMI_ROW = "GPU-abc, 122880, 40960, 81920, 17\n"

# What a REAL GB10 prints while serving a 26B model (measured on the cluster
# 2026-07-24): the GPU shares the host's unified pool, so nvidia-smi withholds
# every memory field. Occupancy has to come from /proc/meminfo instead.
GB10_ROW = "GPU-abc, [N/A], [N/A], [N/A], 0\n"
MEMINFO = "MemTotal:       126950000 kB\nMemAvailable:    18000000 kB\n"


def gb10_result(query_row: str) -> CmdResult:
    """The combined nvidia-smi + /proc/meminfo probe, as the node returns it."""
    return CmdResult(0, query_row + "#MEM#\n" + MEMINFO, "")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def cfg(tmp_path) -> SparkConfig:
    return SparkConfig(
        id="spark1", name="spark-test", host=HOST, ssh_user="u", api_port=PORT,
        registry_path=tmp_path / "spark_models_spark1.yaml", load_timeout_s=5,
    )


class Calls(list):
    """Recorded `run_wsl` commands, with `.plan` overriding results by substring."""

    plan: dict[str, CmdResult]


@pytest.fixture
def calls(monkeypatch) -> Calls:
    recorded = Calls()
    recorded.plan = {}

    async def fake_run_wsl(command, *, login=True, timeout=30.0, settings=None):
        recorded.append(command)
        for needle, result in recorded.plan.items():
            if needle in command:
                return result
        if "nvidia-smi" in command:
            return CmdResult(0, SMI_ROW, "")
        return CmdResult(0, "ok", "")

    monkeypatch.setattr(spark_mod, "run_wsl", fake_run_wsl)
    return recorded


def make_unit(cfg, seed=True) -> SparkUnit:
    reg = SparkRegistry(cfg.registry_path)
    if seed:
        reg.upsert(SparkModelEntry(alias="m1", recipe="recipe-1", served_name="served-1",
                                   tp=1, load_timeout_s=5))
    return SparkUnit(Settings(_env_file=None), cfg, reg, JobManager())


async def wait_job(job, timeout: float = 10.0):
    """Await a fire-and-forget Job to reach a terminal state."""
    deadline = time.monotonic() + timeout
    while job.state in ("pending", "running") and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    return job


def drain_after_stop(unit, calls, port):
    """Make the fake node behave like a real one across a reload: the old model
    vanishes after `sparkrun stop` and reappears once `sparkrun run` fires —
    the drain re-probe (invariant 10: stop's rc lies) requires exactly that."""
    from llmconfig.schemas import ServedModel
    real = unit.spark.served_info

    async def fake(p=None):
        stopped = any("sparkrun stop" in c for c in calls)
        ran = any("sparkrun run" in c for c in calls)
        if p == port and stopped and not ran:
            return ServedModel()
        return await real(p)

    unit.spark.served_info = fake


def models_route(served: str | None):
    data = [{"id": served}] if served else []
    return respx.get(f"{BASE}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": data})
    )


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_parse_spark_nodes_and_skips_malformed():
    parsed = _parse_spark_nodes("a=1.1.1.1=alpha, b=2.2.2.2 , ,broken, =3.3.3.3, c=")
    assert parsed == [("a", "1.1.1.1", "alpha"), ("b", "2.2.2.2", "b")]


def test_sparks_disabled_by_default():
    s = Settings(_env_file=None)   # isolate: the box's .env sets COMPANION_ENABLED=1
    assert s.sparks() == []
    assert [c.id for c in s.units()] == ["primary"]


def test_units_includes_lanes_then_sparks():
    s = Settings(_env_file=None, spark_enabled=True, companion_enabled=True)
    ids = [c.id for c in s.units()]
    assert ids == ["primary", "companion", "spark1", "spark2", "spark3", "spark4"]
    assert [c.host for c in s.sparks()] == [
        "192.168.1.50", "192.168.1.51", "192.168.1.52", "192.168.1.53"
    ]


def test_display_name_is_ordinal_with_chassis_hostname():
    """"Spark 1 (spark-cc9b)" — the ordinal is what a human uses day to day, the
    chassis hostname is what identifies the box in NetBox and `sparkrun status`."""
    s = Settings(_env_file=None, spark_enabled=True)
    assert [c.name for c in s.sparks()] == [
        "Spark 1 (spark-cc9b)", "Spark 2 (spark-4cd0)",
        "Spark 3 (spark-b984)", "Spark 4 (spark-f04a)",
    ]


def test_display_name_without_a_hostname_is_just_the_ordinal():
    s = Settings(_env_file=None, spark_enabled=True, spark_nodes="spark7=10.0.0.7")
    assert [c.name for c in s.sparks()] == ["Spark 7"]


def test_seeded_served_names_do_not_collide_with_the_3090_alias(tmp_path):
    """The 3090 serves `gemma-4-26b` from an AWQ-4bit build at 32k ctx. A Spark
    serving that same bare name made an un-laned /v1 request resolve silently to
    the wrong weights and half the context."""
    reg = SparkRegistry(tmp_path / "seed.yaml")
    served = {e.served_name for e in reg.entries()}
    assert "gemma-4-26b" not in served, "bare name collides with the 3090's alias"
    assert "gemma-4-26b-fp8" in served


def test_spark_config_derived_fields(cfg):
    assert cfg.api_base == BASE
    assert cfg.gpu_uuid == "spark:spark1"   # synthetic — not a local nvidia-smi UUID
    assert cfg.kind == "spark"


# --------------------------------------------------------------------------- #
# Curated catalog
# --------------------------------------------------------------------------- #
def test_registry_seeds_from_packaged_default(tmp_path):
    reg = SparkRegistry(tmp_path / "cat.yaml")
    assert (tmp_path / "cat.yaml").exists()
    assert reg.entries(), "packaged seed should provide at least one recipe"


def test_registry_roundtrip(tmp_path):
    path = tmp_path / "cat.yaml"
    reg = SparkRegistry(path)
    reg.upsert(SparkModelEntry(alias="x", recipe="r", served_name="s"))
    assert SparkRegistry(path).get("x").recipe == "r"
    assert reg.remove("x") and not reg.remove("x")


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #
def models_route_with_root(served: str, root: str):
    return respx.get(f"{BASE}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": served, "root": root}]})
    )


@respx.mock
async def test_status_carries_the_backend_root(cfg, calls):
    """Two units can serve the SAME name from different weights, so the API must
    expose the backend's `root` (the real HF repo) to tell them apart."""
    models_route_with_root("gemma-4-26b", "google/gemma-4-26B-A4B-it")
    u = make_unit(cfg)
    st = await u.status()
    assert st.loaded.model == "gemma-4-26b"
    assert st.loaded.root == "google/gemma-4-26B-A4B-it"


@respx.mock
async def test_status_serving(cfg, calls):
    models_route("served-1")
    u = make_unit(cfg)
    await u.status()          # first call kicks off the background telemetry fetch
    if u._gpu_task:
        await u._gpu_task
    st = await u.status()

    assert isinstance(st, LaneStatus)
    assert st.kind == "spark" and st.host == HOST
    assert st.owner == "spark" and st.reachable is True
    assert st.loaded and st.loaded.server == "spark" and st.loaded.model == "served-1"
    assert st.ollama_up is False and st.vllm_up is False
    assert st.gpu.found and st.gpu.total_mb == 122880


@respx.mock
async def test_status_idle_but_reachable(cfg, calls):
    models_route(None)
    u = make_unit(cfg)
    await u.status()
    if u._gpu_task:
        await u._gpu_task
    st = await u.status()
    assert st.owner == "free" and st.reachable is True and st.loaded is None


@respx.mock
async def test_status_unreachable_node(cfg, calls):
    respx.get(f"{BASE}/v1/models").mock(side_effect=httpx.ConnectError("down"))
    calls.plan["nvidia-smi"] = CmdResult(255, "", "ssh: connect refused")
    u = make_unit(cfg)
    await u.status()
    if u._gpu_task:
        await u._gpu_task
    st = await u.status()
    assert st.owner == "unknown" and st.reachable is False and st.gpu.found is False


@respx.mock
async def test_status_never_awaits_ssh(cfg, calls):
    """The SSH probe must stay off the request path (it is polled every 2.5 s)."""
    models_route(None)
    u = make_unit(cfg)
    await u.status()
    assert not any("nvidia-smi" in c for c in calls), "status() must not block on ssh"
    if u._gpu_task:
        await u._gpu_task
    assert any("nvidia-smi" in c for c in calls), "telemetry should refresh in background"


def test_stats_command_quotes_the_separator():
    """The probe must survive a real shell, not just the parser.

    `echo #MEM#` unquoted makes `#` open a comment: the marker never prints and
    the grep sharing that line never runs, silently degrading the probe to a bare
    nvidia-smi — which is what left every Spark reporting 0 % memory even after
    the meminfo fallback was written. The earlier test mocked the command's
    *output*, so it could not catch a bug in the command itself.
    """
    for cmd in (spark_mod._STATS_CMD, spark_mod._METRICS_CMD):
        assert f"echo '{spark_mod._STATS_SEP}'" in cmd, f"separator must be quoted: {cmd}"
        assert "/proc/meminfo" in cmd
        # Nothing may sit unquoted after a bare '#'.
        assert "echo #" not in cmd


@respx.mock
async def test_unified_memory_falls_back_to_meminfo(cfg, calls):
    """Regression: a GB10 serving a 26B model reported vram 0 %.

    nvidia-smi returns `[N/A]` for memory.total, memory.used AND memory.free on
    GB10 — the first cut only had a fallback for `total`, so used stayed 0 and the
    UI showed a fully-loaded node as empty. Occupancy now comes from /proc/meminfo.
    """
    models_route(None)
    calls.plan["nvidia-smi"] = gb10_result(GB10_ROW)
    u = make_unit(cfg)
    await u.status()
    if u._gpu_task:
        await u._gpu_task
    g = (await u.status()).gpu

    assert g.found is True
    assert g.total_mb == 126950000 // 1024                  # from MemTotal
    assert g.used_mb == (126950000 - 18000000) // 1024       # MemTotal - MemAvailable
    assert 80 < g.vram_pct < 90, f"expected a loaded node, got {g.vram_pct}%"


@respx.mock
async def test_real_nvidia_smi_numbers_still_win(cfg, calls):
    """On a card that does report memory, nvidia-smi must take precedence."""
    models_route(None)
    calls.plan["nvidia-smi"] = gb10_result(SMI_ROW)
    u = make_unit(cfg)
    await u.status()
    if u._gpu_task:
        await u._gpu_task
    g = (await u.status()).gpu
    assert g.total_mb == 122880 and g.used_mb == 40960


@respx.mock
async def test_probe_backoff_after_repeated_failures(cfg, calls):
    # Timeout, not ConnectError: a fast refusal now means "alive, slot empty"
    # and correctly does NOT open the breaker (review 2026-07-29).
    route = respx.get(f"{BASE}/v1/models").mock(side_effect=httpx.ConnectTimeout("down"))
    u = make_unit(cfg)
    for _ in range(6):
        await u.status()
    # 3 failures trip the breaker; later polls short-circuit instead of re-probing.
    assert route.call_count == u._fails_before_backoff


# --------------------------------------------------------------------------- #
# Load / unload
# --------------------------------------------------------------------------- #
@respx.mock
async def test_load_stops_then_runs_then_waits(cfg, calls):
    models_route("served-1")
    u = make_unit(cfg)
    calls.plan["sparkrun status"] = CmdResult(0, "Job: recipe-1  (tp=1)  [aaaa0000bbbb]  (1 container(s))\n  solo       10.9.9.9   Up 1 hour   img", "")
    drain_after_stop(u, calls, 8000)
    job = u.load(LoadRequest(server="spark", model="m1", lane="spark1", force=True))
    await wait_job(job)

    assert job.state == "succeeded", job.error
    stop_i = next(i for i, c in enumerate(calls) if "sparkrun stop" in c)
    run_i = next(i for i, c in enumerate(calls) if "sparkrun run" in c)
    assert stop_i < run_i, "must stop the old workload before launching the new one"
    assert "recipe-1" in calls[run_i] and "--tp 1" in calls[run_i]
    assert job.result["server"] == "spark" and job.result["model"] == "served-1"


@respx.mock
async def test_launch_command_matches_verified_sparkrun_flags(cfg, calls):
    """Pins the three flags verified against sparkrun 0.2.40 on the live cluster.

    Each was wrong in the first cut and each fails differently:
      --cluster with --hosts  → without the cluster, sparkrun SSHes as the local
                                WSL user and every node returns Permission denied
      --no-follow             → without it `run` tails logs and never returns, so
                                the load hangs until its timeout
      --port/--served-model-name → without them the node may serve a different
                                name/port than this app probes for
    """
    models_route("served-1")
    u = make_unit(cfg)
    calls.plan["sparkrun status"] = CmdResult(0, "Job: recipe-1  (tp=1)  [aaaa0000bbbb]  (1 container(s))\n  solo       10.9.9.9   Up 1 hour   img", "")
    drain_after_stop(u, calls, 8000)
    await wait_job(u.load(LoadRequest(server="spark", model="m1", lane="spark1", force=True)))

    run = next(c for c in calls if "sparkrun run" in c)
    assert "--cluster" in run and "--hosts" in run, "cluster supplies the ssh user"
    assert "--no-follow" in run, "without this the launch never returns"
    assert f"--port {PORT}" in run
    assert "--served-model-name served-1" in run, "pin the served name to the catalog"

    stop = next(c for c in calls if "sparkrun stop" in c)
    # sparkrun stop still errors without a TARGET or --all -- unchanged. What
    # changed TWICE is how it is satisfied: first the recipe name as TARGET, and
    # now the JOB ID resolved from `sparkrun status`, because stopping by recipe
    # name prints success while stopping nothing (live, 2026-07-26).
    assert "sparkrun stop aaaa0000bbbb" in stop,         "reload must target its own sparkrun JOB ID, not the recipe name or --all"
    assert "--cluster" in stop, "without the cluster the SSH user is wrong"


def test_seeded_recipes_are_namespaced():
    """sparkrun resolves registry recipes by namespaced name (@registry/name); a
    bare guess fails with "Recipe '<name>' not found"."""
    reg = SparkRegistry(__import__("pathlib").Path(__file__).parent / "_seed_check.yaml")
    try:
        for e in reg.entries():
            assert e.recipe.startswith("@"), f"{e.alias}: recipe '{e.recipe}' is not namespaced"
    finally:
        reg.path.unlink(missing_ok=True)


@respx.mock
async def test_load_already_serving_short_circuits(cfg, calls):
    models_route("served-1")
    u = make_unit(cfg)
    job = u.load(LoadRequest(server="spark", model="m1", lane="spark1"))
    await wait_job(job)
    assert job.state == "succeeded"
    assert not any("sparkrun run" in c for c in calls), "no relaunch when already serving"


@respx.mock
async def test_load_unknown_alias_fails(cfg, calls):
    models_route(None)
    u = make_unit(cfg)
    job = u.load(LoadRequest(server="spark", model="nope", lane="spark1"))
    await wait_job(job)
    assert job.state == "failed" and "unknown Spark model" in job.error


@respx.mock
async def test_load_reports_missing_sparkrun(cfg, calls):
    models_route(None)
    calls.plan["sparkrun run"] = CmdResult(127, "", "sparkrun: not found")
    u = make_unit(cfg)
    job = u.load(LoadRequest(server="spark", model="m1", lane="spark1"))
    await wait_job(job)
    assert job.state == "failed" and "sparkrun not found" in job.error


@respx.mock
async def test_load_times_out_when_model_never_appears(cfg, calls):
    models_route(None)  # node never reports the target
    u = make_unit(cfg)
    u.s.poll_interval_s = 0.01
    job = u.load(LoadRequest(server="spark", model="m1", lane="spark1"))
    await wait_job(job)
    assert job.state == "failed" and "did not serve" in job.error


@respx.mock
async def test_unload_stops_workload(cfg, calls):
    models_route(None)
    u = make_unit(cfg)
    await u.unload(UnloadRequest(lane="spark1"))
    assert any("sparkrun stop" in c for c in calls)


# --------------------------------------------------------------------------- #
# Cross-cutting: a spark owner must count as "managed", not "free"
# --------------------------------------------------------------------------- #
def test_classify_usage_treats_spark_as_managed():
    s = Settings(_env_file=None)
    base = dict(id="spark1", name="s", kind="spark", owner="spark",
                ollama_up=False, vllm_up=False)
    idle = LaneStatus(**base, idle_s=s.usage_active_window_s + 100)
    busy = LaneStatus(**base, idle_s=1.0)
    assert classify_usage(idle, None, s) == "idle"   # would be "free" if spark were unmanaged
    assert classify_usage(busy, None, s) == "active"


@respx.mock
async def test_status_reports_the_served_context_window(cfg, calls):
    """The window a client's prompt budget must respect — vLLM/Spark serve at
    whatever --max-model-len the launch set, NOT the model's architectural max."""
    respx.get(f"{BASE}/v1/models").mock(return_value=httpx.Response(200, json={
        "data": [{"id": "gemma-4-26b-fp8", "root": "google/gemma-4-26B-A4B-it",
                  "max_model_len": 65536}]}))
    u = make_unit(cfg)
    st = await u.status()
    assert st.loaded.context_len == 65536


@respx.mock
async def test_missing_context_is_zero_not_a_crash(cfg, calls):
    """A backend that omits max_model_len must degrade to 'unknown', not raise."""
    respx.get(f"{BASE}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "m"}]}))
    u = make_unit(cfg)
    st = await u.status()
    assert st.loaded.context_len == 0 and st.loaded.root == ""


# --------------------------------------------------------------------------- #
# Multi-model residency (several workloads on one node, one per slot port)
# --------------------------------------------------------------------------- #
def slot_route(port: int, served: str | None, root: str = "", ctx: int = 0):
    """Mock ONE slot's /v1/models. Each slot runs its own vLLM process, so the
    multi-model dimension is across ports, not inside a single response."""
    data = ([{"id": served, "root": root, "max_model_len": ctx}] if served else [])
    return respx.get(f"http://{HOST}:{port}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": data})
    )


@respx.mock
async def test_status_reports_every_resident_model(cfg, calls):
    """The whole point of the change: a node holding several models reports all
    of them, not just whatever answered on the base port."""
    unit = make_unit(cfg)
    slot_route(8000, "served-1", root="org/one", ctx=65536)
    slot_route(8001, "served-2", root="org/two", ctx=32768)
    slot_route(8002, None)
    slot_route(8003, None)

    st = await unit.status()

    assert [m.model for m in st.loaded_models] == ["served-1", "served-2"]
    assert [m.port for m in st.loaded_models] == [8000, 8001]
    assert [m.context_len for m in st.loaded_models] == [65536, 32768]
    assert [m.root for m in st.loaded_models] == ["org/one", "org/two"]
    assert st.owner == "spark"


@respx.mock
async def test_loaded_stays_the_first_model_for_backwards_compatibility(cfg, calls):
    """`loaded` is the documented back-compat surface (invariant 8) — off-box
    consumers switch on it, so it must remain the primary occupant, not vanish."""
    unit = make_unit(cfg)
    slot_route(8000, "served-1")
    slot_route(8001, "served-2")
    slot_route(8002, None)
    slot_route(8003, None)

    st = await unit.status()

    assert st.loaded is not None
    assert st.loaded == st.loaded_models[0]
    assert st.loaded.model == "served-1"


@respx.mock
async def test_a_gap_in_the_slots_does_not_hide_later_models(cfg, calls):
    """Slot 0 empty must not stop slot 2 being seen — the old code read the base
    port only, so an occupied higher slot was invisible."""
    unit = make_unit(cfg)
    slot_route(8000, None)
    slot_route(8001, None)
    slot_route(8002, "served-2")
    slot_route(8003, None)

    st = await unit.status()

    assert [(m.model, m.port) for m in st.loaded_models] == [("served-2", 8002)]
    assert st.loaded.model == "served-2"
    assert st.owner == "spark"


@respx.mock
async def test_empty_node_reports_no_models(cfg, calls):
    unit = make_unit(cfg)
    for p in (8000, 8001, 8002, 8003):
        slot_route(p, None)

    st = await unit.status()

    assert st.loaded_models == []
    assert st.loaded is None
    assert st.owner in ("free", "unknown")


@respx.mock
async def test_list_models_flags_every_resident_entry_with_its_port(cfg, calls):
    """Previously exactly one catalog row could be `loaded`; the UI dropdown and
    the CLI marker both keyed off that."""
    reg = SparkRegistry(cfg.registry_path)
    reg.upsert(SparkModelEntry(alias="m1", recipe="r1", served_name="served-1"))
    reg.upsert(SparkModelEntry(alias="m2", recipe="r2", served_name="served-2"))
    reg.upsert(SparkModelEntry(alias="m3", recipe="r3", served_name="served-3"))
    unit = SparkUnit(Settings(_env_file=None), cfg, reg, JobManager())

    slot_route(8000, "served-1")
    slot_route(8001, "served-3")
    slot_route(8002, None)
    slot_route(8003, None)

    models = {m.alias: m for m in await unit.spark.list_models()}

    assert (models["m1"].loaded, models["m1"].port) == (True, 8000)
    assert (models["m3"].loaded, models["m3"].port) == (True, 8001)
    assert (models["m2"].loaded, models["m2"].port) == (False, 0)


def test_slot_ports_follow_max_models(tmp_path):
    c = SparkConfig(id="s", name="s", host=HOST, ssh_user="u", api_port=8000,
                    registry_path=tmp_path / "r.yaml", max_models=3)
    assert c.slot_ports == (8000, 8001, 8002)
    assert c.api_base_for(8002) == f"http://{HOST}:8002"
    assert c.api_base == f"http://{HOST}:8000"  # slot 0, unchanged for old callers


# --------------------------------------------------------------------------- #
# Multi-model lifecycle: co-residency, slot allocation, admission control
# --------------------------------------------------------------------------- #
class NodeState(dict):
    """{port: served_name} for a fake node, mutated by the fake sparkrun.

    Static routes are not enough for load tests: the slot must be EMPTY until the
    launch happens, otherwise the unit short-circuits on "already serving" and
    never runs the command under test.
    """


def stateful_node(monkeypatch, cfg, initial: dict[int, str] | None = None):
    """Wire respx + a fake `sparkrun` that actually mutates node state.

    Returns (state, calls). `sparkrun run --port P --served-model-name N` fills
    slot P; `sparkrun stop <recipe>` empties whichever slot that recipe fills;
    `sparkrun stop --all` empties everything.
    """
    state = NodeState(initial or {})
    calls: list[str] = []
    recipe_to_served = {
        (e.recipe or e.alias): (e.served_name or e.alias)
        for e in SparkRegistry(cfg.registry_path).entries()
    }
    served_to_recipe = {v: k for k, v in recipe_to_served.items()}

    def job_id_for(served: str) -> str:
        # Deterministic 12-hex per served name, like sparkrun's job ids.
        # Reversed first: 'served-1'/'served-2' share their leading bytes, and a
        # forward-hex prefix would collide two jobs onto one id.
        return (served[::-1].encode().hex() + "0" * 12)[:12]

    async def fake_run_wsl(command, *, login=True, timeout=30.0, settings=None):
        calls.append(command)
        if "nvidia-smi" in command:
            return CmdResult(0, SMI_ROW, "")
        if "sparkrun status" in command:
            # Mirror the real output shape: a Job line (recipe + [id]) then a
            # host line — the targeted-stop path resolves job ids from this.
            lines = []
            for prt, nm in sorted(state.items()):
                recipe = served_to_recipe.get(nm, nm)
                lines.append(f"Job: {recipe}  (tp=1)  [{job_id_for(nm)}]  (1 container(s))")
                lines.append(f"  solo       {cfg.host}    Up 1 hour   img")
            return CmdResult(0, "\n".join(lines), "")
        if "sparkrun run" in command:
            port = int(command.split("--port ")[1].split()[0])
            served = command.split("--served-model-name ")[1].split()[0]
            state[port] = served
        elif "sparkrun stop" in command:
            if "--all" in command:
                state.clear()
            else:
                target = command.split("sparkrun stop ")[1].split()[0]
                for prt, nm in list(state.items()):
                    if job_id_for(nm) == target:
                        del state[prt]
        return CmdResult(0, "ok", "")

    monkeypatch.setattr(spark_mod, "run_wsl", fake_run_wsl)

    for port in cfg.slot_ports:
        def make(p):
            def responder(request):
                nm = state.get(p)
                data = [{"id": nm, "root": "", "max_model_len": 0}] if nm else []
                return httpx.Response(200, json={"data": data})
            return responder
        respx.get(f"http://{cfg.host}:{port}/v1/models").mock(side_effect=make(port))

    return state, calls


def two_model_unit(cfg, f1=0.4, f2=0.4) -> SparkUnit:
    reg = SparkRegistry(cfg.registry_path)
    reg.upsert(SparkModelEntry(alias="m1", recipe="recipe-1", served_name="served-1",
                               load_timeout_s=5, mem_fraction=f1))
    reg.upsert(SparkModelEntry(alias="m2", recipe="recipe-2", served_name="served-2",
                               load_timeout_s=5, mem_fraction=f2))
    return SparkUnit(Settings(_env_file=None), cfg, reg, JobManager())


@respx.mock
async def test_loading_a_second_model_does_not_stop_the_first(cfg, monkeypatch):
    """The core of the change. Previously every load began `sparkrun stop --all`,
    so loading B tore down A."""
    unit = two_model_unit(cfg)
    state, calls = stateful_node(monkeypatch, cfg, {8000: "served-1"})

    job = await wait_job(unit.load(LoadRequest(server="spark", model="m2", lane="spark1")))

    assert job.state == "succeeded", job.error
    assert not any("--all" in c for c in calls),         "loading a co-resident model must never sweep the node"
    assert state == {8000: "served-1", 8001: "served-2"}, "A must survive B's load"
    run = next(c for c in calls if "sparkrun run" in c)
    assert "--port 8001" in run, "must land on the first FREE slot, not slot 0"
    assert "--gpu-mem 0.4" in run, "declared budget must reach sparkrun"


@respx.mock
async def test_second_model_is_refused_when_the_budget_will_not_fit(cfg, monkeypatch):
    """Admission control is the only thing between co-residency and an OOM at
    load: a Spark has no eviction-wait gate to observe memory beforehand."""
    unit = two_model_unit(cfg, f1=0.7, f2=0.5)   # 0.7 + 0.5 > 0.95 headroom
    state, calls = stateful_node(monkeypatch, cfg, {8000: "served-1"})

    job = await wait_job(unit.load(LoadRequest(server="spark", model="m2", lane="spark1")))

    assert job.state == "failed"
    assert "already committed" in (job.error or "")
    assert not any("sparkrun run" in c for c in calls), "must refuse before launching"
    assert state == {8000: "served-1"}, "the resident model must be untouched"


@respx.mock
async def test_unbudgeted_model_still_claims_the_whole_node(cfg, monkeypatch):
    """mem_fraction 0.0 means "unset" -- the pre-multi-model behaviour. Sharing a
    node with one is refused rather than silently overcommitting it."""
    unit = two_model_unit(cfg, f1=0.0, f2=0.4)   # A has no declared budget
    state, calls = stateful_node(monkeypatch, cfg, {8000: "served-1"})

    job = await wait_job(unit.load(LoadRequest(server="spark", model="m2", lane="spark1")))

    assert job.state == "failed"
    assert "no mem_fraction" in (job.error or "")


@respx.mock
async def test_node_full_is_a_clear_error(tmp_path, monkeypatch):
    c = SparkConfig(id="spark1", name="spark-test", host=HOST, ssh_user="u",
                    api_port=PORT, registry_path=tmp_path / "r.yaml",
                    load_timeout_s=5, max_models=2)
    unit = two_model_unit(c, f1=0.2, f2=0.2)
    unit.registry.upsert(SparkModelEntry(alias="m3", recipe="recipe-3",
                                         served_name="served-3", load_timeout_s=5,
                                         mem_fraction=0.2))
    state, calls = stateful_node(monkeypatch, c, {8000: "served-1", 8001: "served-2"})

    job = await wait_job(unit.load(LoadRequest(server="spark", model="m3", lane="spark1")))

    assert job.state == "failed"
    assert "no free slot" in (job.error or "")


@respx.mock
async def test_unload_one_model_leaves_the_others(cfg, calls):
    unit = two_model_unit(cfg)
    slot_route(8000, "served-1")
    slot_route(8001, "served-2")
    slot_route(8002, None)
    slot_route(8003, None)

    calls.plan["sparkrun status"] = CmdResult(0, "Job: recipe-2  (tp=1)  [cccc0000dddd]  (1 container(s))\n  solo       10.9.9.9   Up 1 hour   img", "")
    await unit.unload(UnloadRequest(lane="spark1", model="m2"))

    stop = next(c for c in calls if "sparkrun stop" in c)
    assert "sparkrun stop cccc0000dddd" in stop, "targeted = the recipe's JOB ID"
    assert "--all" not in stop, "a targeted unload must not free the node"


@respx.mock
async def test_unload_without_a_model_still_frees_the_whole_node(cfg, calls):
    """What the idle reaper and a lease's free_on_preempt mean by "free the unit"."""
    unit = two_model_unit(cfg)
    for p, m in ((8000, "served-1"), (8001, "served-2"), (8002, None), (8003, None)):
        slot_route(p, m)

    await unit.unload(UnloadRequest(lane="spark1"))

    stop = next(c for c in calls if "sparkrun stop" in c)
    assert "--all" in stop


@respx.mock
async def test_reloading_a_resident_model_reuses_its_slot(cfg, calls):
    unit = two_model_unit(cfg)
    slot_route(8000, "served-1")
    slot_route(8001, "served-2")
    slot_route(8002, None)
    slot_route(8003, None)
    drain_after_stop(unit, calls, 8001)

    calls.plan["sparkrun status"] = CmdResult(0, "Job: recipe-2  (tp=1)  [cccc0000dddd]  (1 container(s))\n  solo       10.9.9.9   Up 1 hour   img", "")
    job = await wait_job(
        unit.load(LoadRequest(server="spark", model="m2", lane="spark1", force=True))
    )

    assert job.state == "succeeded", job.error
    stop = next(c for c in calls if "sparkrun stop" in c)
    assert "sparkrun stop cccc0000dddd" in stop, "free only its own slot (by job id)"
    run = next(c for c in calls if "sparkrun run" in c)
    assert "--port 8001" in run, "the freed slot is the lowest free one, so it is reused"


# --------------------------------------------------------------------------- #
# Per-model idle reaping — the unit clock is the MAX across models, so reaping
# off it alone would let one busy model keep every idle neighbour resident.
# --------------------------------------------------------------------------- #
IDLE_S = 16 * 60  # past the 15-minute default


def reaper_for(unit, **overrides):
    """An IdleReaper driving exactly one Spark, with no Monitor signal."""
    settings = Settings(_env_file=None, **overrides)
    orch = SimpleNamespace(units={unit.cfg.id: unit}, lanes={}, keepalive=None)
    monitor = SimpleNamespace(last_util_activity=lambda uuid, threshold, since: None)
    # Sparks are reap-exempt by default; SparkConfig is frozen, so swap in a copy.
    unit.cfg = replace(unit.cfg, idle_unload_enabled=True)
    return IdleReaper(settings, orch, monitor, LeaseManager(settings, orch))


@respx.mock
async def test_reaping_an_idle_model_leaves_a_busy_neighbour_resident(cfg, monkeypatch):
    """The whole point: m2 goes idle while m1 keeps working, and only m2 is evicted."""
    unit = two_model_unit(cfg)
    state, calls = stateful_node(monkeypatch, cfg, {8000: "served-1", 8001: "served-2"})
    now = time.time()
    unit.last_activity = now                      # the unit clock says "busy"...
    unit.model_activity = {"m1": now, "m2": now - IDLE_S}   # ...but only m1 is

    await reaper_for(unit)._tick()

    assert state == {8000: "served-1"}, "only the idle model may be evicted"
    assert not any("--all" in c for c in calls), "reaping one model must not sweep the node"
    assert any("sparkrun stop " in c and "--all" not in c for c in calls),         "the stop must be targeted (by job id)"


@respx.mock
async def test_a_busy_model_is_not_reaped(cfg, monkeypatch):
    unit = two_model_unit(cfg)
    state, _ = stateful_node(monkeypatch, cfg, {8000: "served-1", 8001: "served-2"})
    unit.last_activity = unit.last_activity - IDLE_S
    unit.model_activity = {"m1": time.time(), "m2": time.time()}

    await reaper_for(unit)._tick()

    assert state == {8000: "served-1", 8001: "served-2"}, "both models are in use"


@respx.mock
async def test_all_idle_models_are_reaped_over_successive_ticks(cfg, monkeypatch):
    """One victim per tick — each reap gets its own lock acquisition and re-check."""
    unit = two_model_unit(cfg)
    state, _ = stateful_node(monkeypatch, cfg, {8000: "served-1", 8001: "served-2"})
    stale = time.time() - IDLE_S
    unit.last_activity = stale
    unit.model_activity = {"m1": stale, "m2": stale - 60}
    reaper = reaper_for(unit)

    await reaper._tick()
    assert state == {8000: "served-1"}, "the stalest model goes first"
    unit.model_activity["m1"] = stale  # the reap touched only the victim's clock

    await reaper._tick()
    assert state == {}, "the second tick takes the remaining idle model"


@respx.mock
async def test_a_per_model_lease_shields_only_that_model_from_the_reaper(cfg, monkeypatch):
    unit = two_model_unit(cfg)
    state, _ = stateful_node(monkeypatch, cfg, {8000: "served-1", 8001: "served-2"})
    stale = time.time() - IDLE_S
    unit.last_activity = stale
    unit.model_activity = {"m1": stale, "m2": stale}
    reaper = reaper_for(unit)
    reaper.leases.claim(LeaseClaimRequest(unit=cfg.id, holder="alice", model="m1"))

    await reaper._tick()

    assert state == {8000: "served-1"}, "the leased model survives, the unleased one does not"


@respx.mock
async def test_a_unit_wide_lease_shields_every_model(cfg, monkeypatch):
    unit = two_model_unit(cfg)
    state, _ = stateful_node(monkeypatch, cfg, {8000: "served-1", 8001: "served-2"})
    stale = time.time() - IDLE_S
    unit.last_activity = stale
    unit.model_activity = {"m1": stale, "m2": stale}
    reaper = reaper_for(unit)
    reaper.leases.claim(LeaseClaimRequest(unit=cfg.id, holder="alice"))  # no model

    await reaper._tick()

    assert state == {8000: "served-1", 8001: "served-2"}


@respx.mock
async def test_activity_is_clocked_per_model_whatever_name_the_caller_uses(cfg, monkeypatch):
    """The gateway touches with the requested id, a load with the alias, and the
    reaper reads the node's served name — all three must land on one clock."""
    unit = two_model_unit(cfg)
    stateful_node(monkeypatch, cfg, {8000: "served-1"})
    unit.touch(model="served-1")          # as the reaper/residency names it

    assert set(unit.model_activity) == {"m1"}, "clocks are keyed by catalog alias"
    assert unit.idle_for("m1") < 5 and unit.idle_for("served-1") < 5


@respx.mock
async def test_clocks_for_departed_models_are_forgotten(cfg, monkeypatch):
    """A stale clock for a model that has left would be the oldest one forever,
    defeating the reaper's cheap pre-probe guard on every tick."""
    unit = two_model_unit(cfg)
    stateful_node(monkeypatch, cfg, {8000: "served-1"})
    unit.model_activity = {"m1": time.time(), "m2": time.time() - 99999}

    await unit.status()

    assert set(unit.model_activity) == {"m1"}


async def test_launch_timeout_honours_the_recipes_budget(cfg, monkeypatch):
    """`sparkrun run` pulls (or builds) the Docker image before the container
    starts; capping it at the NODE default (900 s) timed out recipes whose own
    budget is larger (gemma declares 3600 s)."""
    reg = SparkRegistry(cfg.registry_path)
    reg.upsert(SparkModelEntry(alias="slow", recipe="r-slow", served_name="served-slow",
                               load_timeout_s=77, mem_fraction=0.4))
    unit = SparkUnit(Settings(_env_file=None), cfg, reg, JobManager())

    timeouts: dict[str, float] = {}

    async def fake_run_wsl(command, *, login=True, timeout=30.0, settings=None):
        if "sparkrun run" in command:
            timeouts["run"] = timeout
            port = int(command.split("--port ")[1].split()[0])
            state[port] = "served-slow"
        if "nvidia-smi" in command:
            return CmdResult(0, SMI_ROW, "")
        return CmdResult(0, "ok", "")

    state: dict[int, str] = {}
    monkeypatch.setattr(spark_mod, "run_wsl", fake_run_wsl)
    with respx.mock:
        for port in cfg.slot_ports:
            def make(p):
                def responder(request):
                    nm = state.get(p)
                    data = [{"id": nm, "root": "", "max_model_len": 0}] if nm else []
                    return httpx.Response(200, json={"data": data})
                return responder
            respx.get(f"http://{cfg.host}:{port}/v1/models").mock(side_effect=make(port))

        job = await wait_job(unit.load(LoadRequest(server="spark", model="slow", lane="spark1")))

    assert job.state == "succeeded", job.error
    assert timeouts["run"] == 77.0, "the launch must get the recipe budget, not the node default"


@respx.mock
async def test_unload_prunes_the_departed_models_clock(cfg, monkeypatch):
    """After the LAST model leaves, the status() prune never fires (empty probe),
    so a stale clock would defeat the reaper's cheap guard forever."""
    unit = two_model_unit(cfg)
    state, _ = stateful_node(monkeypatch, cfg, {8000: "served-1", 8001: "served-2"})
    unit.model_activity = {"m1": time.time(), "m2": time.time()}

    await unit.unload(UnloadRequest(server=None, lane="spark1", model="served-2"))
    assert set(unit.model_activity) == {"m1"}, "a targeted unload drops just that clock"

    await unit.unload(UnloadRequest(server=None, lane="spark1"))
    assert unit.model_activity == {}, "freeing the node drops every clock"


@respx.mock
async def test_admission_refuses_when_measured_free_memory_contradicts_the_budgets(cfg, monkeypatch):
    """Declared budgets can be fiction: a model launched before budgets existed
    runs at its recipe default (~0.8 of the pool). The declared sum passed while
    the node had 11 GB free, and the new vLLM wedged silently at allocation
    (spark4, 2026-07-26). Admission must believe the measurement over the claim."""
    unit = two_model_unit(cfg, f1=0.4, f2=0.18)
    state, calls = stateful_node(monkeypatch, cfg, {8000: "served-1"})
    # nvidia-smi says only ~9% of the pool is free — served-1 is way over its 0.4.
    calls_low = "GPU-abc, 122880, 111573, 11307, 17\n"
    async def fake_run_wsl(command, *, login=True, timeout=30.0, settings=None):
        if "nvidia-smi" in command:
            return CmdResult(0, calls_low, "")
        return CmdResult(0, "ok", "")
    monkeypatch.setattr(spark_mod, "run_wsl", fake_run_wsl)

    job = await wait_job(unit.load(LoadRequest(server="spark", model="m2", lane="spark1")))

    assert job.state == "failed"
    assert "actually free" in (job.error or ""), job.error
    assert state == {8000: "served-1"}, "the resident must be untouched by the refusal"


async def test_serve_status_and_real_log_come_from_inside_the_container(cfg, calls):
    """sparkrun's container is a `sleep infinity` placeholder; the server is
    exec'd inside and logs to /tmp/sparkrun_serve.log IN the container, so
    `docker logs` shows only the CUDA banner. The probes must exec in."""
    b = make_unit(cfg).spark

    calls.plan["base64 -d"] = CmdResult(
        0, "SERVE_DEAD\nAvailable KV cache memory: -10.07 GiB\n", "")
    assert await b.serve_status(8001) == "dead"
    tail = await b.logs(n=5, port=8001)
    assert "Available KV cache memory" in tail, "the REAL serve log must surface"
    # The probe crosses four quoting layers, none of which quotes survive, so
    # the script must travel base64-encoded — the visible command may contain
    # NO quote characters at all beyond the ssh template's own.
    import base64 as _b64
    probe = next(c for c in calls if "base64 -d" in c)
    payload = probe.split("echo ", 1)[1].split(" |", 1)[0].strip("'")
    script = _b64.b64decode(payload).decode()
    assert "docker exec" in script and "port.8001" in script
    assert "sparkrun_serve.pid" in script
    assert '"' not in script.split("docker exec", 1)[0], "outer loop must stay quote-free"

    calls.plan["base64 -d"] = CmdResult(0, "SERVE_ALIVE\n", "")
    assert await b.serve_status(8001) == "alive"
    calls.plan["base64 -d"] = CmdResult(0, "", "")
    assert await b.serve_status(8001) == "unknown", "no matching container = unknown, never dead"


# --------------------------------------------------------------------------- #
# Placement-driven eviction (LoadRequest.evict) — re-validated under the lock
# --------------------------------------------------------------------------- #
@respx.mock
async def test_evict_stops_the_victim_then_loads(cfg, monkeypatch):
    unit = two_model_unit(cfg, f1=0.5, f2=0.5)
    state, calls = stateful_node(monkeypatch, cfg, {8000: "served-1"})
    unit.model_activity = {"m1": time.time() - 999}      # victim is idle

    job = await wait_job(unit.load(LoadRequest(server="spark", model="m2",
                                               lane="spark1", evict=["m1"])))

    assert job.state == "succeeded", job.error
    assert "served-1" not in state.values(), "the victim was stopped"
    assert "served-2" in state.values(), "the target loaded"
    assert any("to make room" in l for l in job.log)


@respx.mock
async def test_evict_refuses_a_victim_that_became_active(cfg, monkeypatch):
    """Placement ran on a snapshot; the victim got traffic since. The unit is
    the gate: refuse with placement_conflict so the gateway re-places."""
    unit = two_model_unit(cfg, f1=0.5, f2=0.5)
    state, _ = stateful_node(monkeypatch, cfg, {8000: "served-1"})
    unit.touch(model="m1")                               # active NOW

    job = await wait_job(unit.load(LoadRequest(server="spark", model="m2",
                                               lane="spark1", evict=["m1"])))

    assert job.state == "failed"
    assert "placement_conflict" in (job.error or "")
    assert state == {8000: "served-1"}, "the active victim survives untouched"


@respx.mock
async def test_evict_refuses_a_victim_that_gained_a_lease(cfg, monkeypatch):
    unit = two_model_unit(cfg, f1=0.5, f2=0.5)
    state, _ = stateful_node(monkeypatch, cfg, {8000: "served-1"})
    unit.model_activity = {"m1": time.time() - 999}
    orch = SimpleNamespace(units={"spark1": unit})
    unit.leases = LeaseManager(Settings(_env_file=None), orch)
    unit.leases.claim(LeaseClaimRequest(unit="spark1", holder="alice", model="m1"))

    job = await wait_job(unit.load(LoadRequest(server="spark", model="m2",
                                               lane="spark1", evict=["m1"])))

    assert job.state == "failed" and "gained a lease" in (job.error or "")
    assert state == {8000: "served-1"}


@respx.mock
async def test_evict_of_an_already_gone_victim_is_a_noop(cfg, monkeypatch):
    unit = two_model_unit(cfg, f1=0.5, f2=0.5)
    state, calls = stateful_node(monkeypatch, cfg, {})    # victim already gone

    job = await wait_job(unit.load(LoadRequest(server="spark", model="m2",
                                               lane="spark1", evict=["m1"])))

    assert job.state == "succeeded", job.error
    assert not any("sparkrun stop" in c for c in calls), "nothing to stop"


def test_declared_budgets_matches_admit_arithmetic(cfg):
    unit = two_model_unit(cfg, f1=0.4, f2=0.18)
    b = unit.declared_budgets(["served-1", "served-2", "stranger"])
    assert b == {"served-1": 0.4, "served-2": 0.18, "stranger": 0.0},         "unknown residents read as whole-node claims, exactly as _admit treats them"


# --------------------------------------------------------------------------- #
# Fit-aware catalog — addable computed BESIDE the admission arithmetic
# --------------------------------------------------------------------------- #
@respx.mock
async def test_list_models_addable_matches_admit(cfg, monkeypatch):
    """For every non-resident, non-tp>1, non-needs-empty entry: addable=False
    exactly when a load would be refused by the declared half of _admit."""
    reg = SparkRegistry(cfg.registry_path)
    reg.upsert(SparkModelEntry(alias="big", recipe="r-big", served_name="served-big",
                               mem_fraction=0.6))
    reg.upsert(SparkModelEntry(alias="small", recipe="r-small", served_name="served-small",
                               mem_fraction=0.3))
    reg.upsert(SparkModelEntry(alias="huge", recipe="r-huge", served_name="served-huge",
                               mem_fraction=0.9))
    reg.upsert(SparkModelEntry(alias="whole", recipe="r-whole", served_name="served-whole",
                               mem_fraction=0.0))
    reg.upsert(SparkModelEntry(alias="empty-only", recipe="r-eo", served_name="served-eo",
                               mem_fraction=0.3, needs_empty_node=True))
    unit = SparkUnit(Settings(_env_file=None), cfg, reg, JobManager())
    stateful_node(monkeypatch, cfg, {8000: "served-big"})   # 0.6 committed

    models = {m.alias: m for m in await unit.spark.list_models()}

    assert models["big"].addable and "reload" in models["big"].add_note
    assert models["small"].addable, "0.6 + 0.3 <= 0.95"
    assert not models["huge"].addable and "needs 0.90" in models["huge"].add_note
    assert not models["whole"].addable and "whole-node" in models["whole"].add_note
    assert not models["empty-only"].addable and "EMPTY node" in models["empty-only"].add_note


@respx.mock
async def test_list_models_addable_on_an_empty_node(cfg, monkeypatch):
    reg = SparkRegistry(cfg.registry_path)
    reg.upsert(SparkModelEntry(alias="whole", recipe="r", served_name="served-w",
                               mem_fraction=0.0))
    reg.upsert(SparkModelEntry(alias="eo", recipe="r2", served_name="served-eo",
                               mem_fraction=0.3, needs_empty_node=True))
    unit = SparkUnit(Settings(_env_file=None), cfg, reg, JobManager())
    stateful_node(monkeypatch, cfg, {})

    models = {m.alias: m for m in await unit.spark.list_models()}
    assert models["whole"].addable, "an empty node takes anything"
    assert models["eo"].addable and "FIRST" in models["eo"].add_note


@respx.mock
async def test_list_models_unbudgeted_resident_blocks_everything(cfg, monkeypatch):
    reg = SparkRegistry(cfg.registry_path)
    reg.upsert(SparkModelEntry(alias="old", recipe="r", served_name="served-old",
                               mem_fraction=0.0))
    reg.upsert(SparkModelEntry(alias="small", recipe="r2", served_name="served-small",
                               mem_fraction=0.2))
    unit = SparkUnit(Settings(_env_file=None), cfg, reg, JobManager())
    stateful_node(monkeypatch, cfg, {8000: "served-old"})

    models = {m.alias: m for m in await unit.spark.list_models()}
    assert not models["small"].addable
    assert "served-old" in models["small"].add_note


# --------------------------------------------------------------------------- #
# Targeted stop resolves the sparkrun JOB ID — stopping by recipe name lies
# --------------------------------------------------------------------------- #
STATUS_OUT = (
    "Job: @official/emb-vllm  (tp=1)  [aaaa11112222]  (1 container(s))\n"
    "  solo       10.9.9.9        Up 9 hours   tf5\n"
    "Job: @official/emb-vllm  (tp=1)  [bbbb33334444]  (1 container(s))\n"
    "  solo       10.9.9.10       Up 2 hours   tf5\n"
    "Job: @eugr/other  (tp=1)  [cccc55556666]  (1 container(s))\n"
    "  solo       10.9.9.9        Up 1 hour    vllm-node\n"
)


async def test_targeted_stop_resolves_the_job_id_on_this_host(cfg, calls):
    """`sparkrun stop <recipe> --hosts` prints success and stops NOTHING (live,
    2026-07-26). The stop must go by the job id from `sparkrun status`, matched
    to THIS host — the same recipe runs on other nodes under other ids."""
    b = make_unit(cfg).spark
    calls.plan["sparkrun status"] = CmdResult(0, STATUS_OUT, "")

    r = await b.stop(recipe="@official/emb-vllm")
    assert r.ok
    stop_cmd = calls[-1]
    assert "sparkrun stop aaaa11112222" in stop_cmd,         f"must stop THIS host's job id, got: {stop_cmd}"
    assert "bbbb33334444" not in stop_cmd, "the other node's job must be untouched"
    assert "--cluster" in stop_cmd, "without --cluster the SSH user is wrong"


async def test_targeted_stop_with_no_tracked_job_is_a_clean_noop(cfg, calls):
    b = make_unit(cfg).spark
    calls.plan["sparkrun status"] = CmdResult(0, "", "")

    r = await b.stop(recipe="@official/emb-vllm")
    assert r.ok and "no tracked sparkrun job" in r.out
    assert not any("sparkrun stop" in c for c in calls), "nothing was stopped"


async def test_stop_all_still_uses_dash_all(cfg, calls):
    b = make_unit(cfg).spark
    await b.stop()
    assert any("sparkrun stop" in c and "--all" in c for c in calls)
