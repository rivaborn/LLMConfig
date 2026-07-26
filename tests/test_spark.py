"""DGX Spark units — config parsing, the curated catalog, status, and loads.

Everything is faked: `run_wsl` (so no wsl.exe / sparkrun / ssh) and the node's
OpenAI endpoint via respx. No cluster required.
"""
import asyncio
import time

import httpx
import pytest
import respx

import llmconfig.backends.spark as spark_mod
from llmconfig.config import Settings, SparkConfig, _parse_spark_nodes
from llmconfig.idle import classify_usage
from llmconfig.jobs import JobManager
from llmconfig.proc import CmdResult
from llmconfig.registry import SparkRegistry
from llmconfig.schemas import LaneStatus, LoadRequest, SparkModelEntry, UnloadRequest
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
    route = respx.get(f"{BASE}/v1/models").mock(side_effect=httpx.ConnectError("down"))
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
    await wait_job(u.load(LoadRequest(server="spark", model="m1", lane="spark1", force=True)))

    run = next(c for c in calls if "sparkrun run" in c)
    assert "--cluster" in run and "--hosts" in run, "cluster supplies the ssh user"
    assert "--no-follow" in run, "without this the launch never returns"
    assert f"--port {PORT}" in run
    assert "--served-model-name served-1" in run, "pin the served name to the catalog"

    stop = next(c for c in calls if "sparkrun stop" in c)
    assert "--all" in stop, "sparkrun stop errors without a TARGET or --all"


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
