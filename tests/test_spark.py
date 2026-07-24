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
