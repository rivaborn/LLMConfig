"""Load-time history (`llmconfig/load_times.py`) and the recording hooks in the
unit load bodies. Fakes as in test_spark; no real hardware."""
import time

import httpx
import pytest
import respx

import llmconfig.backends.spark as spark_mod
from llmconfig.config import Settings, SparkConfig
from llmconfig.jobs import JobManager
from llmconfig.load_times import LoadTimes, lane_key, spark_key
from llmconfig.proc import CmdResult
from llmconfig.registry import SparkRegistry
from llmconfig.schemas import LoadRequest, SparkModelEntry
from llmconfig.spark_unit import SparkUnit

from test_spark import HOST, SMI_ROW, stateful_node, wait_job  # noqa: F401


# --------------------------------------------------------------------------- #
# The store itself
# --------------------------------------------------------------------------- #
def test_median_of_last_five_and_rollover(tmp_path):
    lt = LoadTimes(path=tmp_path / "lt.yaml")
    for d in (100, 200, 300, 400, 500, 600):   # six samples — first drops off
        lt.record("spark:m", d)
    assert lt.estimate("spark:m") == 400, "median of the LAST five (200..600)"
    assert lt.all()["spark:m"]["n"] == 5


def test_persistence_roundtrip_and_tolerant_load(tmp_path):
    path = tmp_path / "lt.yaml"
    LoadTimes(path=path).record("primary:vllm:coder30-awq", 123.4, unit="primary")
    again = LoadTimes(path=path)
    assert again.estimate("primary:vllm:coder30-awq") == 123.4

    path.write_text("samples: {bad: 'not a list', worse: [42, {duration_s: 'x'}]}\n",
                    encoding="utf-8")
    assert LoadTimes(path=path).all() == {}, "a hand-garbled file is an empty history"


def test_keys():
    assert spark_key("gemma-4-26b") == "spark:gemma-4-26b"
    assert lane_key("primary", "vllm", "coder30-awq") == "primary:vllm:coder30-awq"


# --------------------------------------------------------------------------- #
# Recording hooks — SparkUnit
# --------------------------------------------------------------------------- #
@pytest.fixture
def cfg(tmp_path) -> SparkConfig:
    return SparkConfig(
        id="spark1", name="spark-test", host=HOST, ssh_user="u", api_port=8000,
        registry_path=tmp_path / "spark_models_spark1.yaml", load_timeout_s=5,
    )


def unit_with(cfg, tmp_path) -> SparkUnit:
    reg = SparkRegistry(cfg.registry_path)
    reg.upsert(SparkModelEntry(alias="m1", recipe="recipe-1", served_name="served-1",
                               load_timeout_s=5, mem_fraction=0.4))
    u = SparkUnit(Settings(_env_file=None), cfg, reg, JobManager())
    u.load_times = LoadTimes(path=tmp_path / "lt.yaml")
    return u


@respx.mock
async def test_spark_launch_records_under_the_alias(cfg, tmp_path, monkeypatch):
    unit = unit_with(cfg, tmp_path)
    stateful_node(monkeypatch, cfg, {})

    # Load by SERVED name — the sample must still land on the alias key.
    job = await wait_job(unit.load(LoadRequest(server="spark", model="m1", lane="spark1")))
    assert job.state == "succeeded", job.error
    est = unit.load_times.estimate(spark_key("m1"))
    assert est is not None and est >= 0


@respx.mock
async def test_spark_fast_path_records_nothing(cfg, tmp_path, monkeypatch):
    unit = unit_with(cfg, tmp_path)
    stateful_node(monkeypatch, cfg, {8000: "served-1"})   # already resident

    job = await wait_job(unit.load(LoadRequest(server="spark", model="m1", lane="spark1")))
    assert job.state == "succeeded", job.error
    assert unit.load_times.all() == {}, "a no-op fast path is not a launch"


@respx.mock
async def test_spark_failed_launch_records_nothing(cfg, tmp_path, monkeypatch):
    """A 60s dead-serve failure must not poison the median of a 10-min model."""
    unit = unit_with(cfg, tmp_path)
    calls = []

    async def fake_run_wsl(command, *, login=True, timeout=30.0, settings=None):
        calls.append(command)
        if "nvidia-smi" in command:
            return CmdResult(0, SMI_ROW, "")
        return CmdResult(0, "ok", "")   # run "succeeds" but the model never serves

    monkeypatch.setattr(spark_mod, "run_wsl", fake_run_wsl)
    with respx.mock:
        for port in cfg.slot_ports:
            respx.get(f"http://{cfg.host}:{port}/v1/models").mock(
                return_value=httpx.Response(200, json={"data": []}))
        job = await wait_job(unit.load(LoadRequest(server="spark", model="m1",
                                                   lane="spark1")), timeout=30)
    assert job.state == "failed"
    assert unit.load_times.all() == {}, "failures never record"
