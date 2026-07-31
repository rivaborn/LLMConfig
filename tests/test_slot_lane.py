"""SlotLane — multi-model GPU lane with per-slot lifecycle isolation.

The invariant under test everywhere: NOTHING a slot does can touch its sibling.
Fakes are per-slot backends over one shared world, so a cross-kill would show
up as the sibling's serving state changing.
"""
import asyncio
import time
from types import SimpleNamespace

import pytest

import llmconfig.slot_lane as slot_mod
from llmconfig.config import LaneConfig, Settings, _parse_vllm_slots
from llmconfig.gpu import GpuInfo
from llmconfig.jobs import JobManager
from llmconfig.proc import CmdResult
from llmconfig.schemas import Job, LoadRequest, UnloadRequest
from llmconfig.slot_lane import SlotLane

GiB = 1024 ** 3


class World:
    def __init__(self):
        self.serving: dict[str, str] = {}     # slot alias -> served name
        self.used_mb = 600

    def gpu(self, uuid="GPU-C"):
        return GpuInfo(found=True, uuid=uuid, total_mb=8192,
                       used_mb=self.used_mb, free_mb=8192 - self.used_mb)


class FakeSlotBackend:
    """One slot's backend; records every lifecycle call with its alias."""

    def __init__(self, world: World, alias: str, reg):
        self.w = world
        self.alias = alias
        self.reg = reg
        self.calls: list = []

    async def served(self):
        return self.w.serving.get(self.alias)

    async def served_info(self):
        from llmconfig.schemas import ServedModel
        m = self.w.serving.get(self.alias)
        return ServedModel(name=m, root=f"fake-org/{m}" if m else "",
                           context_len=18000 if m else 0)

    async def stop_instance(self, alias):
        self.calls.append(("stop_instance", alias))
        self.w.serving.pop(alias, None)

    async def serve_instance(self, alias):
        self.calls.append(("serve_instance", alias))
        e = self.reg.get(alias)
        self.w.serving[alias] = e.served_name or alias
        return CmdResult(0, "", "")

    async def wait_ready(self, served, timeout, on_log=None, alias=None):
        return self.w.serving.get(self.alias) == served

    async def journal_tail(self, alias, n=40):
        return ""

    async def aclose(self):
        pass


def entry(alias, served=None, status="ok"):
    return SimpleNamespace(alias=alias, served_name=served or alias,
                           status=status, notes="", load_timeout_s=120)


def make(monkeypatch, *, slots="surya2=11438:4600,qwen25-relay=11441:2100"):
    s = Settings(_env_file=None, evict_timeout_s=2, poll_interval_s=0.01)
    cfg = LaneConfig(
        id="companion", name="RTX 3070 Ti", gpu_uuid="GPU-C",
        vram_total_mb=8192, vram_free_baseline_mb=600,
        ollama_url="http://x", ollama_service_name="X",
        vllm_relay_url="http://127.0.0.1:11438",
        vllm_serve_script="/home/u/vllm/serve-companion.sh",
        vllm_systemd_unit="vllm-companion@",
        registry_path=None, vllm_enabled=True,
        vllm_slots=_parse_vllm_slots(slots),
    )
    entries = {"surya2": entry("surya2", served="surya-ocr-2"),
               "qwen25-relay": entry("qwen25-relay", served="qwen2.5-1.5b"),
               "smoke": entry("smoke")}          # catalogued, NO slot
    reg = SimpleNamespace(get=lambda a: entries.get(a),
                          entries=lambda: list(entries.values()))
    jobs = JobManager()
    ka = SimpleNamespace(ensure=lambda: True, alive=lambda: True)
    lane = SlotLane(s, cfg, reg, jobs, ka)
    world = World()
    lane.backends = {a: FakeSlotBackend(world, a, reg) for a in lane.slots}

    async def fake_query_gpu(s=None, uuid=None, **kw):
        return world.gpu(uuid or "GPU-C")

    monkeypatch.setattr(slot_mod, "query_gpu", fake_query_gpu)
    return lane, world, jobs


async def wait(job: Job, timeout=10.0):
    deadline = time.monotonic() + timeout
    while job.state in ("pending", "running") and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    return job


def test_slot_table_parses_and_skips_malformed():
    assert _parse_vllm_slots("a=1:2, b=3:4") == (("a", 1, 2), ("b", 3, 4))
    assert _parse_vllm_slots("a=1:2,broken,c=x:y,a=9:9,=1:2,d=0:5") == (("a", 1, 2),)
    assert _parse_vllm_slots("") == ()


async def test_load_never_touches_the_sibling_slot(monkeypatch):
    lane, world, jobs = make(monkeypatch)
    world.serving["qwen25-relay"] = "qwen2.5-1.5b"          # sibling resident

    job = await wait(lane.load(LoadRequest(server="vllm", model="surya2", lane="companion")))
    assert job.state == "succeeded", job.error
    assert world.serving.get("surya2") == "surya-ocr-2"
    assert world.serving.get("qwen25-relay") == "qwen2.5-1.5b", \
        "the relay must survive a surya2 (re)load"
    assert lane.backends["qwen25-relay"].calls == [], \
        "no lifecycle call may ever reach the sibling's backend"


async def test_unslotted_alias_refused_with_actionable_error(monkeypatch):
    lane, world, jobs = make(monkeypatch)
    job = await wait(lane.load(LoadRequest(server="vllm", model="smoke", lane="companion")))
    assert job.state == "failed"
    assert "no slot" in job.error and "COMPANION_VLLM_SLOTS" in job.error


async def test_ollama_load_refused(monkeypatch):
    lane, world, jobs = make(monkeypatch)
    job = await wait(lane.load(LoadRequest(server="ollama", model="qwen2.5:1.5b",
                                           lane="companion")))
    assert job.state == "failed"
    assert "vLLM-only" in job.error


async def test_targeted_unload_stops_only_that_slot(monkeypatch):
    lane, world, jobs = make(monkeypatch)
    world.serving = {"surya2": "surya-ocr-2", "qwen25-relay": "qwen2.5-1.5b"}

    await lane.unload(UnloadRequest(server="vllm", lane="companion",
                                    model="surya-ocr-2"))
    assert "surya2" not in world.serving
    assert world.serving.get("qwen25-relay") == "qwen2.5-1.5b"

    # Targeted unload of a non-resident: no-op (neighbour-survives contract).
    await lane.unload(UnloadRequest(server="vllm", lane="companion",
                                    model="surya-ocr-2"))
    assert world.serving.get("qwen25-relay") == "qwen2.5-1.5b"


async def test_full_unload_stops_every_slot_individually(monkeypatch):
    lane, world, jobs = make(monkeypatch)
    world.serving = {"surya2": "surya-ocr-2", "qwen25-relay": "qwen2.5-1.5b"}

    await lane.unload(UnloadRequest(server=None, lane="companion"))
    assert world.serving == {}
    assert ("stop_instance", "surya2") in lane.backends["surya2"].calls
    assert ("stop_instance", "qwen25-relay") in lane.backends["qwen25-relay"].calls


async def test_status_reports_all_residents_with_ports(monkeypatch):
    lane, world, jobs = make(monkeypatch)
    world.serving = {"surya2": "surya-ocr-2", "qwen25-relay": "qwen2.5-1.5b"}

    st = await lane.status()
    assert st.owner == "vllm" and st.vllm_up is True and st.ollama_up is False
    got = {(m.model, m.port) for m in st.loaded_models}
    assert got == {("surya-ocr-2", 11438), ("qwen2.5-1.5b", 11441)}


async def test_vllm_up_and_relay_url_for(monkeypatch):
    lane, world, jobs = make(monkeypatch)
    assert await lane.vllm_up() is False
    world.serving["qwen25-relay"] = "qwen2.5-1.5b"
    assert await lane.vllm_up() is True

    assert lane.relay_url_for("surya2") == "http://127.0.0.1:11438"
    assert lane.relay_url_for("qwen25-relay") == "http://127.0.0.1:11441"
    assert lane.relay_url_for("smoke") == lane.cfg.vllm_relay_url, \
        "slotless alias falls back to the lane relay — routing must not 500"


async def test_orchestrator_builds_slot_lane_when_slots_configured(tmp_path):
    from llmconfig.orchestrator import Orchestrator
    from llmconfig.registry import Registry
    s = Settings(_env_file=None, gpu_uuid="GPU-P", registry_path=tmp_path / "p.yaml",
                 companion_enabled=True, companion_gpu_uuid="GPU-C",
                 companion_vllm_enabled=True,
                 companion_registry_path=tmp_path / "c.yaml",
                 companion_vllm_slots="surya2=11438:4600,qwen25-relay=11441:2100")
    orch = Orchestrator(s, Registry(s.registry_path), JobManager())
    assert isinstance(orch.units["companion"], SlotLane)
    assert not isinstance(orch.units["primary"], SlotLane)
    assert "companion" in orch.lanes, "SlotLane must count as a GPU lane"
    assert list(orch.units["companion"].slots) == ["surya2", "qwen25-relay"]