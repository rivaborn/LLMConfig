"""Companion-lane behavior: the two lanes are independent, the per-lane vLLM stop is
scoped (no cross-kill), and a configured default auto-loads.
"""
import llmconfig.backends.vllm as vllm_mod
import llmconfig.lane as lane_mod
from llmconfig.backends.vllm import VllmBackend
from llmconfig.config import Settings
from llmconfig.gpu import GpuInfo
from llmconfig.jobs import JobManager
from llmconfig.orchestrator import Orchestrator
from llmconfig.proc import CmdResult
from llmconfig.registry import Registry
from llmconfig.schemas import LoadRequest, OllamaModel, UnloadRequest

GiB = 1024 ** 3


class World:
    def __init__(self):
        self.base = 300
        self.used_mb = 300
        self.ollama: dict[str, tuple[int, int]] = {}
        self.vllm = None

    def gpu(self, uuid):
        return GpuInfo(found=True, uuid=uuid, total_mb=8192, used_mb=self.used_mb, free_mb=8192 - self.used_mb)


class FakeOllama:
    def __init__(self, w):
        self.w = w
        self.calls = []

    async def up(self):
        return True

    async def ensure_running(self, wait_s=20.0):
        return True

    async def loaded(self):
        return [OllamaModel(name=n, size_bytes=s, loaded=True, size_vram_bytes=v) for n, (s, v) in self.w.ollama.items()]

    async def loaded_names(self):
        return list(self.w.ollama)

    async def unload_all(self):
        self.calls.append("unload_all")
        names = list(self.w.ollama)
        self.w.ollama.clear()
        if names:
            self.w.used_mb = self.w.base
        return names

    async def unload(self, m):
        self.calls.append("unload")
        self.w.ollama.pop(m, None)
        if not self.w.ollama:
            self.w.used_mb = self.w.base

    async def load(self, m, keep_alive=-1, num_gpu=None, timeout=900.0):
        self.calls.append(("load", m))
        self.w.ollama = {m: (2 * GiB, 2 * GiB)}
        self.w.used_mb = 2000

    async def block_count(self, m):
        return 32


class FakeVllm:
    def __init__(self, w, reg):
        self.w = w
        self.reg = reg
        self.calls = []

    async def served(self):
        return self.w.vllm

    async def up(self):
        return self.w.vllm is not None

    async def stop(self):
        self.calls.append("stop")
        self.w.vllm = None
        self.w.used_mb = self.w.base

    async def serve(self, alias):
        self.calls.append(("serve", alias))
        self.w.vllm = self.reg.served_name(alias)
        self.w.used_mb = 6000
        return CmdResult(0, "", "")

    async def wait_ready(self, served, timeout, on_log=None, alias=None):
        return self.w.vllm == served

    async def journal_tail(self, alias, n=40):
        return ""


class FakeKeepalive:
    def __init__(self):
        self.ensure_calls = 0

    def ensure(self):
        self.ensure_calls += 1
        return True

    def alive(self):
        return True

    def stop(self):
        pass


def _make(monkeypatch, tmp_path):
    s = Settings(
        _env_file=None,
        gpu_uuid="GPU-P",
        companion_enabled=True,
        companion_gpu_uuid="GPU-C",
        registry_path=tmp_path / "p.yaml",
        companion_registry_path=tmp_path / "c.yaml",
        evict_timeout_s=5,
        poll_interval_s=0.001,
    )
    jobs = JobManager()
    orch = Orchestrator(s, Registry(s.registry_path), jobs)
    wp, wc = World(), World()
    for lane, w in ((orch.primary, wp), (orch.lane("companion"), wc)):
        lane.ollama = FakeOllama(w)
        lane.vllm = FakeVllm(w, lane.registry)
        lane.keepalive = FakeKeepalive()
    worlds = {"GPU-P": wp, "GPU-C": wc}

    async def fake_query_gpu(set_=None, uuid=None):
        return worlds[uuid].gpu(uuid)

    monkeypatch.setattr(lane_mod, "query_gpu", fake_query_gpu)
    return s, orch, jobs, wp, wc


async def _run(orch, jobs, lane_id, req):
    job = orch.lane(lane_id).load(req)
    await jobs._tasks[job.id]
    return job


async def test_primary_load_leaves_companion_untouched(monkeypatch, tmp_path):
    _, orch, jobs, wp, wc = _make(monkeypatch, tmp_path)
    # companion is busy serving its own model
    wc.ollama = {"companion-model": (3 * GiB, 3 * GiB)}
    wc.used_mb = 3000

    job = await _run(orch, jobs, "primary", LoadRequest(server="ollama", model="big", lane="primary"))

    assert job.state == "succeeded", job.error
    assert "big" in wp.ollama
    # companion lane must be completely untouched
    assert wc.ollama == {"companion-model": (3 * GiB, 3 * GiB)}
    assert wc.used_mb == 3000
    assert orch.lane("companion").ollama.calls == []
    assert orch.lane("companion").vllm.calls == []


async def test_companion_load_leaves_primary_untouched(monkeypatch, tmp_path):
    _, orch, jobs, wp, wc = _make(monkeypatch, tmp_path)
    # primary is busy serving a big vLLM model
    wp.vllm = "qwen3-coder-30b"
    wp.used_mb = 20000

    job = await _run(orch, jobs, "companion", LoadRequest(server="vllm", model="qwen3-4b", lane="companion"))

    assert job.state == "succeeded", job.error
    assert wc.vllm == "qwen3-4b"
    # primary lane must be completely untouched (still serving its big model)
    assert wp.vllm == "qwen3-coder-30b"
    assert wp.used_mb == 20000
    assert orch.primary.ollama.calls == []
    assert orch.primary.vllm.calls == []


async def test_unknown_lane_rejected(monkeypatch, tmp_path):
    _, orch, jobs, _, _ = _make(monkeypatch, tmp_path)
    try:
        orch.load(LoadRequest(server="ollama", model="x", lane="nope"))
    except KeyError:
        return
    raise AssertionError("loading an unknown lane should raise")


async def test_vllm_stop_is_lane_scoped(monkeypatch, tmp_path):
    """Regression: stopping one lane's vLLM must not cross-kill the other lane.
    No global `pkill -f venv/bin/vllm`; only this lane's unit + serve script."""
    calls = []

    async def fake_run_wsl(cmd, **kw):
        calls.append(cmd)
        return CmdResult(0, "", "")

    monkeypatch.setattr(vllm_mod, "run_wsl", fake_run_wsl)
    s = Settings()
    reg = Registry(tmp_path / "r.yaml")
    companion = VllmBackend(
        s, reg,
        relay_url="http://127.0.0.1:11438",
        serve_script="/home/folar/vllm/serve-companion.sh",
        systemd_unit="vllm-companion@",
    )
    await companion.stop()
    blob = "\n".join(calls)
    assert "vllm-companion@" in blob and "serve-companion.sh" in blob
    assert "venv/bin/vllm" not in blob, "global pkill would cross-kill the primary lane's vLLM"
    assert "stop 'vllm@*'" not in blob, "companion stop must not touch the primary unit"


async def test_autoload_fires_configured_default(monkeypatch, tmp_path):
    s = Settings(
        _env_file=None,
        gpu_uuid="GPU-P",
        companion_enabled=True,
        companion_gpu_uuid="GPU-C",
        companion_default_server="ollama",
        companion_default_model="auto-me",
        registry_path=tmp_path / "p.yaml",
        companion_registry_path=tmp_path / "c.yaml",
        evict_timeout_s=5,
        poll_interval_s=0.001,
    )
    jobs = JobManager()
    orch = Orchestrator(s, Registry(s.registry_path), jobs)
    wp, wc = World(), World()
    for lane, w in ((orch.primary, wp), (orch.lane("companion"), wc)):
        lane.ollama = FakeOllama(w)
        lane.vllm = FakeVllm(w, lane.registry)
        lane.keepalive = FakeKeepalive()
    worlds = {"GPU-P": wp, "GPU-C": wc}

    async def fake_query_gpu(set_=None, uuid=None):
        return worlds[uuid].gpu(uuid)

    monkeypatch.setattr(lane_mod, "query_gpu", fake_query_gpu)

    started = orch.autoload_defaults()
    for j in started:
        await jobs._tasks[j.id]

    # only the companion had a default → exactly one load, onto the companion
    assert len(started) == 1
    assert "auto-me" in wc.ollama
    assert wp.ollama == {}
