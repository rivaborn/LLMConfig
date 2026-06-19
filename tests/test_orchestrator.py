"""Orchestrator logic with in-memory fakes for the two backends + GPU.

Exercises the core guarantees: eviction before load, the VRAM-free gate, and the
pack-then-spill verification — without touching wsl.exe / nvidia-smi / Ollama.
"""
import llmconfig.lane as lane_mod
from llmconfig.config import Settings
from llmconfig.gpu import GpuInfo
from llmconfig.jobs import JobManager
from llmconfig.orchestrator import Orchestrator
from llmconfig.proc import CmdResult
from llmconfig.registry import Registry
from llmconfig.schemas import LoadRequest, OllamaModel, UnloadRequest

GiB = 1024 ** 3
BASE_MB = 400  # driver baseline (GPU "free")


class World:
    def __init__(self):
        self.vllm_served = None
        self.ollama: dict[str, tuple[int, int]] = {}  # name -> (size_bytes, vram_bytes)
        self.used_mb = BASE_MB
        self.next_load = (20 * GiB, 20 * GiB)  # (size, vram) of the next Ollama load
        self.blocks = 64

    def gpu(self) -> GpuInfo:
        return GpuInfo(found=True, uuid="GPU-x", total_mb=24576, used_mb=self.used_mb, free_mb=24576 - self.used_mb)


class FakeOllama:
    def __init__(self, w: World):
        self.w = w
        self.running = True

    async def up(self):
        return self.running

    async def ensure_running(self, wait_s=20.0):
        self.running = True
        return True

    async def loaded(self):
        return [OllamaModel(name=n, size_bytes=s, loaded=True, size_vram_bytes=v) for n, (s, v) in self.w.ollama.items()]

    async def loaded_names(self):
        return list(self.w.ollama)

    async def unload_all(self):
        names = list(self.w.ollama)
        self.w.ollama.clear()
        if names:
            self.w.used_mb = BASE_MB
        return names

    async def unload(self, model):
        self.w.ollama.pop(model, None)
        if not self.w.ollama:
            self.w.used_mb = BASE_MB

    async def load(self, model, keep_alive=-1, num_gpu=None, timeout=900.0):
        size, vram = self.w.next_load
        self.w.ollama = {model: (size, vram)}
        self.w.used_mb = max(BASE_MB, vram // (1024 * 1024))

    async def block_count(self, model):
        return self.w.blocks


class FakeVllm:
    def __init__(self, w: World, reg: Registry):
        self.w = w
        self.reg = reg

    async def served(self):
        return self.w.vllm_served

    async def up(self):
        return self.w.vllm_served is not None

    async def stop(self):
        self.w.vllm_served = None
        self.w.used_mb = BASE_MB

    async def serve(self, alias):
        self.w.vllm_served = self.reg.served_name(alias)
        self.w.used_mb = 20000
        return CmdResult(0, "", "")

    async def wait_ready(self, served_name, timeout, on_log=None, alias=None):
        return self.w.vllm_served == served_name

    async def journal_tail(self, alias, n=40):
        return "fake journal"


class FakeKeepalive:
    def __init__(self):
        self.ensure_calls = 0
        self.stopped = False

    def ensure(self):
        self.ensure_calls += 1
        return True

    def alive(self):
        return self.ensure_calls > 0 and not self.stopped

    def stop(self):
        self.stopped = True


def _make(monkeypatch, tmp_path):
    world = World()
    registry = Registry(tmp_path / "reg.yaml")
    settings = Settings(gpu_uuid="GPU-x", evict_timeout_s=5, poll_interval_s=0.01)
    jobs = JobManager()
    orch = Orchestrator(settings, registry, jobs)
    # Arbitration lives on the lane; swap in the fakes there.
    lane = orch.primary
    lane.ollama = FakeOllama(world)
    lane.vllm = FakeVllm(world, registry)
    lane.keepalive = FakeKeepalive()

    async def fake_query_gpu(s=None, uuid=None):
        return world.gpu()

    monkeypatch.setattr(lane_mod, "query_gpu", fake_query_gpu)
    return world, orch, jobs


async def _run_load(orch, jobs, req):
    job = orch.load(req)
    await jobs._tasks[job.id]
    return job


async def test_load_vllm_evicts_ollama(monkeypatch, tmp_path):
    world, orch, jobs = _make(monkeypatch, tmp_path)
    world.ollama = {"qwen3:32b": (20 * GiB, 20 * GiB)}
    world.used_mb = 20000

    job = await _run_load(orch, jobs, LoadRequest(server="vllm", model="coder30-awq"))

    assert job.state == "succeeded", job.error
    assert world.ollama == {}, "Ollama models must be evicted before vLLM starts"
    assert world.vllm_served == "qwen3-coder-30b"
    assert job.result["server"] == "vllm"
    # WSL must be held open or the model dies on the distro's idle-shutdown.
    assert orch.primary.keepalive.ensure_calls >= 1, "vLLM load must start the WSL keepalive"


async def test_load_vllm_blocked_alias_refused(monkeypatch, tmp_path):
    world, orch, jobs = _make(monkeypatch, tmp_path)
    job = await _run_load(orch, jobs, LoadRequest(server="vllm", model="coder30-fp8"))
    assert job.state == "failed"
    assert "blocked" in job.error.lower()


async def test_load_ollama_fits_fully(monkeypatch, tmp_path):
    world, orch, jobs = _make(monkeypatch, tmp_path)
    world.vllm_served = "qwen3-coder-30b"  # vLLM is holding the GPU
    world.used_mb = 20000
    world.next_load = (20 * GiB, 20 * GiB)  # fits entirely

    job = await _run_load(orch, jobs, LoadRequest(server="ollama", model="qwen3:32b"))

    assert job.state == "succeeded", job.error
    assert world.vllm_served is None, "vLLM must be stopped to free VRAM first"
    assert "qwen3:32b" in world.ollama
    assert job.result["fully_on_gpu"] is True
    assert job.result["spilled"] is False


async def test_load_ollama_spills_when_oversized(monkeypatch, tmp_path):
    world, orch, jobs = _make(monkeypatch, tmp_path)
    world.next_load = (30 * GiB, 22 * GiB)  # 8 GiB must spill; GPU nearly full (free ~2 GiB)

    job = await _run_load(orch, jobs, LoadRequest(server="ollama", model="qwen3.6:35b-a3b"))

    assert job.state == "succeeded", job.error
    assert job.result["spilled"] is True
    assert job.result["on_cpu_bytes"] == 8 * GiB
    # GPU is essentially full, so this is an expected spill — not flagged premature
    assert not any("premature" in line.lower() for line in job.log)


async def test_load_ollama_premature_spill_flagged(monkeypatch, tmp_path):
    world, orch, jobs = _make(monkeypatch, tmp_path)
    world.next_load = (30 * GiB, 5 * GiB)  # only 5 GiB on GPU but tons free ⇒ premature

    job = await _run_load(orch, jobs, LoadRequest(server="ollama", model="weird"))

    assert job.state == "succeeded", job.error
    assert job.result["spilled"] is True
    assert any("premature" in line.lower() for line in job.log)


async def test_unload_frees_gpu(monkeypatch, tmp_path):
    world, orch, jobs = _make(monkeypatch, tmp_path)
    world.ollama = {"qwen3:32b": (20 * GiB, 20 * GiB)}
    world.used_mb = 20000

    status = await orch.unload(UnloadRequest())

    assert world.ollama == {}
    assert status.owner == "free"


async def test_status_reports_owner(monkeypatch, tmp_path):
    world, orch, jobs = _make(monkeypatch, tmp_path)
    world.vllm_served = "qwen3-coder-30b"
    world.used_mb = 20000

    status = await orch.status()
    assert status.owner == "vllm"
    assert status.vllm_up is True
    assert status.loaded.model == "qwen3-coder-30b"
