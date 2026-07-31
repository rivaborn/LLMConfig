"""Orchestrator logic with in-memory fakes for the two backends + GPU.

Exercises the core guarantees: eviction before load, the VRAM-free gate, and the
pack-then-spill verification — without touching wsl.exe / nvidia-smi / Ollama.
"""
import llmconfig.lane as lane_mod
import llmconfig.orchestrator as orch_mod
from llmconfig.config import Settings
from llmconfig.gpu import GpuInfo
from llmconfig.jobs import JobManager
from llmconfig.orchestrator import Orchestrator
from llmconfig.proc import CmdResult
from llmconfig.registry import Registry
from llmconfig.schemas import LoadRequest, OllamaModel, ServedModel, UnloadRequest, VllmAliasEntry

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

    async def served_info(self):
        # Mirrors a real relay's /v1/models: name, root (which distinguishes
        # same-named models on different units) and the served context window.
        m = self.w.vllm_served
        return ServedModel(name=m, root=(f"fake-org/{m}" if m else ""),
                           context_len=32768 if m else 0)

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
    settings = Settings(_env_file=None, gpu_uuid="GPU-x", evict_timeout_s=5, poll_interval_s=0.01)
    jobs = JobManager()
    orch = Orchestrator(settings, registry, jobs)
    # Arbitration lives on the lane; swap in the fakes there.
    lane = orch.primary
    lane.ollama = FakeOllama(world)
    lane.vllm = FakeVllm(world, registry)
    lane.keepalive = FakeKeepalive()

    async def fake_query_gpu(s=None, uuid=None, **kw):
        return world.gpu()

    async def fake_query_all(s=None):
        return {"GPU-x": world.gpu()}

    monkeypatch.setattr(lane_mod, "query_gpu", fake_query_gpu)
    monkeypatch.setattr(orch_mod, "query_all_gpus", fake_query_all)
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
    # A blocked alias must be refused (synthetic entry — decoupled from the catalog).
    orch.primary.registry.upsert(VllmAliasEntry(alias="zzz-blocked", served_name="zzz",
                                                status="blocked", managed_by="serve.sh"))
    job = await _run_load(orch, jobs, LoadRequest(server="vllm", model="zzz-blocked"))
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


# --------------------------------------------------------------------------- #
# Boot autoload ordering (needs_empty_node first — see schemas.boot_order_key)
# --------------------------------------------------------------------------- #
async def test_autoload_orders_needs_empty_node_first(monkeypatch, tmp_path):
    """A reranker-shaped default (needs_empty_node) must be DISPATCHED before its
    co-residents even when lane_defaults.yaml lists it last — file order is only
    a tiebreak (live incident 2026-07-28: fastsafetensors beside a resident)."""
    from types import SimpleNamespace

    from llmconfig.lane_state import LaneDefaults

    s = Settings(_env_file=None, gpu_uuid="GPU-x", registry_path=tmp_path / "p.yaml",
                 spark_enabled=True, spark_nodes="spark9=10.0.0.9")
    jobs = JobManager()
    orch = Orchestrator(s, Registry(s.registry_path), jobs)
    orch.defaults = LaneDefaults(s, path=tmp_path / "ld.yaml")

    entries = {
        "emb": SimpleNamespace(needs_empty_node=False, mem_fraction=0.33),
        "rr":  SimpleNamespace(needs_empty_node=True,  mem_fraction=0.35),
        "big": SimpleNamespace(needs_empty_node=False, mem_fraction=0.80),
    }
    unit = orch.units["spark9"]
    unit.registry = SimpleNamespace(get=lambda a: entries.get(a))

    dispatched: list[str] = []

    def fake_load(req):
        dispatched.append(req.model)
        job = jobs.create(kind=f"load:{req.model}")

        async def body(j):
            return {}

        return jobs.start(job, body)

    unit.load = fake_load

    # Deliberately wrong file order: reranker LAST, biggest budget in the middle.
    orch.defaults.set("spark9", "spark", "emb")
    orch.defaults.add("spark9", "spark", "big")
    orch.defaults.add("spark9", "spark", "rr")

    started = orch.autoload_defaults()
    for j in started:                       # completed fakes are already reaped
        t = jobs._tasks.get(j.id)
        if t is not None:
            await t

    assert dispatched == ["rr", "big", "emb"], \
        "needs_empty_node first, then biggest budget, regardless of file order"


async def test_autoload_lane_defaults_keep_file_order(monkeypatch, tmp_path):
    """GPU-lane entries have no ordering fields — boot_order_key degrades to a
    constant and the sort must keep the user's file order (Ollama tags resolve
    to no registry entry at all)."""
    from llmconfig.lane_state import LaneDefaults

    s = Settings(_env_file=None, gpu_uuid="GPU-x", registry_path=tmp_path / "p.yaml")
    jobs = JobManager()
    orch = Orchestrator(s, Registry(s.registry_path), jobs)
    orch.defaults = LaneDefaults(s, path=tmp_path / "ld.yaml")

    dispatched: list[str] = []

    def fake_load(req):
        dispatched.append(req.model)
        job = jobs.create(kind=f"load:{req.model}")

        async def body(j):
            return {}

        return jobs.start(job, body)

    orch.primary.load = fake_load
    orch.defaults.set("primary", "ollama", "b-tag:1")
    orch.defaults.add("primary", "ollama", "a-tag:1")

    started = orch.autoload_defaults()
    for j in started:                       # completed fakes are already reaped
        t = jobs._tasks.get(j.id)
        if t is not None:
            await t

    assert dispatched == ["b-tag:1", "a-tag:1"], "no fields -> stable file order"
