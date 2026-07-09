"""Idle auto-unload policy (`llmconfig/idle.py`): the hybrid activity signal, the
reap-through-`Lane.unload` path, and the WSL-keepalive release — with the same
in-memory World/fakes as test_orchestrator (no wsl.exe / nvidia-smi / real servers).
"""
import time

import llmconfig.lane as lane_mod
import llmconfig.orchestrator as orch_mod
from llmconfig.config import Settings
from llmconfig.gpu import GpuInfo
from llmconfig.idle import IdleReaper, classify_usage
from llmconfig.jobs import JobManager
from llmconfig.monitor import Monitor
from llmconfig.orchestrator import Orchestrator
from llmconfig.proc import CmdResult
from llmconfig.registry import Registry
from llmconfig.schemas import GpuOut, LaneStatus, LoadedModel, OllamaModel

GiB = 1024 ** 3
BASE_MB = 400
IDLE = 16 * 60  # comfortably past the 15-min default timeout


class World:
    def __init__(self, uuid="GPU-x"):
        self.uuid = uuid
        self.vllm_served = None
        self.ollama: dict[str, tuple[int, int]] = {}
        self.used_mb = BASE_MB

    def gpu(self) -> GpuInfo:
        return GpuInfo(found=True, uuid=self.uuid, total_mb=24576,
                       used_mb=self.used_mb, free_mb=24576 - self.used_mb)


class FakeOllama:
    def __init__(self, w: World):
        self.w = w

    async def up(self):
        return True

    async def ensure_running(self, wait_s=20.0):
        return True

    async def loaded(self):
        return [OllamaModel(name=n, size_bytes=s, loaded=True, size_vram_bytes=v)
                for n, (s, v) in self.w.ollama.items()]

    async def loaded_names(self):
        return list(self.w.ollama)

    async def unload_all(self):
        names = list(self.w.ollama)
        self.w.ollama.clear()
        if names:
            self.w.used_mb = BASE_MB
        return names


class FakeVllm:
    def __init__(self, w: World):
        self.w = w

    async def served(self):
        return self.w.vllm_served

    async def up(self):
        return self.w.vllm_served is not None

    async def stop(self):
        self.w.vllm_served = None
        self.w.used_mb = BASE_MB


class FakeKeepalive:
    def __init__(self):
        self.ensure_calls = 0
        self.stopped = False

    def ensure(self):
        self.ensure_calls += 1
        self.stopped = False
        return True

    def alive(self):
        return self.ensure_calls > 0 and not self.stopped

    def stop(self):
        self.stopped = True


class FakeMonitor:
    """`last_util_activity` stand-in: returns the configured per-UUID spike ts."""

    def __init__(self):
        self.spikes: dict[str, float] = {}

    def last_util_activity(self, uuid, threshold, since):
        ts = self.spikes.get(uuid)
        return ts if ts is not None and ts > since else None


def _make(monkeypatch, tmp_path, *, two_lanes=False, mon=None, **overrides):
    settings = Settings(
        _env_file=None, gpu_uuid="GPU-x", registry_path=tmp_path / "reg.yaml",
        evict_timeout_s=5, poll_interval_s=0.01,
        **({"companion_enabled": True, "companion_gpu_uuid": "GPU-y",
            "companion_registry_path": tmp_path / "comp.yaml"} if two_lanes else {}),
        **overrides,
    )
    jobs = JobManager()
    orch = Orchestrator(settings, Registry(settings.registry_path), jobs)
    keepalive = FakeKeepalive()
    orch.keepalive = keepalive  # the reaper releases the ORCHESTRATOR's keepalive
    worlds: dict[str, World] = {}
    for lane in orch.lanes.values():
        w = World(uuid=lane.cfg.gpu_uuid)
        worlds[lane.cfg.id] = w
        lane.ollama = FakeOllama(w)
        lane.vllm = FakeVllm(w)
        lane.keepalive = keepalive

    async def fake_query_gpu(s=None, uuid=None):
        for w in worlds.values():
            if w.uuid == uuid:
                return w.gpu()
        return next(iter(worlds.values())).gpu()

    async def fake_query_all(s=None):
        return {w.uuid: w.gpu() for w in worlds.values()}

    monkeypatch.setattr(lane_mod, "query_gpu", fake_query_gpu)
    monkeypatch.setattr(orch_mod, "query_all_gpus", fake_query_all)

    reaper = IdleReaper(settings, orch, mon if mon is not None else FakeMonitor())
    return worlds, orch, reaper, keepalive


def _load_ollama(world: World, lane, model="qwen3:32b"):
    world.ollama = {model: (20 * GiB, 20 * GiB)}
    world.used_mb = 20000


def _load_vllm(world: World, keepalive: FakeKeepalive, served="qwen3-coder-30b"):
    world.vllm_served = served
    world.used_mb = 20000
    keepalive.ensure()  # a real vLLM load holds the WSL distro open


async def test_reaps_ollama_after_timeout(monkeypatch, tmp_path):
    worlds, orch, reaper, ka = _make(monkeypatch, tmp_path)
    lane = orch.primary
    _load_ollama(worlds["primary"], lane)
    lane.last_activity = time.time() - IDLE

    await reaper._tick()

    assert worlds["primary"].ollama == {}, "idle Ollama model must be unloaded"
    assert (await lane.status()).owner == "free"
    assert ka.stopped is False, "no vLLM was reaped — keepalive untouched"


async def test_fresh_lane_not_reaped(monkeypatch, tmp_path):
    worlds, orch, reaper, _ = _make(monkeypatch, tmp_path)
    _load_ollama(worlds["primary"], orch.primary)
    orch.primary.touch()  # recent activity (also the startup-grace semantics)

    await reaper._tick()

    assert "qwen3:32b" in worlds["primary"].ollama


async def test_util_spike_resets_timer(monkeypatch, tmp_path):
    mon = FakeMonitor()
    worlds, orch, reaper, _ = _make(monkeypatch, tmp_path, mon=mon)
    lane = orch.primary
    _load_ollama(worlds["primary"], lane)
    lane.last_activity = time.time() - IDLE
    mon.spikes["GPU-x"] = time.time() - 30  # direct-to-backend client 30 s ago

    await reaper._tick()

    assert "qwen3:32b" in worlds["primary"].ollama, "recent util spike must block the reap"
    assert time.time() - lane.last_activity < 60, "the spike ts must advance the timer"


async def test_touch_never_moves_backwards(monkeypatch, tmp_path):
    _, orch, _, _ = _make(monkeypatch, tmp_path)
    lane = orch.primary
    lane.touch()
    now = lane.last_activity
    lane.touch(now - 3600)  # a stale Monitor sample
    assert lane.last_activity == now


async def test_swap_in_progress_skips(monkeypatch, tmp_path):
    worlds, orch, reaper, _ = _make(monkeypatch, tmp_path)
    lane = orch.primary
    _load_ollama(worlds["primary"], lane)
    lane.last_activity = time.time() - IDLE

    async with lane._lock:  # a swap is mid-flight
        await reaper._tick()

    assert "qwen3:32b" in worlds["primary"].ollama


async def test_disabled_flag_noops(monkeypatch, tmp_path):
    _, _, reaper, _ = _make(monkeypatch, tmp_path, idle_unload_enabled=False)
    reaper.start()
    assert reaper._task is None


async def test_free_lane_not_reaped(monkeypatch, tmp_path):
    _, orch, reaper, _ = _make(monkeypatch, tmp_path)
    lane = orch.primary
    lane.last_activity = time.time() - IDLE
    calls = []

    async def spy_unload(req):
        calls.append(req)

    monkeypatch.setattr(lane, "unload", spy_unload)
    await reaper._tick()
    assert calls == [], "nothing loaded — unload must not be called"


async def test_keepalive_released_when_last_vllm_reaped(monkeypatch, tmp_path):
    worlds, orch, reaper, ka = _make(monkeypatch, tmp_path)
    lane = orch.primary
    _load_vllm(worlds["primary"], ka)
    lane.last_activity = time.time() - IDLE

    await reaper._tick()

    assert worlds["primary"].vllm_served is None
    assert ka.stopped is True, "no lane serves vLLM anymore — WSL hold must be released"


async def test_keepalive_kept_while_other_lane_serves_vllm(monkeypatch, tmp_path):
    worlds, orch, reaper, ka = _make(monkeypatch, tmp_path, two_lanes=True)
    _load_vllm(worlds["primary"], ka)
    orch.primary.last_activity = time.time() - IDLE
    _load_vllm(worlds["companion"], ka, served="qwen3-4b")
    orch.lane("companion").touch()  # companion is active → not reaped

    await reaper._tick()

    assert worlds["primary"].vllm_served is None, "idle primary vLLM must be reaped"
    assert worlds["companion"].vllm_served == "qwen3-4b"
    assert ka.stopped is False, "companion still serves vLLM — keep the WSL hold"


async def test_no_monitor_still_reaps(monkeypatch, tmp_path):
    # A never-started real Monitor (no samples): the util signal goes quiet and the
    # reap runs on timestamps alone — the off-box / MONITOR_ENABLED=false case.
    worlds, orch, reaper, _ = _make(monkeypatch, tmp_path)
    reaper.monitor = Monitor(reaper.s, orch)
    lane = orch.primary
    _load_ollama(worlds["primary"], lane)
    lane.last_activity = time.time() - IDLE

    await reaper._tick()

    assert worlds["primary"].ollama == {}


async def test_companion_exempt_from_reaping_by_default(monkeypatch, tmp_path):
    worlds, orch, reaper, _ = _make(monkeypatch, tmp_path, two_lanes=True)
    _load_ollama(worlds["primary"], orch.primary)
    orch.primary.last_activity = time.time() - IDLE
    _load_ollama(worlds["companion"], orch.lane("companion"), model="qwen2.5:1.5b")
    orch.lane("companion").last_activity = time.time() - IDLE

    await reaper._tick()

    assert worlds["primary"].ollama == {}, "primary still participates"
    assert "qwen2.5:1.5b" in worlds["companion"].ollama, \
        "companion is exempt unless COMPANION_IDLE_UNLOAD_ENABLED is set"


async def test_companion_reaped_when_opted_in(monkeypatch, tmp_path):
    worlds, orch, reaper, _ = _make(monkeypatch, tmp_path, two_lanes=True,
                                    companion_idle_unload_enabled=True)
    _load_ollama(worlds["companion"], orch.lane("companion"), model="qwen2.5:1.5b")
    orch.lane("companion").last_activity = time.time() - IDLE

    await reaper._tick()

    assert worlds["companion"].ollama == {}


async def test_lane_failure_does_not_kill_tick(monkeypatch, tmp_path):
    worlds, orch, reaper, _ = _make(monkeypatch, tmp_path, two_lanes=True,
                                    companion_idle_unload_enabled=True)
    _load_ollama(worlds["companion"], orch.lane("companion"))
    orch.lane("companion").last_activity = time.time() - IDLE
    orch.primary.last_activity = time.time() - IDLE

    async def boom(gpu=None):
        raise RuntimeError("nvidia-smi exploded")

    monkeypatch.setattr(orch.primary, "status", boom)
    await reaper._tick()  # must not raise

    assert worlds["companion"].ollama == {}, "the healthy lane must still be reaped"


async def test_lane_status_reports_idle_s(monkeypatch, tmp_path):
    _, orch, _, _ = _make(monkeypatch, tmp_path)
    orch.primary.last_activity = time.time() - 120
    status = await orch.status()
    idle_s = status.lanes[0].idle_s
    assert idle_s is not None and 119 <= idle_s <= 130


# --------------------------------------------------------------------------- #
# classify_usage — the free / idle / active tri-state behind GET /api/usage
# --------------------------------------------------------------------------- #
def _lane_status(owner, idle_s=None, swap=False, model=None):
    return LaneStatus(
        id="primary", name="RTX 3090", owner=owner, ollama_up=True,
        vllm_up=owner == "vllm", gpu=GpuOut(found=False),
        loaded=LoadedModel(server=owner, model=model) if model else None,
        swap_in_progress=swap, idle_s=idle_s,
    )


def test_classify_free_when_nothing_loaded():
    s = Settings(_env_file=None)
    assert classify_usage(_lane_status("free"), None, s) == "free"
    assert classify_usage(_lane_status("unknown"), None, s) == "free"


def test_classify_idle_when_loaded_past_window():
    s = Settings(_env_file=None)
    st = _lane_status("vllm", idle_s=300.0, model="gemma-4-26b")
    assert classify_usage(st, None, s) == "idle"
    assert classify_usage(st, 0.0, s) == "idle"  # current util below threshold


def test_classify_active_within_window():
    s = Settings(_env_file=None)
    st = _lane_status("ollama", idle_s=12.0, model="qwen3:32b")
    assert classify_usage(st, None, s) == "active"
    # boundary: exactly the window is still active
    st.idle_s = s.usage_active_window_s
    assert classify_usage(st, None, s) == "active"


def test_classify_active_on_current_util_despite_stale_idle_s():
    # A direct-to-backend client is generating right now, but its util hasn't been
    # folded into idle_s yet (that happens on the reaper's next tick).
    s = Settings(_env_file=None)
    st = _lane_status("vllm", idle_s=900.0, model="gemma-4-26b")
    assert classify_usage(st, 91.7, s) == "active"


def test_classify_active_during_swap():
    s = Settings(_env_file=None)
    assert classify_usage(_lane_status("free", swap=True), None, s) == "active"
