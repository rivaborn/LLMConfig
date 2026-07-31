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
from llmconfig.leases import LeaseManager
from llmconfig.monitor import Monitor
from llmconfig.orchestrator import Orchestrator
from llmconfig.proc import CmdResult
from llmconfig.registry import Registry
from llmconfig.schemas import GpuOut, LaneStatus, LeaseClaimRequest, LoadedModel, OllamaModel, ServedModel

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
        # companion_vllm_enabled: these tests drive the companion's vLLM half
        # (keepalive accounting across lanes); the real box ships it off because
        # serve-companion.sh does not exist.
        **({"companion_enabled": True, "companion_gpu_uuid": "GPU-y",
            "companion_vllm_enabled": True,
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

    async def fake_query_gpu(s=None, uuid=None, **kw):
        for w in worlds.values():
            if w.uuid == uuid:
                return w.gpu()
        return next(iter(worlds.values())).gpu()

    async def fake_query_all(s=None):
        return {w.uuid: w.gpu() for w in worlds.values()}

    monkeypatch.setattr(lane_mod, "query_gpu", fake_query_gpu)
    monkeypatch.setattr(orch_mod, "query_all_gpus", fake_query_all)

    leases = LeaseManager(settings, orch)
    reaper = IdleReaper(settings, orch, mon if mon is not None else FakeMonitor(), leases)
    reaper.leases_mgr = leases  # convenience handle for the lease-aware tests below
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


# --------------------------------------------------------------------------- #
# Leases vs the reaper
# --------------------------------------------------------------------------- #
def _claim(reaper, holder="alice", unit="primary", **kw):
    return reaper.leases_mgr.claim(LeaseClaimRequest(unit=unit, holder=holder, **kw))[0]


async def test_leased_lane_not_reaped(monkeypatch, tmp_path):
    worlds, orch, reaper, _ = _make(monkeypatch, tmp_path)
    lane = orch.primary
    _load_ollama(worlds["primary"], lane)
    lane.last_activity = time.time() - IDLE
    _claim(reaper, preemptible=False)

    await reaper._tick()

    assert "qwen3:32b" in worlds["primary"].ollama, "a claimed lane must not be reaped"


async def test_preemptible_lease_also_blocks_reaping(monkeypatch, tmp_path):
    """The reaper is a power optimisation, not a competing caller — 'preemptible'
    means another *caller* may take the unit, not that the reaper may."""
    worlds, orch, reaper, _ = _make(monkeypatch, tmp_path)
    lane = orch.primary
    _load_ollama(worlds["primary"], lane)
    lane.last_activity = time.time() - IDLE
    _claim(reaper, preemptible=True)

    await reaper._tick()

    assert "qwen3:32b" in worlds["primary"].ollama


async def test_expired_lease_stops_blocking_the_reaper(monkeypatch, tmp_path):
    worlds, orch, reaper, _ = _make(monkeypatch, tmp_path)
    lane = orch.primary
    _load_ollama(worlds["primary"], lane)
    lane.last_activity = time.time() - IDLE
    lease = _claim(reaper)
    lease._deadline = time.monotonic() - 1  # lapsed

    await reaper._tick()

    assert worlds["primary"].ollama == {}, "a lapsed lease must not keep blocking"


async def test_lease_claimed_during_status_probe_blocks_the_reap(monkeypatch, tmp_path):
    """The fused post-await re-check. A claim can land during `await lane.status()`;
    without re-checking, the new holder's unit would be unloaded underneath it."""
    worlds, orch, reaper, _ = _make(monkeypatch, tmp_path)
    lane = orch.primary
    _load_ollama(worlds["primary"], lane)
    lane.last_activity = time.time() - IDLE

    real_status = lane.status

    async def status_then_claim(*a, **kw):
        st = await real_status(*a, **kw)
        _claim(reaper, holder="late-arrival")  # lands mid-probe
        return st

    monkeypatch.setattr(lane, "status", status_then_claim)

    await reaper._tick()

    assert "qwen3:32b" in worlds["primary"].ollama, "reaped a lane claimed mid-probe"


async def test_lease_blocking_still_folds_the_util_signal(monkeypatch, tmp_path):
    """The monitor fold-in sits ABOVE the lease guard, so a leased lane's idle_s
    keeps tracking reality and it's immediately reapable once the lease lapses."""
    mon = FakeMonitor()
    worlds, orch, reaper, _ = _make(monkeypatch, tmp_path, mon=mon)
    lane = orch.primary
    _load_ollama(worlds["primary"], lane)
    lane.last_activity = time.time() - IDLE
    spike = time.time() - 5
    mon.spikes["GPU-x"] = spike
    _claim(reaper)

    await reaper._tick()

    assert lane.last_activity >= spike, "util signal must fold in even for a leased lane"


async def test_lease_blocks_idle_unload_kill_switch(monkeypatch, tmp_path):
    worlds, orch, reaper, _ = _make(monkeypatch, tmp_path, lease_blocks_idle_unload=False)
    lane = orch.primary
    _load_ollama(worlds["primary"], lane)
    lane.last_activity = time.time() - IDLE
    _claim(reaper, preemptible=False)

    await reaper._tick()

    assert worlds["primary"].ollama == {}, "kill switch must restore the old behaviour"


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


async def test_primary_exempt_when_opted_out(monkeypatch, tmp_path):
    worlds, orch, reaper, _ = _make(monkeypatch, tmp_path,
                                    primary_idle_unload_enabled=False)
    _load_ollama(worlds["primary"], orch.primary)
    orch.primary.last_activity = time.time() - IDLE

    await reaper._tick()

    assert "qwen3:32b" in worlds["primary"].ollama, \
        "PRIMARY_IDLE_UNLOAD_ENABLED=false pins the 3090's resident model"


async def test_primary_still_reaped_by_default(monkeypatch, tmp_path):
    # The new knob defaults True — adding it must not change stock behavior.
    worlds, orch, reaper, _ = _make(monkeypatch, tmp_path)
    _load_ollama(worlds["primary"], orch.primary)
    orch.primary.last_activity = time.time() - IDLE

    await reaper._tick()

    assert worlds["primary"].ollama == {}


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


# --------------------------------------------------------------------------- #
# The pin override (UI checkbox) — beats the static config in BOTH directions
# --------------------------------------------------------------------------- #
async def test_pin_true_shields_a_reapable_lane(monkeypatch, tmp_path):
    from llmconfig.lane_state import LanePins
    worlds, orch, reaper, _ = _make(monkeypatch, tmp_path)   # primary reapable by cfg
    orch.pins = LanePins(orch.s, path=tmp_path / "pins.yaml")
    orch.pins.set("primary", True)
    _load_ollama(worlds["primary"], orch.primary)
    orch.primary.last_activity = time.time() - IDLE

    await reaper._tick()

    assert "qwen3:32b" in worlds["primary"].ollama, \
        "pinned=True overrides the cfg's reapable default"


async def test_pin_false_makes_an_exempt_lane_reapable(monkeypatch, tmp_path):
    from llmconfig.lane_state import LanePins
    worlds, orch, reaper, _ = _make(monkeypatch, tmp_path,
                                    primary_idle_unload_enabled=False)  # .env pins it
    orch.pins = LanePins(orch.s, path=tmp_path / "pins.yaml")
    orch.pins.set("primary", False)
    _load_ollama(worlds["primary"], orch.primary)
    orch.primary.last_activity = time.time() - IDLE

    await reaper._tick()

    assert worlds["primary"].ollama == {}, \
        "pinned=False overrides the .env exemption — the checkbox is the authority"


async def test_pin_absent_keeps_configured_behavior(monkeypatch, tmp_path):
    from llmconfig.lane_state import LanePins
    worlds, orch, reaper, _ = _make(monkeypatch, tmp_path,
                                    primary_idle_unload_enabled=False)
    orch.pins = LanePins(orch.s, path=tmp_path / "pins.yaml")   # no entry
    _load_ollama(worlds["primary"], orch.primary)
    orch.primary.last_activity = time.time() - IDLE

    await reaper._tick()

    assert "qwen3:32b" in worlds["primary"].ollama, "no override -> cfg applies"


def test_lane_pins_persist_and_clear(tmp_path):
    from llmconfig.config import Settings
    from llmconfig.lane_state import LanePins
    s = Settings(_env_file=None)
    p = LanePins(s, path=tmp_path / "pins.yaml")
    assert p.get("primary") is None
    p.set("primary", False)
    assert LanePins(s, path=tmp_path / "pins.yaml").get("primary") is False
    p.set("primary", None)                                     # clear
    assert LanePins(s, path=tmp_path / "pins.yaml").get("primary") is None
    # corrupt file must not raise — no overrides is the safe read
    (tmp_path / "pins.yaml").write_text("lanes: [broken", encoding="utf-8")
    assert LanePins(s, path=tmp_path / "pins.yaml").get("primary") is None


async def test_pin_api_round_trip(monkeypatch, tmp_path):
    """PUT /api/lanes/primary/pin drives the override; /api/status reports the
    EFFECTIVE pin; Sparks are refused (fleet policy, not per-card)."""
    import httpx
    from httpx import ASGITransport

    import llmconfig.main as main_mod
    from llmconfig.lane_state import LanePins

    s = Settings(_env_file=None, monitor_enabled=False, idle_unload_enabled=False,
                 registry_path=tmp_path / "reg.yaml", gpu_uuid="GPU-x",
                 spark_enabled=True, spark_nodes="spark9=10.0.0.9")
    monkeypatch.setattr(main_mod, "get_settings", lambda: s)

    async def fake_query_gpu(set_=None, uuid=None, **kw):
        return GpuInfo(found=True, uuid=uuid or "GPU-x", total_mb=24576,
                       used_mb=400, free_mb=24176)

    monkeypatch.setattr(lane_mod, "query_gpu", fake_query_gpu)
    app = main_mod.create_app()
    app.state.orch.pins = LanePins(s, path=tmp_path / "pins.yaml")

    # Keep /api/status off the network: the spark unit's status probes SSH.
    async def fake_spark_status(gpu=None):
        return LaneStatus(id="spark9", name="spark9", owner="free",
                          ollama_up=False, vllm_up=False, gpu=GpuOut(found=False))

    monkeypatch.setattr(app.state.orch.units["spark9"], "status", fake_spark_status)

    async with httpx.AsyncClient(transport=ASGITransport(app=app),
                                 base_url="http://t") as c:
        # Default: primary reapable by cfg -> effective pinned False
        st = await c.get("/api/status")
        lane = [l for l in st.json()["lanes"] if l["id"] == "primary"][0]
        assert lane["pinned"] is False
        spark = [l for l in st.json()["lanes"] if l["id"] == "spark9"][0]
        assert spark["pinned"] is None, "pinning is a GPU-lane concept"

        r = await c.put("/api/lanes/primary/pin", json={"pinned": True})
        assert r.status_code == 200 and r.json()["pinned"] is True

        st = await c.get("/api/status")
        lane = [l for l in st.json()["lanes"] if l["id"] == "primary"][0]
        assert lane["pinned"] is True

        r = await c.put("/api/lanes/primary/pin", json={"pinned": None})
        assert r.json()["override"] is None and r.json()["pinned"] is False

        r = await c.put("/api/lanes/spark9/pin", json={"pinned": True})
        assert r.status_code == 400

        r = await c.put("/api/lanes/primary/pin", json={})
        assert r.status_code == 400
