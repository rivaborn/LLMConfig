"""Companion-lane behavior: the two lanes are independent, the per-lane vLLM stop is
scoped (no cross-kill), and a configured default auto-loads.
"""
import asyncio
import time

import llmconfig.backends.vllm as vllm_mod
import llmconfig.lane as lane_mod
from llmconfig.backends.vllm import VllmBackend
from llmconfig.config import Settings
from llmconfig.gpu import GpuInfo
from llmconfig.jobs import JobManager
from llmconfig.lane_state import LaneDefaults
from llmconfig.orchestrator import Orchestrator
from llmconfig.proc import CmdResult
from llmconfig.registry import Registry
from llmconfig.schemas import LoadRequest, OllamaModel, ServedModel, UnloadRequest

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

    async def served_info(self):
        # Mirrors a real relay's /v1/models: name, root (which distinguishes
        # same-named models on different units) and the served context window.
        m = self.w.vllm
        return ServedModel(name=m, root=(f"fake-org/{m}" if m else ""),
                           context_len=32768 if m else 0)

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
        # These tests exercise the companion's vLLM half (invariant 2: the stop is
        # lane-scoped, no cross-kill), so they opt in explicitly — the real box
        # ships it off because serve-companion.sh does not exist.
        companion_vllm_enabled=True,
        companion_gpu_uuid="GPU-C",
        registry_path=tmp_path / "p.yaml",
        companion_registry_path=tmp_path / "c.yaml",
        evict_timeout_s=5,
        poll_interval_s=0.001,
    )
    jobs = JobManager()
    orch = Orchestrator(s, Registry(s.registry_path), jobs)
    orch.defaults = LaneDefaults(s, path=tmp_path / "ld.yaml")  # isolate from repo data/lane_defaults.yaml
    wp, wc = World(), World()
    for lane, w in ((orch.primary, wp), (orch.lane("companion"), wc)):
        lane.ollama = FakeOllama(w)
        lane.vllm = FakeVllm(w, lane.registry)
        lane.keepalive = FakeKeepalive()
    worlds = {"GPU-P": wp, "GPU-C": wc}

    async def fake_query_gpu(set_=None, uuid=None, **kw):
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

    job = await _run(orch, jobs, "companion", LoadRequest(server="vllm", model="smoke", lane="companion"))

    assert job.state == "succeeded", job.error
    assert wc.vllm == "smoke"
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
    orch.defaults = LaneDefaults(s, path=tmp_path / "ld.yaml")  # isolate from repo data/lane_defaults.yaml
    wp, wc = World(), World()
    for lane, w in ((orch.primary, wp), (orch.lane("companion"), wc)):
        lane.ollama = FakeOllama(w)
        lane.vllm = FakeVllm(w, lane.registry)
        lane.keepalive = FakeKeepalive()
    worlds = {"GPU-P": wp, "GPU-C": wc}

    async def fake_query_gpu(set_=None, uuid=None, **kw):
        return worlds[uuid].gpu(uuid)

    monkeypatch.setattr(lane_mod, "query_gpu", fake_query_gpu)

    started = orch.autoload_defaults()
    for j in started:
        await jobs._tasks[j.id]

    # only the companion had a default → exactly one load, onto the companion
    assert len(started) == 1
    assert "auto-me" in wc.ollama
    assert wp.ollama == {}


# --------------------------------------------------------------------------- #
# Multi-model defaults (a Spark holds several; a GPU lane exactly one)
# --------------------------------------------------------------------------- #
LEGACY_YAML = """
lanes:
  primary:
    server: ollama
    model: qwen3:32b
"""

GARBLED_YAML = """
lanes:
  primary: 'just a string'
  companion:
    models: [{}, 3]
"""


def test_defaults_read_the_legacy_scalar_shape(tmp_path):
    """An existing lane_defaults.yaml written before the list shape must keep working."""
    path = tmp_path / "ld.yaml"
    path.write_text(LEGACY_YAML, encoding="utf-8")
    d = LaneDefaults(Settings(_env_file=None), path=path)

    assert d.get("primary") == {"server": "ollama", "model": "qwen3:32b"}
    assert d.list("primary") == [{"server": "ollama", "model": "qwen3:32b"}]


def test_defaults_rewrite_the_legacy_shape_as_a_list(tmp_path):
    path = tmp_path / "ld.yaml"
    path.write_text(LEGACY_YAML, encoding="utf-8")
    LaneDefaults(Settings(_env_file=None), path=path).add("primary", "ollama", "gemma")

    reloaded = LaneDefaults(Settings(_env_file=None), path=path)
    assert [e["model"] for e in reloaded.list("primary")] == ["qwen3:32b", "gemma"]


def test_add_is_idempotent_and_set_replaces(tmp_path):
    d = LaneDefaults(Settings(_env_file=None), path=tmp_path / "ld.yaml")
    d.add("spark1", "spark", "a")
    d.add("spark1", "spark", "b")
    d.add("spark1", "spark", "a")            # re-adding must not duplicate

    assert [e["model"] for e in d.list("spark1")] == ["b", "a"]

    d.set("spark1", "spark", "c")            # "the default" still means exactly one
    assert d.list("spark1") == [{"server": "spark", "model": "c"}]


def test_remove_drops_the_unit_once_empty(tmp_path):
    d = LaneDefaults(Settings(_env_file=None), path=tmp_path / "ld.yaml")
    d.add("spark1", "spark", "a")
    d.add("spark1", "spark", "b")

    assert d.remove("spark1", "a") is True
    assert d.remove("spark1", "nope") is False
    assert d.list("spark1") == [{"server": "spark", "model": "b"}]

    d.remove("spark1", "b")
    assert d.all() == {}


def test_a_garbled_defaults_file_does_not_raise(tmp_path):
    """It is user-editable, so a bad hand-edit must degrade to "no defaults"."""
    path = tmp_path / "ld.yaml"
    path.write_text(GARBLED_YAML, encoding="utf-8")

    d = LaneDefaults(Settings(_env_file=None), path=path)

    assert d.all() == {} and d.get("primary") is None


def test_explicit_empty_tombstone_survives_reload(tmp_path):
    """`models: []` means "load nothing" — the cookbook pins a unit empty with it."""
    d = LaneDefaults(Settings(_env_file=None), path=tmp_path / "ld.yaml")
    d.set_empty("spark3")

    again = LaneDefaults(Settings(_env_file=None), path=tmp_path / "ld.yaml")
    assert again.entries_or_none("spark3") == [], "tombstone kept across reload"
    assert again.entries_or_none("spark4") is None, "unset stays unset"


def test_tombstone_suppresses_the_env_seed_in_defaults_for(tmp_path, monkeypatch):
    """An explicitly-empty unit must NOT fall back to the .env default_model."""
    import llmconfig.orchestrator as orch_mod
    from llmconfig.jobs import JobManager
    from llmconfig.orchestrator import Orchestrator
    from llmconfig.registry import Registry

    s = Settings(_env_file=None, gpu_uuid="GPU-x", registry_path=tmp_path / "reg.yaml",
                 companion_enabled=True, companion_gpu_uuid="GPU-y",
                 companion_registry_path=tmp_path / "comp.yaml",
                 companion_default_server="ollama", companion_default_model="qwen2.5:1.5b")
    orch = Orchestrator(s, Registry(s.registry_path), JobManager())
    orch.defaults = LaneDefaults(s, path=tmp_path / "ld.yaml")

    assert orch.defaults_for("companion") == [
        {"server": "ollama", "model": "qwen2.5:1.5b"}], "unset -> .env seed applies"

    orch.defaults.set_empty("companion")
    assert orch.defaults_for("companion") == [], "tombstone beats the seed"


def test_garbled_models_list_is_not_a_tombstone(tmp_path):
    """A non-empty list of garbage means the author TRIED to configure something —
    treat as unset (seed applies), never as deliberate emptiness."""
    path = tmp_path / "ld.yaml"
    path.write_text("lanes:\n  primary:\n    models: [{}, 3]\n", encoding="utf-8")
    d = LaneDefaults(Settings(_env_file=None), path=path)
    assert d.entries_or_none("primary") is None


# --------------------------------------------------------------------------- #
# Ollama-only lane (COMPANION_VLLM_ENABLED=false)
# --------------------------------------------------------------------------- #
def _ollama_only_settings(tmp_path):
    return Settings(_env_file=None, gpu_uuid="GPU-x",
                    registry_path=tmp_path / "reg.yaml",
                    companion_enabled=True, companion_gpu_uuid="GPU-y",
                    companion_registry_path=tmp_path / "comp.yaml")


def test_companion_vllm_is_off_by_default_and_primary_is_not(tmp_path):
    """serve-companion.sh has never existed, so the honest default is off."""
    lanes = {c.id: c for c in _ollama_only_settings(tmp_path).lanes()}
    assert lanes["companion"].vllm_enabled is False
    assert lanes["primary"].vllm_enabled is True


async def test_vllm_load_on_an_ollama_only_lane_refuses_without_evicting(tmp_path, monkeypatch):
    """The point of the flag: the OLD order held WSL open, unloaded the lane's
    working Ollama model and drained VRAM before discovering the systemd unit was
    missing — a request for a model that can never run killed the running one."""
    import llmconfig.lane as lane_mod
    from llmconfig.jobs import JobManager
    from llmconfig.orchestrator import Orchestrator
    from llmconfig.registry import Registry
    from llmconfig.schemas import LoadRequest

    async def fake_query_gpu(set_=None, uuid=None, **kw):
        from llmconfig.gpu import GpuInfo
        return GpuInfo(found=True, uuid=uuid or "GPU-y", total_mb=8192,
                       used_mb=300, free_mb=7892)

    monkeypatch.setattr(lane_mod, "query_gpu", fake_query_gpu)
    s = _ollama_only_settings(tmp_path)
    orch = Orchestrator(s, Registry(s.registry_path), JobManager())
    lane = orch.lanes["companion"]

    evicted: list[str] = []
    stopped: list[str] = []

    async def no_unload():
        evicted.append("ollama")
        return []

    async def no_stop():
        stopped.append("vllm")

    monkeypatch.setattr(lane.ollama, "unload_all", no_unload)
    monkeypatch.setattr(lane.vllm, "stop", no_stop)
    monkeypatch.setattr(lane.keepalive, "ensure", lambda: True)

    job = lane.load(LoadRequest(server="vllm", model="smoke", lane="companion"))
    deadline = time.monotonic() + 10
    while job.state in ("pending", "running") and time.monotonic() < deadline:
        await asyncio.sleep(0.01)

    assert job.state == "failed"
    assert "not available on the companion lane" in (job.error or "")
    assert evicted == [], "must refuse BEFORE touching the Ollama resident"
    assert stopped == [], "and before any WSL round-trip"


async def test_ollama_only_lane_never_probes_its_dead_relay(tmp_path, monkeypatch):
    """A down relay blackholes the SYN (invariant 5), so probing one that will
    never exist costs the full timeout on every /api/status."""
    import llmconfig.lane as lane_mod
    from llmconfig.jobs import JobManager
    from llmconfig.orchestrator import Orchestrator
    from llmconfig.registry import Registry

    async def fake_query_gpu(set_=None, uuid=None, **kw):
        from llmconfig.gpu import GpuInfo
        return GpuInfo(found=True, uuid=uuid or "GPU-y", total_mb=8192,
                       used_mb=300, free_mb=7892)

    monkeypatch.setattr(lane_mod, "query_gpu", fake_query_gpu)
    s = _ollama_only_settings(tmp_path)
    orch = Orchestrator(s, Registry(s.registry_path), JobManager())
    lane = orch.lanes["companion"]

    probes: list[str] = []

    async def counted_served_info():
        probes.append("probe")
        raise AssertionError("the relay must not be probed on an Ollama-only lane")

    async def no_ollama():
        return []

    async def ollama_up():
        return True

    monkeypatch.setattr(lane.vllm, "served_info", counted_served_info)
    monkeypatch.setattr(lane.ollama, "loaded", no_ollama)
    monkeypatch.setattr(lane.ollama, "up", ollama_up)

    st = await lane.status()
    assert probes == []
    assert st.owner == "free" and st.vllm_up is False


async def test_ollama_only_lane_load_and_unload_never_probe_the_relay(tmp_path, monkeypatch):
    """`_load_ollama`'s fast path and `_occupied_by` both asked vLLM what it was
    serving — on an Ollama-only lane that's a blackholed SYN (invariant 5), a
    ~1 s tax on EVERY companion load and targeted unload, for a relay that can
    never exist."""
    import llmconfig.lane as lane_mod
    from llmconfig.gpu import GpuInfo
    from llmconfig.jobs import JobManager
    from llmconfig.orchestrator import Orchestrator
    from llmconfig.registry import Registry

    async def fake_query_gpu(set_=None, uuid=None, **kw):
        return GpuInfo(found=True, uuid=uuid or "GPU-y", total_mb=8192,
                       used_mb=3000, free_mb=5192)

    monkeypatch.setattr(lane_mod, "query_gpu", fake_query_gpu)
    s = _ollama_only_settings(tmp_path)
    jobs = JobManager()
    orch = Orchestrator(s, Registry(s.registry_path), jobs)
    lane = orch.lanes["companion"]

    async def boom(*a, **k):
        raise AssertionError("the relay must not be probed on an Ollama-only lane")

    monkeypatch.setattr(lane.vllm, "served", boom)
    monkeypatch.setattr(lane.vllm, "served_info", boom)

    resident = OllamaModel(name="m:1", size_bytes=GiB, loaded=True,
                           size_vram_bytes=GiB)

    async def loaded():
        return [resident]

    async def loaded_names():
        return ["m:1"]

    async def ollama_up():
        return True

    monkeypatch.setattr(lane.ollama, "loaded", loaded)
    monkeypatch.setattr(lane.ollama, "loaded_names", loaded_names)
    monkeypatch.setattr(lane.ollama, "up", ollama_up)

    # Fast path (already the sole resident) — used to pay the probe up front.
    job = lane.load(LoadRequest(server="ollama", model="m:1", lane="companion"))
    await jobs._tasks[job.id]
    assert job.state == "succeeded", job.error

    # Targeted unload of a non-resident — _occupied_by used to probe too.
    st = await lane.unload(UnloadRequest(server=None, lane="companion", model="ghost:1"))
    assert st.owner == "ollama"
