"""Boot autoload only loads into units that are free or holding an IDLE model.

The defaults are a convenience ("what I usually run here"), not a claim, and this
app restarts far more often than the fleet goes idle. Firing them unconditionally
made every restart an eviction event for whatever was actually working: a restart
during the 2026-08 LitRank embedding backfill would have loaded `vl32` over the
3090 and `qwen35-122b` / the VL reranker pair over spark3 and spark4, taking 3 of
its 5 lanes mid-run.

Pure in-memory: a fake unit stands in for Lane/SparkUnit, so nothing here touches
nvidia-smi, WSL, HTTP, or a real event loop.
"""
import time
from types import SimpleNamespace

import pytest

from llmconfig.config import Settings
from llmconfig.schemas import Job, LaneStatus, LoadedModel


class FakeUnit:
    """Enough of the duck-typed unit contract for the autoload gate."""

    def __init__(self, uid="primary", models=(), idle_s=9999.0,
                 swap=False, job=None, raises=False):
        self.cfg = SimpleNamespace(id=uid, gpu_uuid=f"GPU-{uid}", enabled=True)
        self.models = list(models)
        self.idle_s = idle_s
        self.swap = swap
        self._active_job_id = job
        self.raises = raises
        self.loads: list = []
        self.registry = None

    async def status(self) -> LaneStatus:
        if self.raises:
            raise RuntimeError("node unreachable")
        return LaneStatus(
            id=self.cfg.id, name=self.cfg.id,
            owner="vllm" if self.models else "free",
            ollama_up=False, vllm_up=bool(self.models),
            loaded_models=[LoadedModel(server="vllm", model=m)
                           for m in self.models],
            swap_in_progress=self.swap,
            active_job_id=self._active_job_id,
            idle_s=self.idle_s,
        )

    def load(self, req) -> Job:
        self.loads.append(req.model)
        return Job(id=f"job-{len(self.loads)}", kind=f"load:{req.lane}",
                   state="running", created_at=time.time())


class _SettingsWith:
    """Real Settings plus a `units()` returning our fake unit's config.

    Settings is a pydantic model (no attribute assignment), so units() cannot be
    patched onto it; everything else delegates, which keeps classify_usage and
    `autoload_skip_busy` reading the genuine values rather than stubs.
    """

    def __init__(self, real, cfgs):
        self._real, self._cfgs = real, cfgs

    def units(self):
        return self._cfgs

    def __getattr__(self, name):
        return getattr(self._real, name)


def _orch(unit, *, defaults=(("vllm", "vl32"),), leases=None, **overrides):
    """A minimal Orchestrator with one fake unit, without running __init__."""
    from llmconfig.orchestrator import Orchestrator
    o = Orchestrator.__new__(Orchestrator)
    o.s = _SettingsWith(Settings(_env_file=None, **overrides), [unit.cfg])
    o.units = {unit.cfg.id: unit}
    o.leases = leases
    o.monitor = None
    o.defaults_for = lambda uid: [{"server": s, "model": m} for s, m in defaults]
    return o


# --------------------------------------------------------------------------- #
# The core rule
# --------------------------------------------------------------------------- #
async def test_loads_into_a_free_unit():
    u = FakeUnit(models=[])
    started = await _orch(u).autoload_defaults()
    assert u.loads == ["vl32"] and len(started) == 1


async def test_loads_over_an_idle_model():
    """Idle IS displaceable — that is the user-facing rule ('no models, or
    models that are idle')."""
    u = FakeUnit(models=["harrier-oss-06b"], idle_s=9999.0)
    await _orch(u).autoload_defaults()
    assert u.loads == ["vl32"]


async def test_skips_a_unit_with_an_active_model():
    """The case that matters: a working lane keeps what it has."""
    u = FakeUnit(models=["harrier-oss-06b"], idle_s=1.0)   # < usage_active_window_s
    started = await _orch(u).autoload_defaults()
    assert u.loads == [] and started == []


async def test_skips_a_unit_mid_swap():
    u = FakeUnit(models=[], swap=True)
    await _orch(u).autoload_defaults()
    assert u.loads == []


async def test_skips_a_unit_with_a_load_already_in_flight():
    u = FakeUnit(models=[], job="someone-elses-job")
    await _orch(u).autoload_defaults()
    assert u.loads == []


async def test_unseeable_unit_is_left_alone():
    """A failed status probe must NOT be read as 'free'. If we cannot see what a
    unit holds we must not load over it."""
    u = FakeUnit(raises=True)
    await _orch(u).autoload_defaults()
    assert u.loads == []


# --------------------------------------------------------------------------- #
# Leases — an explicit claim outranks any activity heuristic
# --------------------------------------------------------------------------- #
def _leases_holding(holder="litrank-embed", model="harrier-embed"):
    lease = SimpleNamespace(id="abc123", holder=holder, model=model)
    return SimpleNamespace(active_for=lambda uid, m=None: lease)


async def test_skips_a_leased_unit_even_when_idle():
    """ANY live lease blocks, preemptible included — the same convention the
    idle reaper uses, and for the same reason: autoload is a convenience, not a
    competing request. A batch holder pausing between bursts still owns its unit."""
    u = FakeUnit(models=["harrier-oss-06b"], idle_s=9999.0)
    await _orch(u, leases=_leases_holding()).autoload_defaults()
    assert u.loads == []


async def test_lease_skip_names_the_holder():
    u = FakeUnit(models=[], idle_s=9999.0)
    o = _orch(u, leases=_leases_holding())
    reason = await o.autoload_skip_reason(u)
    assert "litrank-embed" in reason and "abc123" in reason


# --------------------------------------------------------------------------- #
# Escape hatches
# --------------------------------------------------------------------------- #
async def test_skip_busy_false_restores_unconditional_loading():
    u = FakeUnit(models=["harrier-oss-06b"], idle_s=1.0)
    await _orch(u).autoload_defaults(skip_busy=False)
    assert u.loads == ["vl32"], "explicit override must bypass the gate"


async def test_setting_off_restores_unconditional_loading():
    u = FakeUnit(models=["harrier-oss-06b"], idle_s=1.0)
    await _orch(u, autoload_skip_busy=False).autoload_defaults()
    assert u.loads == ["vl32"], "AUTOLOAD_SKIP_BUSY=false is the kill switch"


async def test_a_unit_with_no_defaults_is_never_probed():
    """No defaults -> no work -> no status probe. Cheap, and it keeps a
    tombstoned unit (an explicitly empty list) completely inert."""
    u = FakeUnit(models=[], raises=True)    # would raise if probed
    started = await _orch(u, defaults=()).autoload_defaults()
    assert started == [] and u.loads == []
