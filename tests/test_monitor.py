import time
from collections import deque

import llmconfig.gpu as gpu
import llmconfig.monitor as monitor
from llmconfig.config import Settings
from llmconfig.gpu import GpuMetric
from llmconfig.monitor import Monitor, _bucketize
from llmconfig.proc import CmdResult

G3090 = "GPU-739bece9-8298-7993-f7dd-c8d86cb541f9"
G3070 = "GPU-2caf7863-102e-31e5-be4d-5ec860addc78"


async def test_sample_gpu_metrics_parses_and_handles_na(monkeypatch):
    async def fake_run_argv(argv, timeout=15.0):
        if "--query-gpu" in " ".join(argv) and "temperature.gpu" in " ".join(argv):
            return CmdResult(
                0,
                f"0, {G3070}, RTX 3070 Ti, 73, [N/A], 0, 8192, 11, 8181\n"
                f"1, {G3090}, RTX 3090, 51, 117.36, 95, 24576, 22488, 2088\n",
                "",
            )
        return CmdResult(127, "", "no")

    monkeypatch.setattr(gpu, "run_argv", fake_run_argv)
    ms = await gpu.sample_gpu_metrics(Settings(_env_file=None))
    assert len(ms) == 2
    assert ms[0].power_w is None  # "[N/A]" → None, not a crash
    assert ms[1].temp_c == 51.0 and ms[1].power_w == 117.36 and ms[1].mem_used_mb == 22488
    assert ms[1].index == 1 and ms[1].uuid == G3090


def test_bucketize_peaks_and_skips_not_running():
    now = time.time()
    # two points in the same bucket — peak (max) wins
    pts = deque([(now - 10, 50.0, 0.0, 0.0, 0.0), (now - 9, 80.0, 0.0, 0.0, 0.0)])
    out = _bucketize(pts, now - 3600, 3600, 1)
    assert len(out) == 1 and out[0][1] == 80.0

    # running_idx skips points whose flag is falsy (Ollama not running)
    olla = deque([(now - 20, 90.0, 10.0, "m", True), (now - 10, 0.0, 0.0, "", False)])
    kept = _bucketize(olla, now - 3600, 3600, 1, running_idx=4)
    assert [v for _, v in kept] == [90.0]


class _FakeOllama:
    def __init__(self, running):
        self._running = running

    async def _ps_raw(self):
        return self._running


class _FakeOrch:
    def __init__(self, running):
        self.primary = type("L", (), {"ollama": _FakeOllama(running)})()


async def test_monitor_snapshot_and_history(monkeypatch):
    async def fake_metrics(settings=None):
        return [
            GpuMetric(0, G3070, "RTX 3070 Ti", 73.0, 36.7, 0.0, 8192, 11, 8181),
            GpuMetric(1, G3090, "RTX 3090", 51.0, 117.0, 95.0, 24576, 22488, 2088),
        ]

    # 3090 exposes both sensors; 3070 Ti exposes hotspot only.
    def fake_channels(index):
        return {"hotspot": 66.0, "memory": 64.0} if index == 1 else {"hotspot": 84.0}

    monkeypatch.setattr(monitor, "sample_gpu_metrics", fake_metrics)
    monkeypatch.setattr(monitor.nvapi, "read_thermal_channels", fake_channels)

    running = [{"name": "qwen3.6:27b-96k", "size": 100, "size_vram": 88}]  # spilled 12%
    mon = Monitor(Settings(_env_file=None), _FakeOrch(running))
    await mon._sample_once()

    snap = mon.snapshot()
    assert [g["uuid"] for g in snap["gpus"]] == [G3070, G3090]  # nvidia-smi index order
    g3090 = snap["gpus"][1]
    assert g3090["hotspot_c"] == 66.0 and g3090["junction_c"] == 64.0
    assert g3090["mem_pct"] == round(100 * 22488 / 24576, 1)
    assert snap["gpus"][0]["junction_c"] is None  # 3070 Ti has no junction sensor
    assert snap["ollama"]["spilled"] is True and snap["ollama"]["cpu_pct"] == 12.0

    hist = mon.history(3600)
    assert len(hist["gpus"]) == 2
    assert len(hist["gpus"][1]["series"]["junction"]) == 1
    assert len(hist["gpus"][0]["series"]["junction"]) == 0  # none recorded for the 3070 Ti
    assert len(hist["ollama"]["gpu_pct"]) == 1


async def test_monitor_persists_across_restart(tmp_path, monkeypatch):
    async def fake_metrics(settings=None):
        return [
            GpuMetric(0, G3070, "RTX 3070 Ti", 73.0, 36.7, 0.0, 8192, 11, 8181),
            GpuMetric(1, G3090, "RTX 3090", 51.0, 117.0, 95.0, 24576, 22488, 2088),
        ]

    def fake_channels(index):
        return {"hotspot": 66.0, "memory": 64.0} if index == 1 else {"hotspot": 84.0}

    monkeypatch.setattr(monitor, "sample_gpu_metrics", fake_metrics)
    monkeypatch.setattr(monitor.nvapi, "read_thermal_channels", fake_channels)

    running = [{"name": "qwen3.6:27b-96k", "size": 100, "size_vram": 88}]
    db = tmp_path / "monitor.db"
    s = Settings(_env_file=None, monitor_db_path=db)

    # First instance: open DB, sample once, shut down (flushes + closes).
    m1 = Monitor(s, _FakeOrch(running))
    m1._open_db()
    m1._load_history()
    await m1._sample_once()
    await m1.stop()
    assert db.exists()

    # Fresh instance against the same DB rehydrates the deques from disk.
    m2 = Monitor(s, _FakeOrch(running))
    m2._open_db()
    m2._load_history()
    snap = m2.snapshot()
    assert [g["uuid"] for g in snap["gpus"]] == [G3070, G3090]  # index order preserved
    g3090 = snap["gpus"][1]
    assert g3090["temp_c"] == 51.0 and g3090["hotspot_c"] == 66.0 and g3090["junction_c"] == 64.0
    assert snap["ollama"]["spilled"] is True and snap["ollama"]["cpu_pct"] == 12.0
    assert len(m2._gpus[G3090].points) == 1
    await m2.stop()
