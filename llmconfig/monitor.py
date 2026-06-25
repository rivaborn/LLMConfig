"""In-process GPU/LLM telemetry collector — the data behind the Monitor tab.

A background asyncio task samples every visible GPU (core/hotspot/junction temp,
power, utilization, VRAM) plus the primary Ollama lane's GPU-vs-CPU split, and
keeps a rolling in-memory history (no DB, no extra dependency). The original
`nmon` TUI did the same with SQLite; here the window is small enough to live in
deques and is naturally disposable — it rebuilds the moment the app restarts.

Sources:
  * nvidia-smi (via gpu.sample_gpu_metrics) — core temp, power, util, memory.
  * NVAPI (llmconfig.nvapi) — hotspot + GDDR6X junction temps the 3090 hides
    from nvidia-smi; paired to each card by driver index.
  * Ollama /api/ps (primary lane) — size vs size_vram → on-GPU vs spilled-to-CPU.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Optional

from . import nvapi
from .config import Settings
from .gpu import sample_gpu_metrics

if TYPE_CHECKING:
    from .orchestrator import Orchestrator

log = logging.getLogger(__name__)

# A GPU history point: (ts, temp, hotspot, junction, power, util, mem_used_mb).
# An Ollama point: (ts, gpu_pct, cpu_pct).
_HISTORY_BUCKETS = 180  # max points returned per series; peak-per-bucket keeps spikes


class GpuTrack:
    """Ring buffer + identity for one card."""

    def __init__(self, index: int, uuid: str, name: str, mem_total_mb: int):
        self.index = index
        self.uuid = uuid
        self.name = name
        self.mem_total_mb = mem_total_mb
        self.points: deque[tuple] = deque()


class Monitor:
    def __init__(self, settings: Settings, orch: "Orchestrator"):
        self.s = settings
        self.orch = orch
        self.interval = max(1.0, float(settings.monitor_interval_s))
        self.retention_s = max(1, int(settings.monitor_retention_h)) * 3600
        self._gpus: dict[str, GpuTrack] = {}     # uuid -> track
        self._order: list[str] = []              # uuids in nvidia-smi index order
        self._ollama: deque[tuple] = deque()     # (ts, gpu_pct, cpu_pct, model, running)
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._last_error = ""
        self._last_sample_ts = 0.0

    # ---- lifecycle ----
    def start(self) -> None:
        if not self.s.monitor_enabled or self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="monitor-sampler")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 — best-effort shutdown
                pass
            self._task = None

    async def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                await self._sample_once()
            except Exception as e:  # noqa: BLE001 — a sampler hiccup must never kill the loop
                self._last_error = f"{type(e).__name__}: {e}"
                log.debug("monitor sample failed: %s", self._last_error)
            elapsed = time.monotonic() - t0
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=max(0.0, self.interval - elapsed))
            except asyncio.TimeoutError:
                pass

    # ---- sampling ----
    async def _sample_once(self) -> None:
        metrics = await sample_gpu_metrics(self.s)
        now = time.time()
        if metrics:
            self._last_error = ""
            self._last_sample_ts = now
        order: list[str] = []
        for m in metrics:
            order.append(m.uuid)
            track = self._gpus.get(m.uuid)
            if track is None:
                track = GpuTrack(m.index, m.uuid, m.name, m.mem_total_mb)
                self._gpus[m.uuid] = track
            track.index, track.name = m.index, m.name
            if m.mem_total_mb:
                track.mem_total_mb = m.mem_total_mb
            # NVAPI is a blocking ctypes call — keep it off the event loop.
            try:
                channels = await asyncio.to_thread(nvapi.read_thermal_channels, m.index)
            except Exception:  # noqa: BLE001
                channels = None
            channels = channels or {}
            track.points.append((
                now, m.temp_c, channels.get("hotspot"), channels.get("memory"),
                m.power_w, m.util_pct, m.mem_used_mb,
            ))
            self._prune(track.points, now)
        if order:
            self._order = order
        await self._sample_ollama(now)

    async def _sample_ollama(self, now: float) -> None:
        """Primary lane's Ollama GPU/CPU split (size_vram / size)."""
        try:
            running = await self.orch.primary.ollama._ps_raw()
        except Exception:  # noqa: BLE001 — Ollama down is a normal state, not an error
            running = []
        if running:
            m = running[0]
            size = int(m.get("size", 0) or 0)
            vram = int(m.get("size_vram", 0) or 0)
            gpu_pct = round(100.0 * vram / size, 1) if size else 0.0
            self._ollama.append((now, gpu_pct, round(100.0 - gpu_pct, 1), m.get("name", ""), True))
        else:
            self._ollama.append((now, 0.0, 0.0, "", False))
        self._prune(self._ollama, now)

    def _prune(self, dq: deque, now: float) -> None:
        cutoff = now - self.retention_s
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    # ---- read API ----
    def snapshot(self) -> dict:
        """Latest reading per GPU + rolling aggregates + Ollama split."""
        now = time.time()
        gpus = []
        for uuid in self._order or list(self._gpus):
            t = self._gpus.get(uuid)
            if t is None or not t.points:
                continue
            last = t.points[-1]
            temps_24h = [p[1] for p in t.points if p[1] is not None]
            temps_1h = [p[1] for p in t.points if p[1] is not None and p[0] >= now - 3600]
            mem_used = last[6]
            gpus.append({
                "index": t.index,
                "uuid": t.uuid,
                "name": t.name,
                "temp_c": last[1],
                "hotspot_c": last[2],
                "junction_c": last[3],
                "power_w": last[4],
                "util_pct": last[5],
                "mem_used_mb": mem_used,
                "mem_total_mb": t.mem_total_mb,
                "mem_pct": round(100.0 * mem_used / t.mem_total_mb, 1) if t.mem_total_mb else 0.0,
                "temp_max_24h": max(temps_24h) if temps_24h else None,
                "temp_avg_1h": round(sum(temps_1h) / len(temps_1h), 1) if temps_1h else None,
            })
        ollama = None
        if self._ollama:
            ots, gpu_pct, cpu_pct, model, run = self._ollama[-1]
            ollama = {"running": bool(run), "model": model, "gpu_pct": gpu_pct,
                      "cpu_pct": cpu_pct, "spilled": run and cpu_pct > 0.0}
        return {
            "interval_s": self.interval,
            "retention_h": self.retention_s // 3600,
            "enabled": self.s.monitor_enabled,
            "last_sample_ts": self._last_sample_ts,
            "stale": bool(self._last_sample_ts) and (now - self._last_sample_ts) > self.interval * 3,
            "error": self._last_error,
            "gpus": gpus,
            "ollama": ollama,
        }

    def history(self, window_s: float) -> dict:
        """Bucketed series per GPU (+ Ollama split) over the last `window_s`."""
        now = time.time()
        window_s = max(60.0, min(float(window_s), float(self.retention_s)))
        since = now - window_s
        gpus = []
        for uuid in self._order or list(self._gpus):
            t = self._gpus.get(uuid)
            if t is None:
                continue
            gpus.append({
                "index": t.index,
                "uuid": t.uuid,
                "name": t.name,
                "mem_total_mb": t.mem_total_mb,
                "series": {
                    "temp": _bucketize(t.points, since, window_s, 1),
                    "hotspot": _bucketize(t.points, since, window_s, 2),
                    "junction": _bucketize(t.points, since, window_s, 3),
                    "power": _bucketize(t.points, since, window_s, 4),
                    "vram": _bucketize(t.points, since, window_s, 6),
                },
            })
        ollama = {
            "gpu_pct": _bucketize(self._ollama, since, window_s, 1, running_idx=4),
            "cpu_pct": _bucketize(self._ollama, since, window_s, 2, running_idx=4),
        }
        return {"window_s": window_s, "interval_s": self.interval, "gpus": gpus, "ollama": ollama}


def _bucketize(points: deque, since: float, window_s: float, idx: int,
               running_idx: int | None = None) -> list[list[float]]:
    """Down-sample to <=_HISTORY_BUCKETS peak-per-bucket [ts, value] pairs.

    Peak (max) per bucket preserves spikes that nearest-sample thinning would
    drop. `running_idx`, when given, skips points whose flag at that position is
    falsy (Ollama not running → no spurious 0% in the trace)."""
    width = max(window_s / _HISTORY_BUCKETS, 1e-9)
    buckets: dict[int, list[float]] = {}
    for p in points:
        ts = p[0]
        if ts < since:
            continue
        if running_idx is not None and not p[running_idx]:
            continue
        v = p[idx]
        if v is None:
            continue
        b = int((ts - since) / width)
        cur = buckets.get(b)
        if cur is None or v > cur[1]:
            buckets[b] = [ts, v]
    return [buckets[b] for b in sorted(buckets)]
