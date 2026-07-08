"""In-process GPU/LLM telemetry collector — the data behind the Monitor tab.

A background asyncio task samples every visible GPU (core/hotspot/junction temp,
power, utilization, VRAM) plus the primary Ollama lane's GPU-vs-CPU split, and
keeps a rolling in-memory history in deques. Like the original `nmon` TUI, samples
are also persisted to a small SQLite DB (`monitor_db_path`), so the history window
survives an app/service restart instead of rebuilding from empty: on `start()` the
last `retention_h` of rows are loaded back into the deques. Persistence is
best-effort — a DB failure disables it and the collector keeps running in-memory.

Sources:
  * nvidia-smi (via gpu.sample_gpu_metrics) — core temp, power, util, memory.
  * NVAPI (llmconfig.nvapi) — hotspot + GDDR6X junction temps the 3090 hides
    from nvidia-smi; paired to each card by driver index.
  * Ollama /api/ps (primary lane) — size vs size_vram → on-GPU vs spilled-to-CPU.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from . import nvapi
from .config import Settings
from .gpu import sample_gpu_metrics

if TYPE_CHECKING:
    from .orchestrator import Orchestrator

log = logging.getLogger(__name__)

# A GPU history point: (ts, temp, hotspot, junction, power, util, mem_used_mb).
# An Ollama point: (ts, gpu_pct, cpu_pct, model, running).
_HISTORY_BUCKETS = 180  # max points returned per series; peak-per-bucket keeps spikes
_PRUNE_EVERY = 60       # prune the on-disk window once per this many writes

_SCHEMA = """
CREATE TABLE IF NOT EXISTS gpu_meta (
    uuid TEXT PRIMARY KEY, idx INTEGER, name TEXT, mem_total_mb INTEGER
);
CREATE TABLE IF NOT EXISTS gpu_samples (
    ts REAL, uuid TEXT, temp REAL, hotspot REAL, junction REAL,
    power REAL, util REAL, mem_used_mb INTEGER
);
CREATE INDEX IF NOT EXISTS ix_gpu_samples_ts ON gpu_samples(ts);
CREATE TABLE IF NOT EXISTS ollama_samples (
    ts REAL, gpu_pct REAL, cpu_pct REAL, model TEXT, running INTEGER
);
CREATE INDEX IF NOT EXISTS ix_ollama_samples_ts ON ollama_samples(ts);
"""


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
        # persistence (best-effort SQLite; None when disabled or unavailable)
        self._persist = bool(settings.monitor_persist)
        self._db_path = Path(settings.monitor_db_path)
        self._db: Optional[sqlite3.Connection] = None
        self._db_lock = threading.Lock()
        self._writes = 0

    # ---- lifecycle ----
    def start(self) -> None:
        if not self.s.monitor_enabled or self._task is not None:
            return
        if self._persist:
            try:
                self._open_db()
                self._load_history()
            except Exception as e:  # noqa: BLE001 — never let persistence block startup
                log.warning("monitor persistence disabled (%s): %s", type(e).__name__, e)
                self._db = None
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
        if self._db is not None:
            try:
                with self._db_lock:
                    self._db.commit()
                    self._db.close()
            except Exception:  # noqa: BLE001
                pass
            self._db = None

    # ---- persistence ----
    def _open_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(str(self._db_path), check_same_thread=False)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.executescript(_SCHEMA)
        db.commit()
        self._db = db

    def _load_history(self) -> None:
        """Rehydrate the deques from the retained on-disk window (newest restart)."""
        if self._db is None:
            return
        now = time.time()
        cutoff = now - self.retention_s
        with self._db_lock:
            self._db.execute("DELETE FROM gpu_samples WHERE ts < ?", (cutoff,))
            self._db.execute("DELETE FROM ollama_samples WHERE ts < ?", (cutoff,))
            self._db.commit()
            meta = self._db.execute(
                "SELECT uuid, idx, name, mem_total_mb FROM gpu_meta").fetchall()
            order = sorted(meta, key=lambda r: r[1] if r[1] is not None else 1 << 30)
            for uuid, idx, name, mem_total in order:
                self._gpus[uuid] = GpuTrack(idx or 0, uuid, name or "", mem_total or 0)
            self._order = [r[0] for r in order]
            for uuid in self._order:
                rows = self._db.execute(
                    "SELECT ts,temp,hotspot,junction,power,util,mem_used_mb "
                    "FROM gpu_samples WHERE uuid=? AND ts>=? ORDER BY ts",
                    (uuid, cutoff)).fetchall()
                self._gpus[uuid].points.extend(tuple(r) for r in rows)
            orows = self._db.execute(
                "SELECT ts,gpu_pct,cpu_pct,model,running FROM ollama_samples "
                "WHERE ts>=? ORDER BY ts", (cutoff,)).fetchall()
            self._ollama.extend(
                (ts, gp, cp, model or "", bool(run)) for ts, gp, cp, model, run in orows)

    def _persist_sample(self, now: float, gpu_rows: list[tuple],
                        ollama_row: Optional[tuple]) -> None:
        """Append this tick's rows; prune the on-disk window periodically.

        `gpu_rows`: (uuid, idx, name, mem_total, temp, hotspot, junction, power,
        util, mem_used). Runs in a worker thread — never touches the deques."""
        if self._db is None:
            return
        try:
            with self._db_lock:
                self._db.executemany(
                    "INSERT OR REPLACE INTO gpu_meta(uuid,idx,name,mem_total_mb) "
                    "VALUES(?,?,?,?)",
                    [(r[0], r[1], r[2], r[3]) for r in gpu_rows])
                self._db.executemany(
                    "INSERT INTO gpu_samples"
                    "(ts,uuid,temp,hotspot,junction,power,util,mem_used_mb) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    [(now, r[0], r[4], r[5], r[6], r[7], r[8], r[9]) for r in gpu_rows])
                if ollama_row is not None:
                    self._db.execute(
                        "INSERT INTO ollama_samples(ts,gpu_pct,cpu_pct,model,running) "
                        "VALUES(?,?,?,?,?)",
                        (ollama_row[0], ollama_row[1], ollama_row[2],
                         ollama_row[3], int(bool(ollama_row[4]))))
                self._writes += 1
                if self._writes % _PRUNE_EVERY == 0:
                    cutoff = now - self.retention_s
                    self._db.execute("DELETE FROM gpu_samples WHERE ts < ?", (cutoff,))
                    self._db.execute("DELETE FROM ollama_samples WHERE ts < ?", (cutoff,))
                self._db.commit()
        except Exception as e:  # noqa: BLE001 — a persist hiccup must never kill the loop
            log.debug("monitor persist failed: %s", e)

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
        gpu_rows: list[tuple] = []
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
            hotspot, junction = channels.get("hotspot"), channels.get("memory")
            track.points.append((
                now, m.temp_c, hotspot, junction,
                m.power_w, m.util_pct, m.mem_used_mb,
            ))
            self._prune(track.points, now)
            gpu_rows.append((m.uuid, m.index, m.name, track.mem_total_mb,
                             m.temp_c, hotspot, junction, m.power_w, m.util_pct,
                             m.mem_used_mb))
        if order:
            self._order = order
        await self._sample_ollama(now)
        if self._db is not None:
            ollama_row = self._ollama[-1] if self._ollama else None
            await asyncio.to_thread(self._persist_sample, now, gpu_rows, ollama_row)

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
    def last_util_activity(self, uuid: str, threshold: float, since: float) -> float | None:
        """Newest sample ts for `uuid` with util_pct > threshold and ts > since, else None.

        The idle reaper's utilization signal: scans the deque tail newest-first and
        stops at `since`, so each reaper tick only touches the points added since the
        previous one (~12/min at the 5 s cadence). Returns None when the monitor is
        disabled, the UUID was never sampled, or util read N/A — the reaper then runs
        on gateway/load timestamps alone.
        """
        track = self._gpus.get(uuid)
        if track is None:
            return None
        for p in reversed(track.points):
            if p[0] <= since:
                return None
            util = p[5]
            if util is not None and util > threshold:
                return p[0]
        return None

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
