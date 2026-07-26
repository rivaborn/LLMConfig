"""Measured load durations — the data behind "≈2 min" estimates in the UI.

Loads on this fleet span two orders of magnitude: ~2 min for the reranker, ~12 min
for gemma's quantize-at-load, ~50 min for a first-ever cold weight pull. Nothing
surfaced that before you committed to a load, and `Job` timestamps can't answer it
retroactively — jobs are in-memory (pruned at 50, gone on restart) and their
duration includes queue-wait behind the unit lock.

So the UNITS record: each load body times its own launch span and calls
`record()` on success. The span deliberately excludes everything that isn't the
launch itself — queue-wait, placement-driven victim eviction, admission's SSH
probe — and fast paths ("already serving") never record at all, so the samples
answer exactly one question: *how long does a real launch of this model take?*
Failures never record either — a 60 s dead-serve fast-fail would poison the
median of a model whose real loads take 10 minutes.

Keys fold onto the catalog alias (the canonical name — see
`SparkUnit.canonical_model`) and Sparks share one key across nodes
(`spark:{alias}`) because the four GB10s are identical hardware; a sample from
spark1 is exactly as predictive for spark4. GPU lanes differ (3090 vs 3070 Ti),
so their keys carry the unit id (`{unit_id}:{server}:{alias}`).

Persistence mirrors `LaneDefaults`: a small user-editable YAML in `data/`,
tolerant loader, self-saving mutations. Last N samples per key; the estimate is
the median, so one cold pull skews at most until real loads displace it.
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Optional

import yaml

from .config import REPO_ROOT

LOAD_TIMES_PATH = REPO_ROOT / "data" / "load_times.yaml"
_MAX_SAMPLES = 5


def spark_key(alias: str) -> str:
    return f"spark:{alias}"


def lane_key(unit_id: str, server: str, alias: str) -> str:
    return f"{unit_id}:{server}:{alias}"


class LoadTimes:
    def __init__(self, path: Path | None = None):
        self.path = path or LOAD_TIMES_PATH
        self._data: dict[str, list[dict]] = {}
        self.load()

    def load(self) -> None:
        """Tolerant: this file is user-editable and must never break startup."""
        self._data = {}
        if not self.path.exists():
            return
        try:
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001 — a corrupt history is an empty history
            return
        for key, samples in (raw.get("samples") or {}).items():
            if not isinstance(samples, list):
                continue
            keep = []
            for s in samples:
                if isinstance(s, dict) and isinstance(s.get("duration_s"), (int, float)):
                    keep.append({"duration_s": float(s["duration_s"]),
                                 "unit": str(s.get("unit", "")),
                                 "ts": float(s.get("ts", 0.0))})
            if keep:
                self._data[str(key)] = keep[-_MAX_SAMPLES:]

    def record(self, key: str, duration_s: float, unit: str = "") -> None:
        """Append one successful launch measurement (last N kept)."""
        if duration_s < 0:
            return
        samples = self._data.setdefault(key, [])
        # Floor at 0.1s: real launches take minutes, but a mocked/instant one can
        # land inside the OS monotonic-clock granularity and read exactly 0.
        samples.append({"duration_s": max(round(float(duration_s), 1), 0.1),
                        "unit": unit, "ts": round(time.time(), 1)})
        del samples[:-_MAX_SAMPLES]
        self.save()

    def estimate(self, key: str) -> Optional[float]:
        """Median of the recorded samples, or None with no data."""
        samples = self._data.get(key)
        if not samples:
            return None
        return statistics.median(s["duration_s"] for s in samples)

    def all(self) -> dict[str, dict]:
        """{key: {est_s, n}} — the /api/load-times payload."""
        return {
            key: {"est_s": round(statistics.median(s["duration_s"] for s in samples), 1),
                  "n": len(samples)}
            for key, samples in self._data.items()
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            yaml.safe_dump({"samples": self._data}, sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )
