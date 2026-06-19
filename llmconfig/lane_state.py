"""Persisted per-lane default model — the "what runs on this card" setting.

Lets the user pick a model for a lane (e.g. the 3070 Ti companion) and have it stick
across restarts and auto-load on startup, without editing `.env`. Mirrors the
`Registry` persistence pattern: a small YAML in `data/`, user-editable. The static
`companion_default_*` settings remain the seed/fallback (see `Orchestrator`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from .config import REPO_ROOT, Settings, get_settings

DEFAULTS_PATH = REPO_ROOT / "data" / "lane_defaults.yaml"


class LaneDefaults:
    def __init__(self, settings: Settings | None = None, path: Path | None = None):
        self.s = settings or get_settings()
        self.path = path or DEFAULTS_PATH
        self._data: dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            self._data = raw.get("lanes", {}) or {}
        else:
            self._data = {}

    def get(self, lane_id: str) -> Optional[dict]:
        """The persisted override for a lane, or None if unset."""
        d = self._data.get(lane_id) or {}
        if d.get("model"):
            return {"server": d.get("server", ""), "model": d.get("model", "")}
        return None

    def set(self, lane_id: str, server: str, model: str) -> dict:
        entry = {"server": server, "model": model}
        self._data[lane_id] = entry
        self.save()
        return entry

    def clear(self, lane_id: str) -> bool:
        existed = self._data.pop(lane_id, None) is not None
        if existed:
            self.save()
        return existed

    def all(self) -> dict[str, dict]:
        return dict(self._data)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            yaml.safe_dump({"lanes": self._data}, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
