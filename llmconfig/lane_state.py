"""Persisted per-unit default models — the "what runs on this unit" setting.

Lets the user pick what a unit should serve (e.g. the 3070 Ti companion, or a
Spark node) and have it stick across restarts and auto-load on startup, without
editing `.env`. Mirrors the `Registry` persistence pattern: a small YAML in
`data/`, user-editable. The static `companion_default_*` settings remain the
seed/fallback (see `Orchestrator`).

A unit may have **several** defaults, because a DGX Spark holds several models at
once — an embedder and a reranker alongside a chat model is the whole point. A GPU
lane still takes exactly one (its eviction-wait gate guarantees a single
occupant), which is simply the one-element case.

The stored shape is a list per unit:

```yaml
lanes:
  spark4:
    models:
      - {server: spark, model: qwen3-vl-embedding-8b}
      - {server: spark, model: qwen3-vl-reranker-8b}
```

The previous scalar shape (`{server, model}` directly under the unit id) is read
transparently and rewritten as a list on the next save, so an existing
`lane_defaults.yaml` keeps working untouched.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from .config import REPO_ROOT, Settings, get_settings

DEFAULTS_PATH = REPO_ROOT / "data" / "lane_defaults.yaml"


def _entries(raw) -> list[dict]:
    """Normalize one unit's persisted value to a list of {server, model}.

    Accepts the current list shape, the pre-multi-model scalar shape, and a bare
    list — this file is user-editable, so reading it must never raise.
    """
    if not raw:
        return []
    if isinstance(raw, dict):
        items = raw.get("models")
        if items is None:
            items = [raw] if raw.get("model") else []       # legacy scalar shape
    elif isinstance(raw, list):
        items = raw
    else:
        return []
    out: list[dict] = []
    for it in items:
        if isinstance(it, dict) and it.get("model"):
            out.append({"server": str(it.get("server", "")), "model": str(it["model"])})
    return out


class LaneDefaults:
    def __init__(self, settings: Settings | None = None, path: Path | None = None):
        self.s = settings or get_settings()
        self.path = path or DEFAULTS_PATH
        self._data: dict[str, list[dict]] = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            lanes = raw.get("lanes", {}) or {}
            # A LITERALLY empty `models: []` is kept as a tombstone — "load
            # nothing at startup" (how a cookbook default pins a unit empty). A
            # non-empty list whose entries are all garbage is NOT a tombstone:
            # the author meant to configure something, so treat it as unset and
            # let the .env seed apply rather than silently suppressing it.
            self._data = {}
            for k, v in lanes.items():
                ents = _entries(v)
                if ents:
                    self._data[k] = ents
                elif isinstance(v, dict) and v.get("models") == []:
                    self._data[k] = []                   # deliberate tombstone
        else:
            self._data = {}

    # ---- reads ----
    def list(self, lane_id: str) -> list[dict]:
        """Every persisted default for a unit, in load order (may be empty)."""
        return [dict(e) for e in self._data.get(lane_id, [])]

    def get(self, lane_id: str) -> Optional[dict]:
        """The unit's FIRST default, or None — the back-compat scalar view."""
        entries = self._data.get(lane_id) or []
        return dict(entries[0]) if entries else None

    def entries_or_none(self, lane_id: str) -> Optional[list[dict]]:
        """Distinguish UNSET (None → the .env seed may apply) from explicitly
        empty ([] → the tombstone: load nothing). `list()` collapses both to []
        for callers that don't care."""
        v = self._data.get(lane_id)
        return None if v is None else [dict(e) for e in v]

    def set_empty(self, lane_id: str) -> None:
        """Write the tombstone: this unit deliberately starts with nothing."""
        self._data[lane_id] = []
        self.save()

    def all(self) -> dict[str, list[dict]]:
        return {k: [dict(e) for e in v] for k, v in self._data.items()}

    # ---- writes ----
    def set(self, lane_id: str, server: str, model: str) -> dict:
        """Make `model` the unit's ONLY default, replacing any existing list.

        That is what "set the default for this unit" has always meant, and it stays
        the right semantic for a GPU lane. Use `add` to co-schedule on a Spark.
        """
        entry = {"server": server, "model": model}
        self._data[lane_id] = [entry]
        self.save()
        return entry

    def add(self, lane_id: str, server: str, model: str) -> list[dict]:
        """Also start `model` on this unit. Idempotent — re-adding updates the server."""
        entries = [e for e in self._data.get(lane_id, []) if e["model"] != model]
        entries.append({"server": server, "model": model})
        self._data[lane_id] = entries
        self.save()
        return self.list(lane_id)

    def remove(self, lane_id: str, model: str) -> bool:
        """Drop one default. Removing the last one drops the unit entirely."""
        entries = self._data.get(lane_id) or []
        kept = [e for e in entries if e["model"] != model]
        if len(kept) == len(entries):
            return False
        if kept:
            self._data[lane_id] = kept
        else:
            self._data.pop(lane_id, None)
        self.save()
        return True

    def clear(self, lane_id: str) -> bool:
        existed = self._data.pop(lane_id, None) is not None
        if existed:
            self.save()
        return existed

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: {"models": v} for k, v in self._data.items()}
        self.path.write_text(
            yaml.safe_dump({"lanes": data}, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
