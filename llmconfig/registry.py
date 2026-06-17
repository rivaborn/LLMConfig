"""The vLLM alias registry — the editable catalog of what vLLM *can* serve.

vLLM's `/v1/models` only reports the currently-served model, so the set of
available vLLM models is this registry (seeded from the documented serve.sh
table into ./data/vllm_models.yaml, then user-editable).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from .config import PACKAGE_DIR, Settings, get_settings
from .schemas import VllmAliasEntry

DEFAULT_REGISTRY = PACKAGE_DIR / "data" / "vllm_models.default.yaml"


class Registry:
    def __init__(self, path: Path):
        self.path = path
        self._entries: dict[str, VllmAliasEntry] = {}
        self.load()

    # ---- persistence ----
    def load(self) -> None:
        if not self.path.exists():
            self._seed()
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        self._entries = {}
        for item in raw.get("aliases", []) or []:
            entry = VllmAliasEntry(**item)
            self._entries[entry.alias] = entry

    def _seed(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DEFAULT_REGISTRY, self.path)

    def save(self) -> None:
        data = {"aliases": [e.model_dump() for e in self._entries.values()]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # ---- queries ----
    def entries(self) -> list[VllmAliasEntry]:
        return list(self._entries.values())

    def get(self, alias: str) -> VllmAliasEntry | None:
        return self._entries.get(alias)

    def served_name(self, alias: str) -> str | None:
        e = self.get(alias)
        return (e.served_name or e.alias) if e else None

    # ---- mutations (model management) ----
    def upsert(self, entry: VllmAliasEntry) -> None:
        self._entries[entry.alias] = entry
        self.save()

    def remove(self, alias: str) -> bool:
        existed = self._entries.pop(alias, None) is not None
        if existed:
            self.save()
        return existed


def make_registry(settings: Settings | None = None) -> Registry:
    settings = settings or get_settings()
    return Registry(settings.registry_path)
