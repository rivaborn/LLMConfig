"""Runtime reload of on-disk catalogs and state — without restarting the app.

Why this exists. Every catalog here is read exactly ONCE, from its holder's
`__init__` (`Registry.load()`, `SparkRegistry.load()`, `Cookbook.load()`, …),
and only the *per-node* Spark registry has a write API. So the documented way to
add a multi-node recipe — hand-editing `data/spark_cluster_models.yaml` — was
inert until the `LLMConfig` Scheduled Task was restarted. Live on 2026-08-01 a
`--max-model-len 262144` pin added to a cluster entry sat unread for exactly
that reason, and the restart that applied it dropped the gateway, re-ran the
boot reclaim, and re-triggered `autoload_defaults()`. Editing a catalog should
not cost the fleet a bounce.

The two halves are deliberately asymmetric.

**Catalogs and persisted state re-`load()` in place.** Every holder follows the
same shape: `load()` re-reads its file into fresh in-memory dicts, tolerantly
(a corrupt file degrades to empty rather than raising — see each `load`). Each
is reached through the very object the units already hold (`unit.registry`,
`orch.cluster_registry`, …), so nothing is rebound and no unit is rebuilt. That
is what makes this safe while models are resident: a reload swaps the *catalog*,
never the running deployment. An in-flight job keeps the entry it already
resolved — entries are immutable pydantic objects and the dict is replaced
wholesale, so a concurrent reader either sees the old map or the new one.

**`Settings` is NOT hot-swapped wholesale**, and that is not laziness. Units,
lanes and groups are CONSTRUCTED from it — `settings.lanes()`, `settings.sparks()`,
GPU UUIDs, slot tables, `spark_fabric_links`. Mutating a structural field under a
live orchestrator makes `settings.units()` disagree with `orch.units`, and that
disagreement is exactly what invariant 18's group bookkeeping and invariant 1's
UUID pinning rely on being impossible. So `reload_settings` re-reads `.env`,
applies only fields that are read per-call, and REPORTS the rest as
restart-required instead of pretending they took effect. A field whose
consumption pattern is not obvious is treated as structural: the failure mode of
denying a live change is a restart, the failure mode of allowing one is a lane
pinned to a UUID it no longer matches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .config import Settings

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from .cookbook import Cookbook
    from .load_times import LoadTimes
    from .orchestrator import Orchestrator
    from .registry import Registry

# Fields consumed while BUILDING units/lanes/groups, binding the socket, or
# opening a state file. Changing one of these under a running process yields a
# half-applied world, so they are reported and left alone. When in doubt, add the
# field here — see the module docstring.
STRUCTURAL_FIELDS: frozenset[str] = frozenset({
    # process binding
    "llmconfig_host", "llmconfig_port",
    # primary lane construction
    "gpu_uuid", "vram_total_mb", "vram_free_baseline_mb",
    "ollama_url", "ollama_service_name",
    "vllm_relay_url", "vllm_serve_script", "vllm_systemd_unit", "vllm_venv_activate",
    "wsl_distro", "wsl_user",
    # catalog paths — the holders opened these at construction
    "registry_path", "spark_cluster_registry_path", "spark_group_state_path",
    # spark unit + group construction (invariants 1, 14, 18)
    "spark_enabled", "spark_nodes", "spark_ssh_user", "spark_api_port",
    "spark_cluster", "spark_vram_total_mb", "spark_max_models",
    "spark_fabric_enabled", "spark_fabric_links",
    # sampler + lease machinery started once in the lifespan
    "monitor_enabled", "monitor_interval_s", "monitor_retention_h",
    "monitor_persist", "monitor_db_path",
    "lease_enabled",
})

# Every companion_* field feeds the companion LaneConfig, so the whole prefix is
# structural (including its slot table, which decides Lane vs SlotLane).
STRUCTURAL_PREFIXES: tuple[str, ...] = ("companion_",)

# Never echo these back in a diff — the endpoint is LAN-open on read paths.
SECRET_FIELDS: frozenset[str] = frozenset({"llmconfig_api_key", "hf_token"})

_REDACTED = "(changed, redacted)"


def is_structural(field: str) -> bool:
    """True when `field` may only change across a restart."""
    return field in STRUCTURAL_FIELDS or field.startswith(STRUCTURAL_PREFIXES)


def _describe(field: str, old: Any, new: Any) -> str:
    if field in SECRET_FIELDS:
        return _REDACTED
    return f"{old!r} -> {new!r}"


def _count(holder: Any) -> Optional[int]:
    """Best-effort entry count for the reload report; None when not countable."""
    entries = getattr(holder, "entries", None)
    if callable(entries):
        try:
            return len(entries())
        except Exception:  # noqa: BLE001 — a count is never worth failing a reload
            return None
    return None


def reload_catalogs(
    orch: "Orchestrator",
    registry: "Registry",
    cookbook: Optional["Cookbook"] = None,
    load_times: Optional["LoadTimes"] = None,
) -> dict[str, Optional[int]]:
    """Re-read every disk-backed catalog and state file in place.

    Returns `{label: entry_count or None}`. Holders are de-duplicated by identity
    because the primary lane shares the app's `Registry` instance (see
    `Orchestrator.__init__`), so reloading it twice would be harmless but would
    report a phantom second catalog.
    """
    out: dict[str, Optional[int]] = {}
    seen: set[int] = set()

    def _do(label: str, holder: Any) -> None:
        if holder is None or id(holder) in seen:
            return
        seen.add(id(holder))
        holder.load()
        out[label] = _count(holder)

    # vLLM alias catalog (the primary lane holds this same object).
    _do("vllm_registry", registry)

    # Per-unit catalogs: Lane/SlotLane and SparkUnit both expose `.registry`.
    # A SparkGroup does not — it reads the cluster registry, reloaded below.
    for uid, unit in orch.units.items():
        _do(f"unit:{uid}", getattr(unit, "registry", None))

    # Multi-node recipes live ONLY here (invariant 18) — the entry that started
    # all this, and the one with no write API.
    _do("cluster_registry", getattr(orch, "cluster_registry", None))

    # Persisted policy/state that is likewise read once at construction.
    _do("lane_defaults", getattr(orch, "defaults", None))
    _do("lane_pins", getattr(orch, "pins", None))
    _do("group_placements", getattr(orch, "group_placements", None))
    _do("cookbook", cookbook)
    _do("load_times", load_times)

    return out


def reload_settings(settings: Settings) -> tuple[dict[str, str], dict[str, str]]:
    """Re-read `.env`/env and apply the runtime-safe half.

    Mutates `settings` IN PLACE rather than rebinding, because `get_settings()`
    is `@lru_cache`d and every holder captured that one instance at construction —
    an in-place write is what makes the change visible to all of them at once.

    Returns `(applied, restart_required)`, each `{field: "old -> new"}` with
    secrets redacted.
    """
    fresh = Settings()
    applied: dict[str, str] = {}
    restart_required: dict[str, str] = {}

    for field in type(settings).model_fields:
        old = getattr(settings, field, None)
        new = getattr(fresh, field, None)
        if old == new:
            continue
        if is_structural(field):
            restart_required[field] = _describe(field, old, new)
            continue
        try:
            setattr(settings, field, new)
        except Exception:  # noqa: BLE001 — a rejected assignment is a restart, not a 500
            restart_required[field] = _describe(field, old, new)
            continue
        applied[field] = _describe(field, old, new)

    return applied, restart_required
