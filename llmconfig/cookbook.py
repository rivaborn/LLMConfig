"""Cookbook — named fleet states: save what's loaded where, get back to it later.

After experiments (or the load-order landmine striking), rebuilding "the fleet
arrangement I like" was a series of manual loads in the right order. A cookbook
STATE is a snapshot of every enabled unit's resident models, saved under a name;
APPLYING it makes the fleet match — loading what's missing, unloading what isn't
in the state ("all the models, and only those models"). One state can be marked
DEFAULT, which also syncs `LaneDefaults` so the existing startup autoload
reproduces it — including `[]` tombstones for units the state pins empty.

## What a state is NOT

Not enforcement. Apply is a point-in-time action: auto-load will happily
re-materialize a displaced model on its holder's next request, and auto-placement
keeps making its own choices afterwards. `status()` reports whether the fleet (and
`LaneDefaults`) still match; nothing fights drift.

## Apply semantics (the sharp edges, deliberately)

* One apply at a time — a second is refused while a `cookbook:apply:` job is
  live; two applies interleaving through the per-unit locks would produce an
  arbitrary hybrid of both states.
* Units run in PARALLEL, each unit's steps strictly ordered: unload extras
  FIRST (or a load can win the unit lock and fail admission against a
  not-yet-removed co-tenant), then load missing models.
* A missing model with `needs_empty_node` (gemma's quantize-at-load transient,
  the reranker's fastsafetensors loader — see the ops runbook's load-order
  landmine) forces the whole unit to be freed and the FULL target set reloaded,
  that model first. Reproducibility beats minimal churn.
* Leases: a NON-preemptible lease blocks the unload it covers — skipped and
  reported, matching `/api/unload`. A preemptible hold (opencode's auto-hold) is
  displaced but reported in `displaced_holds`, because the holder's next request
  will quietly re-load the model — the operator should know the state may not
  stick.
* Child load jobs are held as OBJECTS (`JobManager._run` mutates them in
  place), never re-fetched by id — the job history prunes at 50 and an id
  lookup could race the prune.
* One failed unit (or model) never cancels the others; failures land in the
  meta-job result, not exceptions.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml

from .fsio import atomic_write_text

from .config import REPO_ROOT, Settings
from .schemas import Job, LoadRequest, UnloadRequest

if TYPE_CHECKING:
    from .jobs import JobManager
    from .leases import LeaseManager
    from .orchestrator import Orchestrator, Unit

COOKBOOK_PATH = REPO_ROOT / "data" / "cookbook.yaml"


def _entries(raw) -> Optional[list[dict]]:
    """One unit's target list, or None for garbage. [] is meaningful (pin empty)."""
    if raw == []:
        return []
    if not isinstance(raw, list):
        return None
    out = []
    for it in raw:
        if isinstance(it, dict) and it.get("model"):
            out.append({"server": str(it.get("server", "")), "model": str(it["model"])})
    return out if out else ([] if not raw else None)


class Cookbook:
    def __init__(self, settings: Settings, orch: "Orchestrator", jobs: "JobManager",
                 leases: "LeaseManager", path: Path | None = None):
        self.s = settings
        self.orch = orch
        self.jobs = jobs
        self.leases = leases
        self.path = path or COOKBOOK_PATH
        self._default: str = ""
        self._states: dict[str, dict] = {}
        self.load()

    # ------------------------------------------------------------------ #
    # Persistence (tolerant — the file is user-editable)
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        self._default, self._states = "", {}
        if not self.path.exists():
            return
        try:
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001 — a corrupt cookbook is an empty cookbook
            return
        for name, st in (raw.get("states") or {}).items():
            if not isinstance(st, dict):
                continue
            units = {}
            for uid, models in (st.get("units") or {}).items():
                ents = _entries(models)
                if ents is not None:
                    units[str(uid)] = ents
            if units:
                self._states[str(name)] = {
                    "saved_at": float(st.get("saved_at") or 0.0), "units": units}
        d = str(raw.get("default") or "")
        self._default = d if d in self._states else ""

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, yaml.safe_dump(
            {"default": self._default, "states": self._states},
            sort_keys=False, allow_unicode=True))

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def states(self) -> dict[str, dict]:
        return {n: {"saved_at": st["saved_at"],
                    "units": {u: [dict(e) for e in ms] for u, ms in st["units"].items()}}
                for n, st in self._states.items()}

    @property
    def default(self) -> str:
        return self._default

    def get(self, name: str) -> Optional[dict]:
        st = self._states.get(name)
        return None if st is None else {
            "saved_at": st["saved_at"],
            "units": {u: [dict(e) for e in ms] for u, ms in st["units"].items()}}

    def default_in_sync(self) -> Optional[bool]:
        """Does LaneDefaults still match the default state? None = no default.

        Starring a model later silently diverges the startup defaults from the
        marked state; we report that, we don't fight it.
        """
        if not self._default:
            return None
        st = self._states[self._default]
        for uid, targets in st["units"].items():
            persisted = self.orch.defaults.entries_or_none(uid)
            want = [(e["server"], e["model"]) for e in targets]
            have = [(e["server"], e["model"]) for e in (persisted or [])]
            if sorted(want) != sorted(have):
                return False
        return True

    # ------------------------------------------------------------------ #
    # Save-current
    # ------------------------------------------------------------------ #
    async def snapshot(self, name: str) -> dict:
        """Record every enabled unit's residency under `name`.

        Refuses mid-swap (RuntimeError) — a snapshot taken during a load would
        record a half-state that apply can never reproduce. Names are folded to
        canonical aliases so apply can actually load them (residency reports
        SERVED names; loads take aliases).
        """
        units: dict[str, list[dict]] = {}
        for unit in self.orch.units.values():
            if not unit.cfg.enabled:
                continue
            # v1: SparkGroups are outside the cookbook. A state naming a model on
            # spark1 AND a multi-node model spanning spark1 would be internally
            # contradictory, and _apply_unit runs units in parallel with no
            # cross-unit ordering — sequencing "free the members, then load the
            # group" is real follow-up work, not a snapshot detail.
            if getattr(unit, "kind", "") == "spark_group":
                continue
            if unit._lock.locked() or unit._active_job_id:
                raise RuntimeError(
                    f"unit '{unit.cfg.id}' has a swap in progress — snapshot would "
                    f"record a half-state; retry when it settles")
            st = await unit.status()
            if st.swap_in_progress:
                raise RuntimeError(
                    f"unit '{unit.cfg.id}' has a swap in progress — retry when it settles")
            targets = []
            for m in st.loaded_models:
                # A multi-node residency (m.group) is not this unit's to record:
                # the model lives in the CLUSTER catalog, so an apply would fail
                # "unknown Spark model" — and even resolving it here would write
                # a state apply cannot reproduce (groups are outside the
                # cookbook, per the exclusion above).
                if getattr(m, "group", ""):
                    continue
                targets.append({"server": m.server,
                                "model": self._canonical(unit, m.server, m.model)})
            units[unit.cfg.id] = targets      # [] recorded too: apply frees the unit
        self._states[name] = {"saved_at": round(time.time(), 1), "units": units}
        self.save()
        return self.get(name)

    def _canonical(self, unit: "Unit", server: str, model: str) -> str:
        """Fold a resident's served name onto the loadable alias."""
        if hasattr(unit, "canonical_model"):          # SparkUnit
            return unit.canonical_model(model)
        if server == "vllm":
            e = unit.registry.find_by_served_name(model)
            return e.alias if e else model
        return model                                   # Ollama tags load as-is

    # ------------------------------------------------------------------ #
    # Delete / default
    # ------------------------------------------------------------------ #
    def delete(self, name: str) -> bool:
        existed = self._states.pop(name, None) is not None
        if self._default == name:
            self._default = ""                # deleting the default clears the marker
        if existed:
            self.save()
        return existed

    def set_default(self, name: str) -> None:
        """Mark `name` AND sync LaneDefaults so the existing startup autoload
        reproduces it — including `[]` tombstones for units pinned empty."""
        st = self._states.get(name)
        if st is None:
            raise KeyError(name)
        self._default = name
        self.save()
        for uid, targets in st["units"].items():
            if not targets:
                self.orch.defaults.set_empty(uid)
                continue
            first, *rest = targets
            self.orch.defaults.set(uid, first["server"], first["model"])
            for e in rest:
                self.orch.defaults.add(uid, e["server"], e["model"])

    # ------------------------------------------------------------------ #
    # Apply
    # ------------------------------------------------------------------ #
    def apply(self, name: str) -> Job:
        """Make the fleet match `name`. Returns ONE meta-job streaming per-unit
        progress; refuses (RuntimeError) while another apply is live."""
        st = self._states.get(name)
        if st is None:
            raise KeyError(name)
        for j in self.jobs.list():
            if j.kind.startswith("cookbook:apply:") and j.state in ("pending", "running"):
                raise RuntimeError(
                    f"another apply is already running (job {j.id}, {j.kind}) — "
                    f"two applies would interleave into an arbitrary hybrid")

        targets_by_unit = {u: [dict(e) for e in ms] for u, ms in st["units"].items()}
        meta = self.jobs.create(kind=f"cookbook:apply:{name}")

        async def body(meta_job: Job) -> dict:
            result = {"loaded": [], "unloaded": [], "skipped": [],
                      "displaced_holds": [], "failed": []}
            units = [self.orch.units[uid] for uid in targets_by_unit
                     if uid in self.orch.units and self.orch.units[uid].cfg.enabled]
            missing_units = sorted(set(targets_by_unit) - {u.cfg.id for u in units})
            for uid in missing_units:
                self.jobs.log(meta_job, f"{uid}: not an enabled unit — skipped")
                result["skipped"].append({"unit": uid, "reason": "unit not enabled/known"})
            # One dead unit must not cancel the others (gather default would).
            outcomes = await asyncio.gather(
                *(self._apply_unit(meta_job, u, targets_by_unit[u.cfg.id], result)
                  for u in units),
                return_exceptions=True)
            for u, out in zip(units, outcomes):
                if isinstance(out, BaseException):
                    self.jobs.log(meta_job, f"{u.cfg.id}: FAILED — {out}")
                    result["failed"].append({"unit": u.cfg.id, "error": str(out)})
            self.jobs.log(meta_job, "apply complete")
            return result

        return self.jobs.start(meta, body)

    async def _apply_unit(self, meta: Job, unit: "Unit", targets: list[dict],
                          result: dict) -> None:
        uid = unit.cfg.id
        st = await unit.status()
        resident = {self._canonical(unit, m.server, m.model): m.server
                    for m in st.loaded_models}
        want = {e["model"]: e["server"] for e in targets}
        target_entries = {e["model"]: (unit.registry.get(e["model"])
                                       if hasattr(unit, "canonical_model") else None)
                          for e in targets}

        # A missing needs_empty_node model forces a full rebuild of this unit —
        # the runbook's load-order landmine: it may only LAUNCH on an empty node.
        needs_rebuild = any(
            (e := target_entries.get(m)) is not None and e.needs_empty_node
            for m in want if m not in resident)
        extras = [m for m in resident if m not in want]
        missing = [m for m in want if m not in resident]
        if needs_rebuild and resident:
            self.jobs.log(meta, f"{uid}: a target needs an empty node — full rebuild")
            extras = list(resident)
            missing = list(want)

        # ---- unloads first (blocked by non-preemptible leases; holds reported) ----
        for m in extras:
            blocker = self.leases.blocks_unleased(uid, m)
            if blocker is not None:
                self.jobs.log(meta, f"{uid}: keeping {m} — non-preemptible lease "
                                    f"held by '{blocker.holder}'")
                result["skipped"].append({"unit": uid, "model": m,
                                          "reason": f"lease held by {blocker.holder}"})
                continue
            hold = self.leases.active_for(uid, m)
            if hold is not None:
                result["displaced_holds"].append(
                    {"unit": uid, "model": m, "holder": hold.holder})
            self.jobs.log(meta, f"{uid}: unloading {m}…")
            await unit.unload(UnloadRequest(server=None, lane=uid, model=m))
            result["unloaded"].append({"unit": uid, "model": m})

        # ---- loads: needs_empty_node first, then biggest budget first ----
        def order(m: str) -> tuple:
            e = target_entries.get(m)
            return (not (e is not None and e.needs_empty_node),
                    -(e.mem_fraction if e is not None else 0.0))

        for m in sorted(missing, key=order):
            self.jobs.log(meta, f"{uid}: loading {m} ({want[m]})…")
            child = unit.load(LoadRequest(server=want[m], model=m, lane=uid))
            emitted = 0
            while child.state in ("pending", "running"):
                while emitted < len(child.log):
                    self.jobs.log(meta, f"{uid}: {child.log[emitted]}")
                    emitted += 1
                await asyncio.sleep(1.0)
            while emitted < len(child.log):
                self.jobs.log(meta, f"{uid}: {child.log[emitted]}")
                emitted += 1
            if child.state == "succeeded":
                result["loaded"].append({"unit": uid, "model": m})
            else:
                err = child.error or "load failed"
                self.jobs.log(meta, f"{uid}: {m} FAILED — {err}")
                result["failed"].append({"unit": uid, "model": m, "error": err})
