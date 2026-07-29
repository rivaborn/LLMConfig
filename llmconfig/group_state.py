"""Multi-node (SparkGroup) shared state: live claims + persisted placements.

Two small classes with very different lifetimes, kept together because every
consumer of one needs the other:

- `GroupState` — the LIVE claims: which node set is currently occupied by which
  multi-node model. In-memory only (residency is re-discovered by probing after a
  restart, same philosophy as Spark slot ports). Every method is **synchronous
  and await-free on purpose** — placement's `_facts_for` and the member units'
  `status()`/`_admit` read it on paths that must not yield (the same contract
  `LeaseManager` keeps, invariant 11).

- `GroupPlacements` — the MEMORY: every (model, node-set) that has ever loaded
  successfully, persisted to `data/spark_group_state.yaml`. This is the
  requirement "once a model has been loaded on a node set, it is listed on those
  cards and available to auto-placement": the orchestrator re-instantiates a
  `SparkGroup` per recorded set at startup (fabric flag permitting), so the
  placer has standing candidates without anyone re-launching by hand first.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import group_id_for
from .fsio import atomic_write_text


@dataclass(frozen=True)
class GroupClaim:
    """One live multi-node deployment: `model` spans `member_ids`, served from the
    head node on `head_port`."""

    group_id: str
    model: str                  # the served name (what the head's /v1/models reports)
    alias: str                  # the cluster-catalog alias behind it
    member_ids: tuple[str, ...]
    head_port: int
    since: float = field(default_factory=time.time)


class GroupState:
    """Live claims, keyed by group id, with a member-id reverse index.

    A member can be in at most ONE live claim (a tp job owns the whole node), so
    `claim()` refuses an overlap outright — the SparkGroup load path checks
    availability under the member locks before claiming, so a refusal here means
    a programming error upstream, not a race to be retried.
    """

    def __init__(self) -> None:
        self._claims: dict[str, GroupClaim] = {}
        self._by_member: dict[str, str] = {}   # member id -> group id

    # ---- reads (sync, await-free — see module docstring) ----
    def get(self, group_id: str) -> GroupClaim | None:
        return self._claims.get(group_id)

    def claim_for(self, member_id: str) -> GroupClaim | None:
        """The live claim covering one member node, or None."""
        gid = self._by_member.get(member_id)
        return self._claims.get(gid) if gid else None

    def all(self) -> list[GroupClaim]:
        return list(self._claims.values())

    # ---- writes ----
    def claim(self, group_id: str, model: str, alias: str,
              member_ids: tuple[str, ...], head_port: int) -> GroupClaim:
        taken = [m for m in member_ids if m in self._by_member
                 and self._by_member[m] != group_id]
        if taken:
            raise RuntimeError(
                f"member(s) {taken} already claimed by "
                f"{sorted({self._by_member[m] for m in taken})} — "
                "a node can belong to one live multi-node deployment at a time"
            )
        c = GroupClaim(group_id=group_id, model=model, alias=alias,
                       member_ids=tuple(member_ids), head_port=head_port)
        self._claims[group_id] = c
        for m in member_ids:
            self._by_member[m] = group_id
        return c

    def release(self, group_id: str) -> bool:
        c = self._claims.pop(group_id, None)
        if c is None:
            return False
        for m in c.member_ids:
            if self._by_member.get(m) == group_id:
                self._by_member.pop(m, None)
        return True


class GroupPlacements:
    """Persisted (model, node-set) history — `data/spark_group_state.yaml`.

    Stored shape (user-editable; reading must never raise):

    ```yaml
    placements:
      deepseek-v4-flash:
        - {members: [spark1, spark2], last_loaded: 1785000000.0, loads: 3}
        - {members: [spark1, spark2, spark3, spark4], last_loaded: ..., loads: 1}
    ```
    """

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, list[dict]] = {}
        self.load()

    def load(self) -> None:
        self._data = {}
        if not self.path.exists():
            return
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        for alias, rows in (raw.get("placements") or {}).items():
            out: list[dict] = []
            for r in rows or []:
                members = sorted(str(m) for m in (r.get("members") or []) if m)
                if len(members) < 2:
                    continue        # a 1-node "placement" is not a group
                out.append({
                    "members": members,
                    "last_loaded": float(r.get("last_loaded") or 0.0),
                    "loads": int(r.get("loads") or 0),
                })
            if out:
                self._data[str(alias)] = out

    def save(self) -> None:
        # Atomic like every other data/ writer (fsio): a power cut mid-write must
        # not tear the file — the tolerant loader would read that as "no
        # placements", silently dropping every recorded node set.
        atomic_write_text(
            self.path,
            yaml.safe_dump({"placements": self._data}, sort_keys=False, allow_unicode=True),
        )

    # ---- reads ----
    def for_alias(self, alias: str) -> list[dict]:
        return [dict(r) for r in self._data.get(alias, [])]

    def all(self) -> dict[str, list[dict]]:
        return {k: [dict(r) for r in v] for k, v in self._data.items()}

    def node_sets(self) -> list[tuple[str, ...]]:
        """Every DISTINCT recorded node set, across all models — what the
        orchestrator instantiates SparkGroups for at startup."""
        seen: dict[tuple[str, ...], None] = {}
        for rows in self._data.values():
            for r in rows:
                seen[tuple(r["members"])] = None
        return list(seen)

    def sets_for(self, alias: str) -> list[tuple[str, ...]]:
        """Recorded node sets for one model, most recently used first — the
        cold-start order auto-placement tries."""
        rows = sorted(self._data.get(alias, []), key=lambda r: -r["last_loaded"])
        return [tuple(r["members"]) for r in rows]

    # ---- writes ----
    def record(self, alias: str, member_ids: tuple[str, ...] | list[str]) -> None:
        """Upsert one successful load. Idempotent per (alias, set); bumps the
        counter and timestamp on repeats."""
        members = sorted(member_ids)
        rows = self._data.setdefault(alias, [])
        for r in rows:
            if r["members"] == members:
                r["last_loaded"] = time.time()
                r["loads"] += 1
                break
        else:
            rows.append({"members": members, "last_loaded": time.time(), "loads": 1})
        self.save()

    def group_ids(self) -> dict[str, tuple[str, ...]]:
        """group id -> node set, for every recorded set."""
        return {group_id_for(s): s for s in self.node_sets()}
