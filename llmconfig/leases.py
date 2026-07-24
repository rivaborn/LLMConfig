"""Leases — resource sharing between callers competing for one LLM unit.

Before this, the only contention signal was *"is a swap running right now?"*
(`swap_in_progress`) plus a decaying 60 s activity heuristic (`idle_s`/`usage`).
Neither is a **claim**: the model was last-writer-wins, so caller B could load a
different model out from under caller A mid-session and A got no signal at all.

A **lease** is one caller's claim on one unit, carrying three things the old
signals couldn't express:

* **whether the work may be interrupted** (`preemptible`),
* **how long it's needed** — deliberately two fields: `expected_duration_s` is the
  honest hint (displayed, never enforced), `ttl_s` a short renewable leash that
  proves the holder is still alive. A crashed client frees the unit in minutes
  instead of pinning it for its whole declared duration.
* **why it ended** — a displaced holder polls `GET /api/leases/{id}` and sees
  `state=revoked` with `revoked_by`/`revoked_reason`, and its next `/v1` request
  gets an in-band 409. That is how "your job was kicked off a unit" is delivered.

## The load-bearing invariant of this module

> Every query and mutation here — `get` / `list` / `active_for` / `brief` /
> `blocks_unleased` / `claim` / `renew` / `release` / `revoke` / `sweep` — is
> `def`, **not** `async def`, and must never await, sleep, or do I/O.

`idle.py:_check_lane` relies on there being **no await** between its final guard
check and `Unit.unload()`: an uncontended `asyncio.Lock` acquires without
yielding, so that sequence is atomic on a single-threaded loop. The lease guard
is fused into that check, so if any of these methods ever became async, a
competing load could interleave and get its freshly loaded model unloaded out
from under it. `tests/test_leases.py::test_query_methods_are_sync` pins this.

## Storage and clocks

In-memory and bounded, mirroring `JobManager` — deliberately **not persisted**.
A lease is a claim on *live* GPU residency; after a restart nothing it protected
is resident and the holder process is gone, so a surviving lease would block the
idle reaper on behalf of a ghost. Restart is handled explicitly instead: an
unknown-but-plausible lease id near startup reports `server_restarted`.

Wall clock (`time.time()`) is serialized for display; **`time.monotonic()` is
authoritative for liveness** (same reasoning as `Lane._wait_vram_free`), so an
NTP step can't mass-expire or mass-extend every lease at once.

## Two asymmetries that look like bugs but aren't

* **Expiry never unloads; preemption *may*.** A lease is permission not to be
  disturbed, not a VRAM reservation — on expiry the model stays resident and the
  idle reaper simply resumes its normal timer. Preemption only frees the unit
  when the displaced lease asked for it (`free_on_preempt`, default off).
* **Activity does not auto-renew.** `lane.touch()` fires for un-leased traffic
  and Monitor util spikes too, so renewing off it would let a stranger's traffic
  extend your hold — and would make a runaway holder immortal, which is exactly
  what the TTL exists to bound. Renewal is explicit; `last_seen_at`/`requests`
  are recorded for observability only.

Leases are **advisory**: clients that bypass LLMConfig (direct Ollama on
:11434/:11435, the vLLM relay on :11437) are ungated, so a non-preemptible lease
is a cooperation contract, not a hard exclusivity guarantee.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .config import Settings
from .schemas import (
    MANAGED_OWNERS,
    Lease,
    LeaseBrief,
    LeaseClaimRequest,
    LeaseState,
    UnloadRequest,
)

if TYPE_CHECKING:
    from .orchestrator import Orchestrator, Unit

log = logging.getLogger(__name__)


class LeaseError(Exception):
    """Base for lease failures; `code` is the machine-readable API discriminator."""

    code = "lease_error"

    def __init__(self, message: str, lease: Optional[Lease] = None):
        super().__init__(message)
        self.message = message
        self.lease = lease


class LeaseConflict(LeaseError):
    """The unit is already claimed and this claim may not take it (→ 409)."""

    code = "lease_held"


class LeaseNotActive(LeaseError):
    """Renewing / acting on a lease that has already ended (→ 409)."""

    code = "lease_not_active"


class UnknownUnit(LeaseError):
    """No such unit id (→ 400, mirroring main._lane())."""

    code = "unknown_unit"


@dataclass
class PendingFree:
    """A queued request to actually evict a unit after its lease was preempted.

    `claim()` is a fast synchronous REST call but `Unit.unload()` is async and can
    block for minutes behind an in-flight swap, so the two are split: revoke now,
    free later on a `LeaseSweeper` tick.
    """

    # The lease that ASKED for the empty card (a claimant with free_on_preempt), or
    # "" when an operator asked via `revoke --free`. The sweeper only honours the
    # request while that same lease still holds the unit.
    lease_id: str
    requested_at: float
    reason: str


class LeaseManager:
    def __init__(self, settings: Settings, orch: "Orchestrator"):
        self.s = settings
        self.orch = orch
        self._leases: dict[str, Lease] = {}
        self._pending_free: dict[str, PendingFree] = {}
        self.started_at = time.time()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _clamp_ttl(self, ttl: Optional[float]) -> tuple[float, str]:
        """Clamp rather than refuse — a caller asking for too long should still get
        a lease, just a shorter one, with the adjustment recorded in `note`."""
        want = float(ttl if ttl is not None else self.s.lease_default_ttl_s)
        lo, hi = float(self.s.lease_min_ttl_s), float(self.s.lease_max_ttl_s)
        ttl_s = max(lo, min(hi, want))
        note = "" if ttl_s == want else f"ttl clamped {want:.0f}s → {ttl_s:.0f}s"
        return ttl_s, note

    def _expire_if_lapsed(self, lease: Lease, now_mono: float) -> None:
        """Lazy expiry — a plain dict mutation, never an await.

        Because this runs on every read path, expiry stays correct even when the
        sweeper is disabled or lagging; the sweeper only *reacts* to it.
        """
        if lease.state == "active" and now_mono >= lease._deadline:
            self._terminate(lease, "expired", reason="expired")

    def _terminate(self, lease: Lease, state: LeaseState, *, by: str = "",
                   reason: str = "", displaced_by: str = "") -> None:
        """Single funnel for every end-of-life transition.

        No-op when the lease is already terminal, so the expiry-vs-preemption race
        is benign: first writer wins and the recorded reason is the first observed.
        """
        if lease.state != "active":
            return
        now = time.time()
        lease.state = state
        if state == "released":
            lease.released_at = now
        else:
            lease.revoked_at = now
            lease.revoked_by = by
            lease.revoked_reason = reason
            lease.revoked_lease_id = displaced_by

    def _prune(self) -> None:
        """Bounded history — drop only terminal leases, oldest first (JobManager's rule)."""
        cap = max(1, int(self.s.lease_max_history))
        if len(self._leases) <= cap:
            return
        terminal = sorted(
            (l for l in self._leases.values() if l.state != "active"),
            key=lambda l: l.acquired_at,
        )
        excess = len(self._leases) - cap
        for l in terminal[:excess]:
            self._leases.pop(l.id, None)

    # ------------------------------------------------------------------ #
    # Queries (sync, no I/O — see the module docstring)
    # ------------------------------------------------------------------ #
    def get(self, lease_id: str) -> Optional[Lease]:
        lease = self._leases.get(lease_id)
        if lease is not None:
            self._expire_if_lapsed(lease, time.monotonic())
        return lease

    def list(self, unit: str | None = None, active_only: bool = False) -> list[Lease]:
        now_mono = time.monotonic()
        out = []
        for lease in self._leases.values():
            self._expire_if_lapsed(lease, now_mono)
            if unit is not None and lease.unit != unit:
                continue
            if active_only and lease.state != "active":
                continue
            out.append(lease)
        return sorted(out, key=lambda l: l.acquired_at, reverse=True)

    def active_for(self, unit_id: str) -> Optional[Lease]:
        """The live lease on a unit, or None. Runs lazy expiry first.

        THIS MUST STAY SYNCHRONOUS — it is called from `idle.py`'s final
        no-await-before-unload guard.
        """
        now_mono = time.monotonic()
        for lease in self._leases.values():
            if lease.unit != unit_id:
                continue
            self._expire_if_lapsed(lease, now_mono)
            if lease.state == "active":
                return lease
        return None

    def brief(self, unit_id: str) -> Optional[LeaseBrief]:
        lease = self.active_for(unit_id)
        if lease is None:
            return None
        return LeaseBrief(
            id=lease.id,
            holder=lease.holder,
            preemptible=lease.preemptible,
            priority=lease.priority,
            expires_at=lease.expires_at,
            expires_in_s=round(max(0.0, lease._deadline - time.monotonic()), 1),
            model=lease.model,
        )

    def blocks_unleased(self, unit_id: str) -> Optional[Lease]:
        """The live lease that should refuse un-leased traffic, if any."""
        if not self.s.lease_block_unleased:
            return None
        lease = self.active_for(unit_id)
        return lease if lease is not None and not lease.preemptible else None

    def blocks_idle_unload(self, unit_id: str) -> Optional[Lease]:
        """The live lease that should stop the idle reaper, if any.

        ANY live lease blocks — not just non-preemptible ones. The reaper is a
        power-saving optimisation, not a competing claimant, and "I declared 45
        minutes and I'm 16 minutes between two bursts" is exactly the case a lease
        exists to serve. The bound is the TTL, so the idle timeout is skipped for
        at most one lease period.
        """
        if not self.s.lease_blocks_idle_unload:
            return None
        return self.active_for(unit_id)

    def pending_free(self) -> dict[str, PendingFree]:
        return dict(self._pending_free)

    def clear_pending(self, unit_id: str) -> None:
        self._pending_free.pop(unit_id, None)

    # ------------------------------------------------------------------ #
    # Mutations (sync)
    # ------------------------------------------------------------------ #
    def claim(self, req: LeaseClaimRequest) -> tuple[Lease, Optional[Lease]]:
        """Acquire a unit. Returns (lease, displaced_or_None); raises on refusal.

        Decision table — `E` is the unit's live lease:

        | E                    | condition                       | outcome              |
        | -------------------- | ------------------------------- | -------------------- |
        | none                 | —                               | grant                |
        | same holder          | —                               | extend in place      |
        | not preemptible      | any (force included)            | refuse               |
        | preemptible          | new priority > held             | grant + preempt      |
        | preemptible          | priority <= held, no force      | refuse               |
        | preemptible          | priority <= held, force         | grant + preempt      |

        `force` deliberately does NOT override a non-preemptible lease — if it did,
        "must not be interrupted" would mean nothing. Break-glass is the explicit
        `POST /api/leases/{id}/revoke`.
        """
        if req.unit not in self.orch.units:
            raise UnknownUnit(
                f"unknown unit '{req.unit}' (have: {', '.join(self.orch.units)})"
            )
        holder = (req.holder or "").strip()
        if not holder:
            raise LeaseError("holder is required")

        ttl_s, clamp_note = self._clamp_ttl(req.ttl_s)
        priority = max(0, min(100, int(req.priority)))
        now, now_mono = time.time(), time.monotonic()
        existing = self.active_for(req.unit)

        # Same holder re-claiming: extend in place rather than fragmenting into two
        # leases (a retrying client must not end up holding one and being blocked
        # by the other).
        if existing is not None and existing.holder == holder:
            existing.ttl_s = ttl_s
            existing.expected_duration_s = req.expected_duration_s or existing.expected_duration_s
            existing.preemptible = req.preemptible
            existing.priority = priority
            existing.free_on_preempt = req.free_on_preempt
            if req.model:
                existing.model = req.model
            if req.server:
                existing.server = req.server
            existing.note = clamp_note or req.note or existing.note
            existing.renewed_at = now
            existing.renew_count += 1
            existing.expires_at = now + ttl_s
            existing._deadline = now_mono + ttl_s
            return existing, None

        displaced: Optional[Lease] = None
        if existing is not None:
            if not existing.preemptible:
                raise LeaseConflict(
                    f"unit '{req.unit}' is held by '{existing.holder}' (non-preemptible) "
                    f"for another {max(0.0, existing._deadline - now_mono):.0f}s",
                    lease=existing,
                )
            if priority <= existing.priority and not req.force:
                raise LeaseConflict(
                    f"unit '{req.unit}' is held by '{existing.holder}' at priority "
                    f"{existing.priority}; claim at a higher priority or pass force=true",
                    lease=existing,
                )
            displaced = existing  # preempted below, once the new lease has an id

        lease = Lease(
            id=uuid.uuid4().hex[:12],
            unit=req.unit,
            holder=holder,
            preemptible=req.preemptible,
            priority=priority,
            ttl_s=ttl_s,
            expected_duration_s=req.expected_duration_s,
            acquired_at=now,
            expires_at=now + ttl_s,
            server=req.server,
            model=req.model,
            note=clamp_note or req.note,
            free_on_preempt=req.free_on_preempt,
        )
        lease._deadline = now_mono + ttl_s

        if displaced is not None:
            self._terminate(displaced, "revoked", by=holder, reason="preempted",
                            displaced_by=lease.id)
            log.info("lease %s (%s) preempted %s (%s) on %s",
                     lease.id, holder, displaced.id, displaced.holder, req.unit)

        # `free_on_preempt` is the CLAIMANT's request for an empty card — "give me
        # this unit with nothing loaded on it" (an external CUDA job, Wait-GpuIdle).
        # Off by default because `Lane.load` already evicts and waits for the VRAM
        # baseline itself, so freeing first is usually a wasted drain/refill plus a
        # window in which a third party can grab the empty card.
        if req.free_on_preempt:
            self._pending_free[req.unit] = PendingFree(lease.id, now, "claimed")

        self._leases[lease.id] = lease
        self._prune()
        return lease, displaced

    def renew(self, lease_id: str, ttl_s: Optional[float] = None) -> Lease:
        """Push the deadline out. Revives an `expired` lease inside the grace window."""
        lease = self._leases.get(lease_id)
        if lease is None:
            raise UnknownUnit(f"unknown lease '{lease_id}'")
        now_mono = time.monotonic()
        self._expire_if_lapsed(lease, now_mono)  # a lapsed lease renews only via grace
        if lease.state == "active" or (
            lease.state == "expired" and lease.in_grace(now_mono, self.s.lease_expiry_grace_s)
        ):
            new_ttl, clamp_note = self._clamp_ttl(ttl_s if ttl_s is not None else lease.ttl_s)
            lease.state = "active"          # revive from the grace window
            lease.revoked_at = None
            lease.revoked_reason = ""
            lease.ttl_s = new_ttl
            lease.renewed_at = time.time()
            lease.renew_count += 1
            lease.expires_at = lease.renewed_at + new_ttl
            lease._deadline = now_mono + new_ttl
            if clamp_note:
                lease.note = clamp_note
            return lease
        raise LeaseNotActive(
            f"lease {lease_id} is {lease.state}"
            + (f" (revoked by '{lease.revoked_by}': {lease.revoked_reason})"
               if lease.revoked_by else ""),
            lease=lease,
        )

    def release(self, lease_id: str) -> Lease:
        """Holder hands the unit back. Idempotent — a terminal lease returns unchanged."""
        lease = self._leases.get(lease_id)
        if lease is None:
            raise UnknownUnit(f"unknown lease '{lease_id}'")
        self._terminate(lease, "released")
        return lease

    def revoke(self, lease_id: str, *, by: str = "operator", reason: str = "admin",
               free: bool = False) -> Lease:
        """Operator break-glass — takes a unit back even from a non-preemptible lease."""
        lease = self._leases.get(lease_id)
        if lease is None:
            raise UnknownUnit(f"unknown lease '{lease_id}'")
        self._terminate(lease, "revoked", by=by, reason=reason)
        if free:
            # No requesting lease — an operator asked, so honour it only while the
            # unit stays unclaimed (see LeaseSweeper._free_unit).
            self._pending_free[lease.unit] = PendingFree("", time.time(), reason)
        return lease

    def note_request(self, lease_id: str) -> None:
        """Record that the holder actually used its lease (observability + the
        `unused` dead-holder detector). Deliberately does NOT extend the deadline."""
        lease = self._leases.get(lease_id)
        if lease is None:
            return
        lease.requests += 1
        lease.last_seen_at = time.time()

    def expire_unused(self, lease_id: str) -> Optional[Lease]:
        lease = self._leases.get(lease_id)
        if lease is None:
            return None
        self._terminate(lease, "expired", reason="unused")
        return lease

    def sweep(self) -> list[Lease]:
        """Flip any lapsed leases and report the ones that changed on this pass."""
        now_mono = time.monotonic()
        newly: list[Lease] = []
        for lease in self._leases.values():
            if lease.state == "active" and now_mono >= lease._deadline:
                self._terminate(lease, "expired", reason="expired")
                newly.append(lease)
        return newly

    # ---- restart disambiguation ----
    def looks_like_pre_restart(self, lease_id: str) -> bool:
        """True for a well-formed id we've never seen, close enough to startup that
        a restart is the likely explanation — lets the gateway say `server_restarted`
        instead of a bare "unknown lease"."""
        if lease_id in self._leases:
            return False
        if len(lease_id) != 12 or not all(c in "0123456789abcdef" for c in lease_id):
            return False
        return (time.time() - self.started_at) < float(self.s.lease_max_ttl_s)


class LeaseSweeper:
    """Background expiry + deferred preemption frees.

    A separate task from `IdleReaper` on purpose: the reaper can be switched off
    entirely (`IDLE_UNLOAD_ENABLED=false`) while lease expiry must still work, and
    this wants a ~5 s cadence against the reaper's 60 s. Because expiry is lazy
    (see `LeaseManager._expire_if_lapsed`), disabling this sweeper degrades logging
    and deferred frees — never correctness.
    """

    def __init__(self, settings: Settings, orch: "Orchestrator", leases: LeaseManager):
        self.s = settings
        self.orch = orch
        self.leases = leases
        self.interval = max(1.0, float(settings.lease_sweep_interval_s))
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    # ---- lifecycle (mirrors IdleReaper / Monitor) ----
    def start(self) -> None:
        if not self.s.lease_enabled or self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="lease-sweeper")

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
            try:
                await self._tick()
            except Exception as e:  # noqa: BLE001 — a tick hiccup must never kill the loop
                log.warning("lease sweeper tick failed: %s: %s", type(e).__name__, e)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                pass

    # ---- one pass ----
    async def _tick(self) -> None:
        for lease in self.leases.sweep():
            log.info("lease %s (%s) expired on %s", lease.id, lease.holder, lease.unit)

        await self._reap_unused()

        pending = self.leases.pending_free()
        if not pending:
            return
        for unit in list(self.orch.units.values()):
            pf = pending.get(unit.cfg.id)
            if pf is None:
                continue
            try:
                await self._free_unit(unit, pf)
            except Exception as e:  # noqa: BLE001 — one unit can't starve the others
                log.warning("lease sweeper: %s free failed: %s: %s",
                            unit.cfg.id, type(e).__name__, e)

    async def _reap_unused(self) -> None:
        """Drop a lease whose holder never used it and whose unit holds nothing.

        Covers the dead-holder case: claim → the load fails → a non-preemptible
        lease would otherwise keep refusing traffic on a card holding nothing for
        its whole TTL.
        """
        window = float(self.s.lease_unused_release_s)
        if window <= 0:
            return
        now = time.time()
        for lease in self.leases.list(active_only=True):
            if lease.requests or (now - lease.acquired_at) < window:
                continue
            unit = self.orch.units.get(lease.unit)
            if unit is None:
                continue
            try:
                st = await unit.status()
            except Exception:  # noqa: BLE001 — a probe failure is not evidence of disuse
                continue
            if st.owner in MANAGED_OWNERS or st.swap_in_progress:
                continue
            self.leases.expire_unused(lease.id)
            log.info("lease %s (%s) released — never used and %s is free",
                     lease.id, lease.holder, lease.unit)

    def _still_wanted(self, unit_id: str, pf: PendingFree) -> bool:
        """Is this queued free still the right thing to do? (sync — see _free_unit)

        A claimant's request is honoured only while that claimant still holds the
        unit; an operator's (`lease_id == ""`) only while nobody else has claimed it.
        Either way we must never yank a model out from under a *different* holder.
        """
        active = self.leases.active_for(unit_id)
        if pf.lease_id:
            return active is not None and active.id == pf.lease_id
        return active is None

    async def _free_unit(self, unit: "Unit", pf: PendingFree) -> None:
        """Evict a preempted unit, mirroring IdleReaper._check_lane's guard ladder."""
        # A queued load is invisible here (`_active_job_id` is only set once the lock
        # is taken), so a free can land just before a queued load reloads — a wasted
        # drain/refill, not a correctness bug: the load's own eviction gate handles it.
        # It is the second reason `free_on_preempt` defaults off.
        if unit._lock.locked() or unit._active_job_id:
            return                                        # swap in flight — retry next tick
        if not self._still_wanted(unit.cfg.id, pf):
            self.leases.clear_pending(unit.cfg.id)
            return
        st = await unit.status()                          # the ONLY await in this ladder
        if st.swap_in_progress or st.owner not in MANAGED_OWNERS:
            self.leases.clear_pending(unit.cfg.id)        # nothing loaded to free
            return
        # Final sync re-check. No await between here and unload(): an uncontended
        # asyncio.Lock acquires without yielding and active_for() is plain dict
        # access, so neither a competing load nor a claim landing during the status
        # probe above can interleave.
        if unit._lock.locked() or not self._still_wanted(unit.cfg.id, pf):
            return
        self.leases.clear_pending(unit.cfg.id)
        log.info("lease sweeper: freeing %s after %s (lease %s)",
                 unit.cfg.id, pf.reason, pf.lease_id)
        await unit.unload(UnloadRequest(server=None, lane=unit.cfg.id))  # invariant 3
        unit.touch()  # restart the idle window, as the reaper does after a reap
