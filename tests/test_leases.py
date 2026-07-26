"""Lease manager + sweeper.

Pure in-memory: a minimal fake unit/orchestrator stands in for `Lane`/`SparkUnit`,
so nothing here touches nvidia-smi, wsl.exe, HTTP or a real event-loop lifecycle.
"""
import asyncio
import inspect
import time
from types import SimpleNamespace

import pytest

from llmconfig.config import Settings
from llmconfig.leases import (
    LeaseConflict,
    LeaseManager,
    LeaseNotActive,
    LeaseSweeper,
    UnknownUnit,
)
from llmconfig.schemas import LaneStatus, LeaseClaimRequest, LoadedModel, UnloadRequest


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeUnit:
    """Just enough of the duck-typed unit contract for the sweeper."""

    def __init__(self, uid: str, owner: str = "ollama"):
        self.cfg = SimpleNamespace(id=uid, gpu_uuid=f"GPU-{uid}", enabled=True,
                                   idle_unload_enabled=True)
        self._lock = asyncio.Lock()
        self._active_job_id = None
        self.owner = owner
        self.models: list[str] = []      # resident models, for the multi-model paths
        self.unloads: list[UnloadRequest] = []
        self.touched = 0
        self.status_hook = None          # optional async callback fired inside status()

    async def status(self, gpu=None) -> LaneStatus:
        if self.status_hook is not None:
            await self.status_hook()
        return LaneStatus(
            id=self.cfg.id, name=self.cfg.id, owner=self.owner,
            ollama_up=True, vllm_up=False,
            loaded_models=[LoadedModel(server="ollama", model=m) for m in self.models],
        )

    async def unload(self, req: UnloadRequest) -> LaneStatus:
        self.unloads.append(req)
        if req.model:
            self.models = [m for m in self.models if m != req.model]
        else:
            self.models = []
            self.owner = "free"
        return await self.status()

    def touch(self, ts=None, model=None) -> None:
        # `model` is part of the duck-typed unit contract now that a unit can hold
        # several models — see SparkUnit.touch.
        self.touched += 1


def _mgr(**overrides):
    settings = Settings(_env_file=None, **overrides)
    units = {"primary": FakeUnit("primary"), "companion": FakeUnit("companion")}
    orch = SimpleNamespace(units=units)
    return LeaseManager(settings, orch), orch, settings


def _claim(mgr, holder="alice", unit="primary", **kw):
    return mgr.claim(LeaseClaimRequest(unit=unit, holder=holder, **kw))


def _backdate(lease, seconds=1.0):
    """Force a lease past its deadline without sleeping."""
    lease._deadline = time.monotonic() - seconds


# --------------------------------------------------------------------------- #
# The atomicity invariant
# --------------------------------------------------------------------------- #
def test_query_methods_are_sync():
    """idle.py's final guard runs these with no await before unload(). If any of
    them became a coroutine, a competing load could interleave and lose its model."""
    for name in ("get", "list", "active_for", "brief", "blocks_unleased",
                 "blocks_idle_unload", "claim", "renew", "release", "revoke", "sweep"):
        fn = getattr(LeaseManager, name)
        assert not inspect.iscoroutinefunction(fn), f"LeaseManager.{name} must stay sync"


# --------------------------------------------------------------------------- #
# claim() decision table
# --------------------------------------------------------------------------- #
def test_claim_on_free_unit_grants():
    mgr, _, _ = _mgr()
    lease, displaced = _claim(mgr)
    assert lease.state == "active" and lease.holder == "alice" and displaced is None
    assert mgr.active_for("primary") is lease


def test_claim_unknown_unit_rejected():
    mgr, _, _ = _mgr()
    with pytest.raises(UnknownUnit):
        _claim(mgr, unit="nope")


def test_same_holder_extends_in_place():
    mgr, _, _ = _mgr()
    first, _ = _claim(mgr, ttl_s=100)
    again, displaced = _claim(mgr, ttl_s=300)
    assert again.id == first.id and displaced is None
    assert again.renew_count == 1 and again.ttl_s == 300
    assert len(mgr.list(unit="primary")) == 1, "a retry must not fragment into two leases"


def test_non_preemptible_refuses_even_with_force():
    mgr, _, _ = _mgr()
    _claim(mgr, holder="alice", preemptible=False)
    for force in (False, True):
        with pytest.raises(LeaseConflict):
            _claim(mgr, holder="bob", force=force, priority=99)
    assert mgr.active_for("primary").holder == "alice"


def test_higher_priority_preempts():
    mgr, _, _ = _mgr()
    first, _ = _claim(mgr, holder="alice", priority=1)
    second, displaced = _claim(mgr, holder="bob", priority=5)
    assert displaced is first
    assert first.state == "revoked" and first.revoked_by == "bob"
    assert first.revoked_reason == "preempted" and first.revoked_lease_id == second.id
    assert mgr.active_for("primary") is second


def test_equal_priority_refused_without_force_and_granted_with():
    mgr, _, _ = _mgr()
    _claim(mgr, holder="alice", priority=3)
    with pytest.raises(LeaseConflict):
        _claim(mgr, holder="bob", priority=3)
    second, displaced = _claim(mgr, holder="bob", priority=3, force=True)
    assert displaced.holder == "alice" and mgr.active_for("primary") is second


def test_ttl_and_priority_are_clamped_not_refused():
    mgr, _, s = _mgr(lease_max_ttl_s=100, lease_min_ttl_s=10)
    hi, _ = _claim(mgr, ttl_s=99999, priority=999)
    assert hi.ttl_s == 100 and hi.priority == 100 and "clamped" in hi.note
    mgr.release(hi.id)
    lo, _ = _claim(mgr, holder="bob", ttl_s=1)
    assert lo.ttl_s == 10


def test_lanes_are_independent():
    mgr, _, _ = _mgr()
    _claim(mgr, holder="alice", unit="primary", preemptible=False)
    other, _ = _claim(mgr, holder="bob", unit="companion")
    assert other.state == "active"


# --------------------------------------------------------------------------- #
# Expiry / grace / lifecycle
# --------------------------------------------------------------------------- #
def test_lazy_expiry_without_the_sweeper():
    """Expiry must be correct even if the sweeper is off or lagging."""
    mgr, _, _ = _mgr()
    lease, _ = _claim(mgr)
    _backdate(lease)
    assert mgr.active_for("primary") is None
    assert lease.state == "expired" and lease.revoked_reason == "expired"


def test_get_and_list_also_expire_lazily():
    mgr, _, _ = _mgr()
    lease, _ = _claim(mgr)
    _backdate(lease)
    assert mgr.get(lease.id).state == "expired"   # a bare poll must not read stale "active"


def test_expired_lease_frees_the_unit_for_a_new_claim():
    mgr, _, _ = _mgr()
    first, _ = _claim(mgr, holder="alice", preemptible=False)
    _backdate(first)
    second, displaced = _claim(mgr, holder="bob")
    assert second.state == "active" and displaced is None


def test_grace_allows_renew_then_hard_expires():
    mgr, _, _ = _mgr(lease_expiry_grace_s=60)
    lease, _ = _claim(mgr)
    _backdate(lease, seconds=1)
    assert mgr.active_for("primary") is None      # no longer blocks anyone
    revived = mgr.renew(lease.id)                 # ...but its own holder can revive it
    assert revived.state == "active" and mgr.active_for("primary") is revived
    _backdate(revived, seconds=999)               # now well past the grace window
    with pytest.raises(LeaseNotActive):
        mgr.renew(revived.id)


def test_release_is_idempotent():
    mgr, _, _ = _mgr()
    lease, _ = _claim(mgr)
    assert mgr.release(lease.id).state == "released"
    assert mgr.release(lease.id).state == "released"


def test_terminate_is_first_writer_wins():
    mgr, _, _ = _mgr()
    lease, _ = _claim(mgr)
    _backdate(lease)
    mgr.sweep()                                    # → expired
    mgr.revoke(lease.id, by="op", reason="admin")  # must not overwrite
    assert lease.state == "expired" and lease.revoked_reason == "expired"


def test_revoke_takes_a_non_preemptible_lease():
    """The break-glass path `force` deliberately does not provide."""
    mgr, _, _ = _mgr()
    lease, _ = _claim(mgr, preemptible=False)
    mgr.revoke(lease.id, by="operator", reason="admin")
    assert lease.state == "revoked"
    assert _claim(mgr, holder="bob")[0].state == "active"


def test_note_request_records_but_does_not_extend():
    mgr, _, _ = _mgr()
    lease, _ = _claim(mgr)
    before = lease._deadline
    mgr.note_request(lease.id)
    assert lease.requests == 1 and lease.last_seen_at is not None
    assert lease._deadline == before, "activity must not auto-renew"


def test_history_is_pruned_but_active_leases_survive():
    mgr, _, _ = _mgr(lease_max_history=3)
    for i in range(6):
        lease, _ = _claim(mgr, holder=f"h{i}", unit="primary")
        mgr.release(lease.id)
    live, _ = _claim(mgr, holder="live")
    assert len(mgr._leases) <= 3
    assert mgr.active_for("primary") is live


def test_blocks_unleased_only_for_non_preemptible():
    mgr, _, _ = _mgr()
    soft, _ = _claim(mgr, holder="alice")
    assert mgr.blocks_unleased("primary") is None
    mgr.release(soft.id)
    hard, _ = _claim(mgr, holder="bob", preemptible=False)
    assert mgr.blocks_unleased("primary") is hard


def test_block_unleased_kill_switch():
    mgr, _, _ = _mgr(lease_block_unleased=False)
    _claim(mgr, preemptible=False)
    assert mgr.blocks_unleased("primary") is None


def test_any_live_lease_blocks_idle_unload():
    mgr, _, _ = _mgr()
    soft, _ = _claim(mgr)
    assert mgr.blocks_idle_unload("primary") is soft   # preemptible blocks too
    _backdate(soft)
    assert mgr.blocks_idle_unload("primary") is None


def test_blocks_idle_unload_kill_switch():
    mgr, _, _ = _mgr(lease_blocks_idle_unload=False)
    _claim(mgr)
    assert mgr.blocks_idle_unload("primary") is None


def test_looks_like_pre_restart():
    mgr, _, _ = _mgr()
    assert mgr.looks_like_pre_restart("a" * 12) is True     # plausible id, never seen
    assert mgr.looks_like_pre_restart("nonsense") is False  # wrong shape
    lease, _ = _claim(mgr)
    assert mgr.looks_like_pre_restart(lease.id) is False    # we know this one


# --------------------------------------------------------------------------- #
# Sweeper
# --------------------------------------------------------------------------- #
def _sweeper(mgr, orch, settings):
    return LeaseSweeper(settings, orch, mgr)


async def test_default_preempt_does_not_unload():
    """Preemption is pure revocation: a load already evicts the card itself, so
    unloading here would be a wasted drain/refill."""
    mgr, orch, s = _mgr()
    _claim(mgr, holder="alice")
    _claim(mgr, holder="bob", priority=5)
    await _sweeper(mgr, orch, s)._tick()
    assert orch.units["primary"].unloads == []


async def test_free_on_preempt_unloads_through_unit_unload():
    """The CLAIMANT asks for an empty card; the eviction goes through Unit.unload."""
    mgr, orch, s = _mgr(lease_unused_release_s=0)
    _claim(mgr, holder="alice")
    _claim(mgr, holder="bob", priority=5, free_on_preempt=True)
    await _sweeper(mgr, orch, s)._tick()
    unit = orch.units["primary"]
    assert len(unit.unloads) == 1 and unit.unloads[0].lane == "primary"
    assert unit.touched == 1


async def test_sweeper_defers_free_while_a_swap_is_in_flight():
    mgr, orch, s = _mgr(lease_unused_release_s=0)
    _claim(mgr, holder="bob", free_on_preempt=True)
    unit, sw = orch.units["primary"], _sweeper(mgr, orch, s)
    async with unit._lock:                      # a load/unload is running
        await sw._tick()
    assert unit.unloads == [], "must never fight an in-flight swap"
    await sw._tick()                            # lock released → the free proceeds
    assert len(unit.unloads) == 1


async def test_pending_free_dropped_when_someone_else_takes_over():
    """A queued free must never yank a model out from under a *different* holder."""
    mgr, orch, s = _mgr(lease_unused_release_s=0)
    _claim(mgr, holder="bob", priority=1, free_on_preempt=True)
    _claim(mgr, holder="carol", priority=5)     # carol owns it now
    await _sweeper(mgr, orch, s)._tick()
    assert orch.units["primary"].unloads == []


async def test_operator_revoke_free_unloads_when_unit_stays_unclaimed():
    mgr, orch, s = _mgr(lease_unused_release_s=0)
    lease, _ = _claim(mgr, holder="alice")
    mgr.revoke(lease.id, by="operator", reason="admin", free=True)
    await _sweeper(mgr, orch, s)._tick()
    assert len(orch.units["primary"].unloads) == 1


async def test_operator_revoke_free_skipped_once_reclaimed():
    mgr, orch, s = _mgr(lease_unused_release_s=0)
    lease, _ = _claim(mgr, holder="alice")
    mgr.revoke(lease.id, by="operator", reason="admin", free=True)
    _claim(mgr, holder="dave")                  # someone claimed before the sweep
    await _sweeper(mgr, orch, s)._tick()
    assert orch.units["primary"].unloads == []


async def test_sweep_expires_and_reports():
    mgr, orch, s = _mgr(lease_unused_release_s=0)
    lease, _ = _claim(mgr)
    _backdate(lease)
    assert [l.id for l in mgr.sweep()] == [lease.id]
    assert mgr.sweep() == [], "already-expired leases are not reported twice"


async def test_unused_lease_released_when_unit_stays_free():
    """A claim whose load failed would otherwise 409 traffic on an empty card."""
    mgr, orch, s = _mgr(lease_unused_release_s=1)
    lease, _ = _claim(mgr, preemptible=False)
    lease.acquired_at = time.time() - 60          # older than the window, never used
    orch.units["primary"].owner = "free"
    await _sweeper(mgr, orch, s)._tick()
    assert lease.state == "expired" and lease.revoked_reason == "unused"


async def test_unused_reaper_spares_a_lease_whose_unit_is_loaded():
    mgr, orch, s = _mgr(lease_unused_release_s=1)
    lease, _ = _claim(mgr)
    lease.acquired_at = time.time() - 60
    orch.units["primary"].owner = "ollama"        # something is resident
    await _sweeper(mgr, orch, s)._tick()
    assert lease.state == "active"


async def test_unused_reaper_spares_a_used_lease():
    mgr, orch, s = _mgr(lease_unused_release_s=1)
    lease, _ = _claim(mgr)
    lease.acquired_at = time.time() - 60
    mgr.note_request(lease.id)
    orch.units["primary"].owner = "free"
    await _sweeper(mgr, orch, s)._tick()
    assert lease.state == "active"


async def test_unused_reaper_spares_a_renewing_holder():
    """Renewing is proof of life — a holder staging work (claimed, renewing, no
    traffic yet) must not lose its claim as 'unused'."""
    mgr, orch, s = _mgr(lease_unused_release_s=1)
    lease, _ = _claim(mgr, preemptible=False)
    lease.acquired_at = time.time() - 60          # old claim, zero requests...
    mgr.renew(lease.id)                           # ...but the holder just renewed
    orch.units["primary"].owner = "free"
    await _sweeper(mgr, orch, s)._tick()
    assert lease.state == "active", "an actively-renewing holder is not a ghost"


async def test_sweeper_disabled_flag_noops():
    mgr, orch, s = _mgr(lease_enabled=False)
    sw = _sweeper(mgr, orch, s)
    sw.start()
    assert sw._task is None
    await sw.stop()


async def test_one_unit_failure_does_not_kill_the_tick():
    mgr, orch, s = _mgr(lease_unused_release_s=0)
    for uid in ("primary", "companion"):
        _claim(mgr, holder=f"b-{uid}", unit=uid, free_on_preempt=True)

    async def boom():
        raise RuntimeError("probe exploded")

    orch.units["primary"].status_hook = boom
    await _sweeper(mgr, orch, s)._tick()
    assert len(orch.units["companion"].unloads) == 1, "the other unit must still be freed"


async def test_takeover_during_the_status_probe_cancels_the_free():
    """The sweeper's post-await re-check: if a higher-priority claim lands while
    `status()` is in flight, the queued free must not fire under the new holder."""
    mgr, orch, s = _mgr(lease_unused_release_s=0)
    _claim(mgr, holder="bob", priority=1, free_on_preempt=True)

    unit = orch.units["primary"]

    async def steal_midway():
        _claim(mgr, holder="carol", priority=9)   # lands during the await

    unit.status_hook = steal_midway
    await _sweeper(mgr, orch, s)._tick()
    assert unit.unloads == [], "a takeover during the probe must block the free"


# --------------------------------------------------------------------------- #
# Per-model scoping — a multi-model unit means a claim need not be node-wide
# --------------------------------------------------------------------------- #
async def test_per_model_lease_only_blocks_its_own_model():
    """The point of per-model leases: alice pinning m2 must not 409 bob's m1
    traffic on the same Spark. A node-wide claim still blocks everything."""
    mgr, _, _ = _mgr()
    _claim(mgr, "alice", model="m2", preemptible=False)

    assert mgr.blocks_unleased("primary", "m2") is not None, "the claimed model is gated"
    assert mgr.blocks_unleased("primary", "m1") is None,         "a co-resident model nobody claimed must stay open"
    assert mgr.blocks_idle_unload("primary", "m1") is None,         "and the reaper must still be free to evict it"


async def test_unit_wide_lease_blocks_every_model():
    mgr, _, _ = _mgr()
    _claim(mgr, "alice", preemptible=False)  # no model = the whole node

    for model in ("m1", "m2", None):
        assert mgr.blocks_unleased("primary", model) is not None, model
        assert mgr.blocks_idle_unload("primary", model) is not None, model


async def test_two_holders_can_claim_different_models_on_one_unit():
    mgr, _, _ = _mgr()
    a, _ = _claim(mgr, "alice", model="m1", preemptible=False)
    b, displaced = _claim(mgr, "bob", model="m2", preemptible=False)  # no LeaseConflict

    assert a.id != b.id and displaced is None, "different models don't contend at all"
    assert {l.holder for l in mgr.active_all("primary")} == {"alice", "bob"}
    assert mgr.blocks_unleased("primary", "m1").holder == "alice"
    assert mgr.blocks_unleased("primary", "m2").holder == "bob"


async def test_same_model_still_conflicts():
    """Per-model scoping must not weaken the guarantee within one model."""
    mgr, _, _ = _mgr()
    _claim(mgr, "alice", model="m1", preemptible=False)
    with pytest.raises(LeaseConflict):
        _claim(mgr, "bob", model="m1")


async def test_free_on_preempt_frees_only_the_leased_model():
    """A per-model claimant wants room for ITS model, not an empty node — freeing
    everything would evict a co-tenant it never contended with."""
    mgr, orch, s = _mgr(lease_unused_release_s=0)
    unit = orch.units["primary"]
    unit.models = ["m1", "m2"]
    _claim(mgr, holder="alice", model="m1")
    _claim(mgr, holder="bob", model="m1", priority=5, free_on_preempt=True)

    await _sweeper(mgr, orch, s)._tick()

    assert [r.model for r in unit.unloads] == ["m1"], "the stop must name the model"
    assert unit.models == ["m2"], "the co-tenant survives"


async def test_free_on_preempt_of_a_unit_wide_lease_still_frees_everything():
    mgr, orch, s = _mgr(lease_unused_release_s=0)
    unit = orch.units["primary"]
    unit.models = ["m1", "m2"]
    _claim(mgr, holder="alice")
    _claim(mgr, holder="bob", priority=5, free_on_preempt=True)

    await _sweeper(mgr, orch, s)._tick()

    assert [r.model for r in unit.unloads] == [None], "no model = free the whole unit"
    assert unit.models == []


async def test_a_queued_free_is_dropped_once_its_model_has_gone():
    """Someone else unloading it first satisfies the request; the sweeper must not
    then fall through and free the node."""
    mgr, orch, s = _mgr(lease_unused_release_s=0)
    unit = orch.units["primary"]
    unit.models = ["m2"]                      # m1 already gone
    _claim(mgr, holder="alice", model="m1")
    _claim(mgr, holder="bob", model="m1", priority=5, free_on_preempt=True)

    await _sweeper(mgr, orch, s)._tick()

    assert unit.unloads == [] and unit.models == ["m2"]


async def test_reap_unused_spares_a_lease_whose_model_is_resident_under_its_served_name():
    """Residency reports SERVED names, leases store the canonical alias. Raw
    comparison reaped a staging lease on `m1` while `served-1` was running."""
    mgr, orch, s = _mgr(lease_unused_release_s=1)
    unit = orch.units["primary"]
    unit.models = ["served-1"]                       # the node's served name
    unit.canonical_model = lambda m: {"served-1": "m1"}.get(m, m)
    lease, _ = _claim(mgr, "alice", model="m1")      # the catalog alias
    lease.acquired_at -= 999                          # well past the unused window

    await _sweeper(mgr, orch, s)._tick()

    assert mgr.get(lease.id).state == "active", \
        "a lease whose model IS resident must never be reaped as unused"


async def test_reap_unused_still_reaps_a_true_ghost():
    """The counterpart: model absent, no traffic, window passed -> reaped."""
    mgr, orch, s = _mgr(lease_unused_release_s=1)
    orch.units["primary"].owner = "free"
    lease, _ = _claim(mgr, "alice", model="m1")
    lease.acquired_at -= 999

    await _sweeper(mgr, orch, s)._tick()

    assert mgr.get(lease.id).state == "expired"


async def test_whole_unit_gate_sees_a_nonpreemptible_lease_behind_a_preemptible_one():
    """With several per-model claims, whichever lease active_for surfaced first
    used to be the only one checked — a preemptible m1 lease masked bob's
    non-preemptible m2 claim and let a whole-node unload through."""
    mgr, _, _ = _mgr()
    _claim(mgr, "alice", model="m1", preemptible=True)
    _claim(mgr, "bob", model="m2", preemptible=False)

    blocker = mgr.blocks_unleased("primary")          # the whole-unit question
    assert blocker is not None and blocker.holder == "bob", \
        "a unit-wide action must be refused by ANY non-preemptible claim"

    # And with only preemptible claims, unit-wide actions stay allowed.
    mgr2, _, _ = _mgr()
    _claim(mgr2, "alice", model="m1", preemptible=True)
    assert mgr2.blocks_unleased("primary") is None
