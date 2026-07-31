"""Idle auto-unload — reap an unused lane so its GPU drops to low-power P8.

A resident model pins the card in the P0 power state (memory clocks never drop:
~117 W on the 3090 doing nothing, vs ~25 W once VRAM is freed). Neither server
lets go on its own here — LLMConfig loads Ollama with `keep_alive:-1` and vLLM
never auto-unloads — so this background loop is the policy that frees the card
after sustained inactivity. Reloading is already hands-off: the /v1 gateway
auto-loads on the next request, and direct-Ollama clients reload through Ollama.

"Activity" is a hybrid of three signals, because some clients bypass the gateway
and talk to Ollama / the vLLM relay directly:
  * a /v1 gateway request routed to the lane (`Lane.touch()` in openai_gateway),
  * a load finishing (`Lane.load`'s finally),
  * a Monitor utilization sample above `idle_unload_util_pct` on the lane's GPU
    (folded in each tick via `Monitor.last_util_activity`, matched by UUID).

A unit that can hold several models at once (a DGX Spark) clocks each model
separately as well, and the reaper decides PER MODEL — the unit clock is the max
across them, so reaping off it alone would let one busy model keep every idle
neighbour resident. One model is reaped per tick, each behind its own lock
acquisition and re-check. A GPU lane holds a single model and has no per-model
clocks, so the same code path collapses to the original unit-level decision.

Invariant: reaping goes ONLY through `Lane.unload` — the lane lock + the
eviction-wait gate — never a private unload path. After reaping the last vLLM
(no lane serving vLLM, no lane lock held) the shared WSL keepalive is released
so the distro can idle-shutdown too; the next vLLM load re-`ensure()`s it.

Participation is per lane (`LaneConfig.idle_unload_enabled`): the companion
3070 Ti is exempt by default — it idles in P8 (~13 W) even with a small model
resident, so reaping it saves ~nothing and would cost the opencode /swap echo
relay its instant response (`COMPANION_IDLE_UNLOAD_ENABLED=1` opts it in).

Degrades gracefully off-box: with no Monitor samples the util signal simply goes
quiet and the timestamps still drive the policy; a tick failure never kills the
loop.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional

from .config import Settings
from .schemas import MANAGED_OWNERS, LaneStatus, LaneUsage, LoadedModel, UnloadRequest

if TYPE_CHECKING:
    from .leases import LeaseManager
    from .monitor import Monitor
    from .orchestrator import Orchestrator, Unit

log = logging.getLogger(__name__)


def classify_usage(st: LaneStatus, current_util_pct: float | None,
                   settings: Settings) -> LaneUsage:
    """Classify a lane free / idle / active — the answer behind GET /api/usage.

    free = nothing we manage holds the card; active = loaded with recent activity
    (`idle_s` within `usage_active_window_s`), currently-visible GPU utilization
    (covers direct-to-backend clients, whose util only folds into `idle_s` on the
    reaper's next tick), or a swap in flight; idle = loaded but none of the above.
    Pure function: the caller supplies the Monitor's latest util for the lane's GPU
    (None when the Monitor is off/off-box — the timestamps still classify).
    """
    if st.swap_in_progress:
        return "active"  # a load/unload is running — the lane is busy, not free
    if st.owner not in MANAGED_OWNERS:
        return "free"
    if st.idle_s is not None and st.idle_s <= settings.usage_active_window_s:
        return "active"
    if current_util_pct is not None and current_util_pct > settings.idle_unload_util_pct:
        return "active"
    return "idle"


class IdleReaper:
    def __init__(self, settings: Settings, orch: "Orchestrator", monitor: "Monitor",
                 leases: "LeaseManager"):
        self.s = settings
        self.orch = orch
        self.monitor = monitor
        # Required, not optional: a None default would let a future call site
        # silently lose lease-awareness and reap a unit somebody had claimed.
        self.leases = leases
        self.interval = max(5.0, float(settings.idle_unload_check_interval_s))
        self.timeout_s = max(60.0, float(settings.idle_unload_after_min) * 60.0)
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        # Log a lease-blocked skip once per (unit, lease) — otherwise a 45-minute
        # lease emits 45 identical lines at the 60 s tick cadence.
        self._lease_logged: dict[str, str] = {}

    # ---- lifecycle (mirrors Monitor) ----
    def start(self) -> None:
        if not self.s.idle_unload_enabled or self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="idle-reaper")

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
                log.warning("idle reaper tick failed: %s: %s", type(e).__name__, e)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                pass

    # ---- one policy pass ----
    async def _tick(self) -> None:
        # Every unit participates (Sparks are opted out via their own
        # idle_unload_enabled, but their idle_s/usage still needs folding in).
        reaped_vllm = False
        for lane in list(self.orch.units.values()):
            try:
                reaped_vllm |= await self._check_lane(lane)
            except Exception as e:  # noqa: BLE001 — one unit's failure can't starve the others
                log.warning("idle reaper: lane %s check failed: %s: %s",
                            lane.cfg.id, type(e).__name__, e)
        # Every tick, not only reap ticks: a release skipped once (another lane
        # mid-swap at that instant) would otherwise never be retried and the
        # keepalive would pin the whole distro's RAM indefinitely. The method's
        # own guards make the no-op case cheap.
        ka = getattr(self.orch, "keepalive", None)
        if reaped_vllm or (ka is not None and ka.alive()):
            await self._maybe_release_keepalive()

    async def _check_lane(self, lane: "Unit") -> bool:
        """Reap the lane if idle past the timeout. Returns True if vLLM was reaped."""
        # Fold in the Monitor util signal (catches direct-to-backend clients). Done
        # even for reap-exempt lanes so their idle_s / usage classification stays honest.
        ts = self.monitor.last_util_activity(
            lane.cfg.gpu_uuid, self.s.idle_unload_util_pct, since=lane.last_activity
        )
        if ts is not None:
            lane.touch(ts)
            # The util signal says "this device is busy" without saying which model is
            # doing the work, so it counts for all of them. Bumping only the unit clock
            # would let per-model reaping evict a model that a direct-to-backend client
            # is actively using.
            for known in list(getattr(lane, "model_activity", {})):
                lane.touch(ts, model=known)
        # Cheap guards before any HTTP/nvidia-smi probing.
        if not lane.cfg.enabled:
            return False
        # The runtime PIN override (the UI checkbox) beats the static config in
        # BOTH directions: pinned=True shields a lane the .env left reapable,
        # pinned=False makes an .env-exempt lane reapable. No entry = config.
        pins = getattr(self.orch, "pins", None)
        pin = pins.get(lane.cfg.id) if pins is not None else None
        if pin is True or (pin is None and not lane.cfg.idle_unload_enabled):
            return False
        # A claimed unit is off-limits — including a *preemptible* claim, because the
        # reaper is a power-saving optimisation, not a competing caller. Checked here
        # so a leased lane costs no nvidia-smi/HTTP probe at all. Only a WHOLE-UNIT
        # claim short-circuits: a lease naming one model must not shield that model's
        # idle neighbours, so it is enforced per model further down instead.
        held = self.leases.blocks_idle_unload(lane.cfg.id)
        if held is not None and not held.model:
            if self._lease_logged.get(lane.cfg.id) != held.id:
                self._lease_logged[lane.cfg.id] = held.id
                log.info("idle reaper: lane %s leased by '%s' (%s) — skipping",
                         lane.cfg.id, held.holder, held.id)
            return False
        self._lease_logged.pop(lane.cfg.id, None)
        if lane._lock.locked() or lane._active_job_id:  # swap in progress
            return False
        # Cheap pre-probe guard on the OLDEST clock this unit knows — the unit clock plus
        # every per-model clock. Using the unit clock alone would be wrong on a multi-model
        # Spark: it is the max across models, so one busy model would keep every idle
        # neighbour resident forever, which is exactly the memory we want back.
        clocks = [lane.last_activity, *getattr(lane, "model_activity", {}).values()]
        if time.time() - min(clocks) < self.timeout_s:
            return False
        # Something we manage must actually hold the unit (free/unknown → nothing to do).
        st = await lane.status()
        if st.swap_in_progress or st.owner not in MANAGED_OWNERS:
            return False

        # Which resident models are individually past the timeout and unleased? A lane
        # holds one model and has no per-model clocks, so this collapses to the old
        # unit-level decision for it.
        idle_of = getattr(lane, "idle_for", None)
        stale: list[tuple[LoadedModel, float]] = []
        for m in st.loaded_models:
            # A multi-node residency (m.group) is one rank of a deployment
            # spanning OTHER nodes — this unit's unload() refuses it outright
            # (a stop on one rank wedges the rest), so choosing it as the
            # victim would just raise on every tick AND shadow a genuinely
            # reapable neighbour (the claimed row has no clock of its own here,
            # so it reads as the stalest). Teardown is /api/cluster/unload only.
            if getattr(m, "group", ""):
                continue
            idle = idle_of(m.model) if idle_of else (time.time() - lane.last_activity)
            if idle >= self.timeout_s and self.leases.blocks_idle_unload(lane.cfg.id, m.model) is None:
                stale.append((m, idle))
        if not stale:
            return False
        # Reap the single stalest model per tick. One unload per pass keeps each reap
        # behind its own lock acquisition and re-check, so a load that lands mid-sweep
        # can't have a later victim chosen from a status snapshot taken before it.
        victim, idle = max(stale, key=lambda p: p[1])
        # Final sync re-check, then reap through the existing lock + eviction-wait
        # path. No await between these checks and unload(): an uncontended asyncio.Lock
        # acquires without yielding and LeaseManager.blocks_idle_unload is pure dict
        # access, so neither a competing load nor a lease claimed during the status
        # probe above can interleave.
        if lane._lock.locked() or self.leases.blocks_idle_unload(lane.cfg.id, victim.model) is not None:
            return False
        log.info("idle reaper: lane %s idle %.1f min — unloading %s (%s)",
                 lane.cfg.id, idle / 60.0, victim.model, victim.server)
        # Name the model even when it is the unit's only occupant: unload(model=…) is a
        # targeted stop, so a co-resident neighbour loaded between the probe and here
        # survives instead of being collateral.
        await lane.unload(UnloadRequest(server=None, lane=lane.cfg.id, model=victim.model))
        # Restart both windows so a slow VRAM drain isn't re-reaped every tick.
        lane.touch(model=victim.model)
        return victim.server == "vllm"

    async def _maybe_release_keepalive(self) -> None:
        """After reaping vLLM: if no lane serves vLLM anymore, drop the shared WSL hold
        so the distro can idle-shutdown. Skipped while any lane lock is held (a swap may
        be about to serve vLLM). Safe against a concurrent load: `_load_vllm` calls
        `keepalive.ensure()` UNDER its lane lock, and the lock check → `stop()` below has
        no await between them; a later load's `ensure()` simply respawns the hold."""
        ka = self.orch.keepalive
        if not ka.alive():
            return
        lanes = list(self.orch.lanes.values())
        for lane in lanes:
            # `vllm_up()` is the shared unit contract: Lane skips its dead relay
            # on an Ollama-only lane (invariant 5's blackholed SYN), SlotLane
            # answers for ALL its slots — reaching into `lane.vllm` here would
            # miss every slot but a default one and release the keepalive under
            # live companion slots.
            if await lane.vllm_up():
                return
        if any(lane._lock.locked() for lane in lanes):
            return
        log.info("idle reaper: no vLLM on any lane — releasing the WSL keepalive")
        ka.stop()
