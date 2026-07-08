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

Invariant: reaping goes ONLY through `Lane.unload` — the lane lock + the
eviction-wait gate — never a private unload path. After reaping the last vLLM
(no lane serving vLLM, no lane lock held) the shared WSL keepalive is released
so the distro can idle-shutdown too; the next vLLM load re-`ensure()`s it.

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
from .schemas import LaneStatus, LaneUsage, UnloadRequest

if TYPE_CHECKING:
    from .lane import Lane
    from .monitor import Monitor
    from .orchestrator import Orchestrator

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
    if st.owner not in ("ollama", "vllm"):
        return "free"
    if st.idle_s is not None and st.idle_s <= settings.usage_active_window_s:
        return "active"
    if current_util_pct is not None and current_util_pct > settings.idle_unload_util_pct:
        return "active"
    return "idle"


class IdleReaper:
    def __init__(self, settings: Settings, orch: "Orchestrator", monitor: "Monitor"):
        self.s = settings
        self.orch = orch
        self.monitor = monitor
        self.interval = max(5.0, float(settings.idle_unload_check_interval_s))
        self.timeout_s = max(60.0, float(settings.idle_unload_after_min) * 60.0)
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

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
        reaped_vllm = False
        for lane in list(self.orch.lanes.values()):
            try:
                reaped_vllm |= await self._check_lane(lane)
            except Exception as e:  # noqa: BLE001 — one lane's failure can't starve the other
                log.warning("idle reaper: lane %s check failed: %s: %s",
                            lane.cfg.id, type(e).__name__, e)
        if reaped_vllm:
            await self._maybe_release_keepalive()

    async def _check_lane(self, lane: "Lane") -> bool:
        """Reap the lane if idle past the timeout. Returns True if vLLM was reaped."""
        # Fold in the Monitor util signal (catches direct-to-backend clients).
        ts = self.monitor.last_util_activity(
            lane.cfg.gpu_uuid, self.s.idle_unload_util_pct, since=lane.last_activity
        )
        if ts is not None:
            lane.touch(ts)
        # Cheap guards before any HTTP/nvidia-smi probing.
        if not lane.cfg.enabled:
            return False
        if lane._lock.locked() or lane._active_job_id:  # swap in progress
            return False
        idle = time.time() - lane.last_activity
        if idle < self.timeout_s:
            return False
        # Something we manage must actually hold the card (free/unknown → nothing to do).
        st = await lane.status()
        if st.swap_in_progress or st.owner not in ("ollama", "vllm"):
            return False
        # Final sync re-check, then reap through the existing lock + eviction-wait
        # path. No await between the check and unload(): an uncontended asyncio.Lock
        # acquires without yielding, so a competing load can't interleave.
        if lane._lock.locked():
            return False
        model = st.loaded.model if st.loaded else "?"
        log.info("idle reaper: lane %s idle %.1f min — unloading %s (%s)",
                 lane.cfg.id, idle / 60.0, model, st.owner)
        await lane.unload(UnloadRequest(server=None, lane=lane.cfg.id))
        lane.touch()  # restart the window so a slow VRAM drain isn't re-reaped every tick
        return st.owner == "vllm"

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
            if await lane.vllm.up():
                return
        if any(lane._lock.locked() for lane in lanes):
            return
        log.info("idle reaper: no vLLM on any lane — releasing the WSL keepalive")
        ka.stop()
