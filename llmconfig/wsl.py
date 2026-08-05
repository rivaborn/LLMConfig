"""Bridge into the WSL2 distro that hosts vLLM.

The control app runs Windows-native; everything vLLM-side (serve.sh, systemctl
--user, nvidia-smi, pkill) is executed through `wsl.exe -d <distro> -u <user>`.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from typing import Awaitable, Callable

from .config import Settings, get_settings
from .proc import CmdResult, run_argv

log = logging.getLogger(__name__)

# WSL_UTF8=1 makes wsl.exe emit its own messages as UTF-8 instead of UTF-16LE
# (otherwise distro-not-found etc. come back as garbled spaced-out text).
_WSL_ENV = {**os.environ, "WSL_UTF8": "1"}

# Don't pop a console window for the detached keepalive (Windows only).
_CREATE_NO_WINDOW = 0x08000000


def wsl_argv(command: str, settings: Settings, *, login: bool) -> list[str]:
    # `-l` (login shell) sources /etc/profile.d (CUDA env etc.); serve.sh also
    # self-exports what it needs, so login is belt-and-suspenders.
    flag = "-lc" if login else "-c"
    return ["wsl.exe", "-d", settings.wsl_distro, "-u", settings.wsl_user, "--", "bash", flag, command]


class WslHealth:
    """Is the `wsl.exe` EXEC path working? Breaker + escalation for when it isn't.

    Two failures on 2026-08-04 motivated this, both invisible to the pre-existing
    machinery because `WslRecovery` was only ever reachable from the BOOT gate
    (`main.py` → `wait_ready(on_stall=...)`):

    * **Nothing recovered a wedge that began at runtime.** The distro answered
      `wsl --status` while every exec hung; Spark telemetry returned rc 124 per
      lane and rendered `found=False` indefinitely. The ladder that fixes this
      already existed and was never called.
    * **Every caller paid the wedge separately.** One `/api/status` fans out to
      four Spark lanes, so each poll spent 4 × 20 s and left 8 orphaned
      `wsl.exe` behind — the failure got *worse* the more it was observed.

    So: count consecutive exec timeouts, OPEN a breaker once they cross
    `wsl_selfheal_after_failures`, and hand the wedge to `WslRecovery` in the
    background. While open, callers get an immediate rc 124 instead of spawning
    another doomed process. One trial is admitted every `wsl_breaker_retry_s`
    (half-open) so the breaker can close itself even if recovery is disabled.

    Only rc 124 counts. rc 127 (no `wsl.exe` at all — a dev box) must never open
    the breaker or call recovery, and a non-zero rc from the *remote* command is
    proof the exec path works, so it RESETS the count.
    """

    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.consecutive_timeouts = 0
        self.opened_at: float | None = None
        self.last_trial: float | None = None
        self._on_wedged: "Callable[[int], Awaitable[object]] | None" = None
        self._task: "asyncio.Task | None" = None

    def register(self, on_wedged: "Callable[[int], Awaitable[object]]") -> None:
        """Wire the escalation (normally `WslRecovery.attempt`).

        Kept as a callback so this module owns *detection* and never *policy* —
        the same split `wait_ready(on_stall=...)` already uses.
        """
        self._on_wedged = on_wedged

    def _now(self) -> float:
        try:
            return asyncio.get_running_loop().time()
        except RuntimeError:
            return 0.0

    @property
    def wedged(self) -> bool:
        return self.opened_at is not None

    def allow(self) -> bool:
        """False when the breaker is open and no trial is due."""
        if not self.s.wsl_breaker_enabled or not self.wedged:
            return True
        now = self._now()
        since = now - (self.last_trial if self.last_trial is not None else self.opened_at or now)
        if since >= self.s.wsl_breaker_retry_s:
            self.last_trial = now
            return True
        return False

    def reason(self) -> str:
        return (f"WSL exec wedged ({self.consecutive_timeouts} consecutive timeouts) — "
                f"short-circuited without spawning wsl.exe")

    def record(self, r: CmdResult) -> None:
        if r.rc == 127:          # no wsl.exe on this box; nothing to recover
            return
        if r.rc != 124:          # the exec path answered — including a failed command
            if self.wedged:
                log.warning("WSL exec recovered — closing the breaker")
            self.consecutive_timeouts = 0
            self.opened_at = None
            self.last_trial = None
            return
        self.consecutive_timeouts += 1
        if self.consecutive_timeouts >= self.s.wsl_selfheal_after_failures and not self.wedged:
            self.opened_at = self._now()
            log.warning("WSL exec wedged after %d consecutive timeouts — breaker OPEN",
                        self.consecutive_timeouts)
        if self.wedged:
            self._escalate()

    def _escalate(self) -> None:
        """Fire recovery in the BACKGROUND — callers must not wait on the ladder.

        Telemetry runs on a 2.5 s poll; awaiting a ~3.5 min ladder here would
        stall `/api/status`. `WslRecovery` guards its own re-entrancy and
        cooldown, so re-scheduling is harmless.
        """
        if self._on_wedged is None or (self._task is not None and not self._task.done()):
            return
        try:
            self._task = asyncio.create_task(self._on_wedged(self.consecutive_timeouts))
        except RuntimeError:
            pass   # no running loop (sync context) — nothing to schedule onto


_health: "WslHealth | None" = None


def health(settings: Settings | None = None) -> WslHealth:
    global _health
    if _health is None:
        _health = WslHealth(settings)
    return _health


def reset_health() -> None:
    """Drop the singleton (tests, and any re-configure of Settings)."""
    global _health
    _health = None


async def run_wsl(
    command: str,
    *,
    login: bool = True,
    timeout: float = 30.0,
    settings: Settings | None = None,
    bypass_breaker: bool = False,
) -> CmdResult:
    """Run a command in the distro.

    `bypass_breaker` is for callers whose whole job is to find out whether the
    exec path works (`probe`, and therefore the recovery ladder's own re-probe).
    Everything else goes through the breaker so a wedged distro costs one cheap
    rc 124 instead of another abandoned `wsl.exe`.
    """
    settings = settings or get_settings()
    tracker = health(settings)
    if not bypass_breaker and not tracker.allow():
        return CmdResult(124, "", tracker.reason())
    r = await run_argv(wsl_argv(command, settings, login=login), timeout=timeout, env=_WSL_ENV)
    tracker.record(r)
    return r


async def probe(*, settings: Settings | None = None, timeout: float | None = None) -> CmdResult:
    """Cheapest possible 'can we execute in the distro at all?' check.

    Deliberately goes through the same `wsl.exe -u <user>` path everything else
    uses. A management query (`wsl --status`) is NOT a substitute: on 2026-07-28
    it answered fine while every `-u folar` exec hung forever, so only the exec
    path tells the truth.
    """
    settings = settings or get_settings()
    return await run_wsl(
        "true", login=False,
        timeout=timeout if timeout is not None else settings.wsl_ready_probe_timeout_s,
        settings=settings,
        bypass_breaker=True,   # the probe IS the breaker's way back; never short-circuit it
    )


async def wait_ready(
    *,
    settings: Settings | None = None,
    deadline_s: float | None = None,
    on_stall: "Callable[[int], Awaitable[object]] | None" = None,
) -> bool:
    """Block until the distro can execute a command, or the deadline passes.

    Returns True once a probe succeeds. Returns False on deadline — callers must
    treat that as 'skip the WSL-dependent work', never as fatal: the app has to
    keep serving on a box with no wsl.exe at all (`run_argv` gives rc 127).

    `on_stall(consecutive_timeouts)` is invoked when probes keep timing out, so
    the caller can escalate to recovery without this module owning that policy.
    A truthy return (WslRecovery.attempt returns True on success) triggers an
    immediate re-probe, deadline notwithstanding.
    """
    settings = settings or get_settings()
    budget = settings.wsl_ready_timeout_s if deadline_s is None else deadline_s
    loop = asyncio.get_running_loop()
    end = loop.time() + budget
    timeouts = 0
    while True:
        r = await probe(settings=settings)
        if r.ok:
            return True
        # rc 127 = no wsl.exe on this machine (dev box). Nothing to wait for.
        if r.rc == 127:
            return False
        timeouts = timeouts + 1 if r.rc == 124 else 0
        if on_stall is not None and timeouts >= settings.wsl_selfheal_after_failures:
            recovered = await on_stall(timeouts)
            timeouts = 0   # a fresh consecutive count either way
            if recovered:
                # The ladder just fixed the distro — re-probe IMMEDIATELY, even
                # past the deadline: returning False after a successful recovery
                # (the ladder can take ~3.5 min of a 5 min budget) would skip
                # the boot autoload on a box that now works (review 2026-07-29).
                continue
        if loop.time() >= end:
            return False
        await asyncio.sleep(settings.wsl_ready_backoff_s)


def user_runtime_prefix() -> str:
    """Export so `systemctl/journalctl --user` work from a non-interactive wsl.exe
    call. Lingering (`loginctl enable-linger folar`) keeps the user manager and
    /run/user/<uid> alive at WSL boot; we just point XDG_RUNTIME_DIR at it.
    """
    return 'export XDG_RUNTIME_DIR="/run/user/$(id -u)";'


def user_systemctl(args: str) -> str:
    return f"{user_runtime_prefix()} systemctl --user {args}"


def user_journalctl(args: str) -> str:
    return f"{user_runtime_prefix()} journalctl --user {args}"


class WslKeepalive:
    """Holds the WSL2 distro open for the app's lifetime.

    WSL2 shuts the whole distro down a few seconds after the last `wsl.exe`
    process exits — which kills the `vllm@<alias>` user unit (and the socat
    relay) moments after a load completes, even with lingering enabled. We hold
    one long-lived `wsl.exe … sleep infinity` process: as long as it runs, the
    distro (and the folar systemd-user session) stays up, so a loaded vLLM model
    survives until LLMConfig explicitly evicts it.
    """

    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self._proc: subprocess.Popen | None = None

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def ensure(self) -> bool:
        """Start the keepalive if it isn't already running. Idempotent."""
        if self.alive():
            return True
        argv = ["wsl.exe", "-d", self.s.wsl_distro, "-u", self.s.wsl_user, "--", "sleep", "infinity"]
        kwargs: dict = dict(
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_WSL_ENV,
        )
        if os.name == "nt":
            kwargs["creationflags"] = _CREATE_NO_WINDOW
        try:
            self._proc = subprocess.Popen(argv, **kwargs)
        except (FileNotFoundError, NotImplementedError):
            self._proc = None  # off-box (no wsl.exe) — nothing to keep alive
            return False
        return self.alive()

    def stop(self) -> None:
        """Release the hold; the distro is then free to idle-shut-down."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except (ProcessLookupError, OSError):
                pass
        self._proc = None


WSL_SERVICE = "WslService"


class WslRecovery:
    """Escalating recovery for a distro whose *exec* path has wedged.

    Derived from the 2026-07-28 incident, where a Windows-Update reboot left the
    distro in a state that answered `wsl --status` but hung every `wsl -u folar`
    forever. The ladder is ordered by what actually worked, not by what should
    have:

      1. Kill orphaned `wsl.exe`. The keepalive pair spawned at boot outlived
         everything else and held the distro open.
      2. `wsl.exe --shutdown`. **Timed out twice and did NOT reap the utility
         VM** — attempted for completeness, never trusted.
      3. Restart `WslService`. The only step that actually cleared `vmmemWSL`;
         it needs a generous timeout (~16 stop-poll cycles were observed).

    Killing `wsl.exe` also kills our own `WslKeepalive`; that is intended — a
    wedged keepalive is part of the problem, and the next load re-`ensure()`s it.
    """

    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self._last_attempt: float = 0.0
        self._busy = False
        self.last_outcome: str = "never run"

    def cooling_down(self) -> bool:
        if not self._last_attempt:
            return False
        return (asyncio.get_running_loop().time() - self._last_attempt) < self.s.wsl_selfheal_cooldown_s

    async def attempt(self, consecutive_timeouts: int = 0) -> bool:
        """Run the ladder once. Returns True if the distro executes afterwards.

        Never raises: recovery runs on a background path and a failure here must
        not take the app down. Re-entrancy and cooldown guarded so a stuck distro
        cannot spin the ladder.
        """
        if not self.s.wsl_selfheal_enabled:
            self.last_outcome = "disabled"
            return False
        if self._busy or self.cooling_down():
            return False
        self._busy = True
        self._last_attempt = asyncio.get_running_loop().time()
        try:
            log.warning(
                "WSL exec wedged (%d consecutive probe timeouts) — running recovery ladder",
                consecutive_timeouts,
            )
            await self._kill_orphans()
            await self._shutdown()
            await self._restart_service()
            ok = (await probe(settings=self.s)).ok
            self.last_outcome = "recovered" if ok else "ladder ran, distro still wedged"
            log.warning("WSL recovery: %s", self.last_outcome)
            return ok
        except Exception as e:  # noqa: BLE001 — recovery must never propagate
            self.last_outcome = f"ladder error: {type(e).__name__}: {e}"
            log.exception("WSL recovery ladder raised")
            return False
        finally:
            self._busy = False

    async def _kill_orphans(self) -> None:
        r = await run_argv(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             "Get-Process -Name wsl -ErrorAction SilentlyContinue | Stop-Process -Force"],
            timeout=30.0,
        )
        log.info("WSL recovery step 1 (kill orphans): rc=%s %s", r.rc, r.text()[:200])

    async def _shutdown(self) -> None:
        # Expected to time out on a wedged distro; step 3 is the real fix.
        r = await run_argv(["wsl.exe", "--shutdown"], timeout=45.0, env=_WSL_ENV)
        log.info("WSL recovery step 2 (--shutdown): rc=%s %s", r.rc, r.text()[:200])

    async def _restart_service(self) -> None:
        from . import winsvc  # local import keeps the module import graph flat

        r = await winsvc.restart_service(WSL_SERVICE, timeout=self.s.wsl_service_restart_timeout_s)
        log.info("WSL recovery step 3 (restart %s): rc=%s %s", WSL_SERVICE, r.rc, r.text()[:200])
