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


async def run_wsl(
    command: str,
    *,
    login: bool = True,
    timeout: float = 30.0,
    settings: Settings | None = None,
) -> CmdResult:
    settings = settings or get_settings()
    return await run_argv(wsl_argv(command, settings, login=login), timeout=timeout, env=_WSL_ENV)


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
    )


async def wait_ready(
    *,
    settings: Settings | None = None,
    deadline_s: float | None = None,
    on_stall: "Callable[[int], Awaitable[None]] | None" = None,
) -> bool:
    """Block until the distro can execute a command, or the deadline passes.

    Returns True once a probe succeeds. Returns False on deadline — callers must
    treat that as 'skip the WSL-dependent work', never as fatal: the app has to
    keep serving on a box with no wsl.exe at all (`run_argv` gives rc 127).

    `on_stall(consecutive_timeouts)` is invoked when probes keep timing out, so
    the caller can escalate to recovery without this module owning that policy.
    """
    settings = settings or get_settings()
    budget = settings.wsl_ready_timeout_s if deadline_s is None else deadline_s
    loop = asyncio.get_running_loop()
    end = loop.time() + budget
    timeouts = 0
    stalled_at = 0
    while True:
        r = await probe(settings=settings)
        if r.ok:
            return True
        # rc 127 = no wsl.exe on this machine (dev box). Nothing to wait for.
        if r.rc == 127:
            return False
        timeouts = timeouts + 1 if r.rc == 124 else 0
        if (
            on_stall is not None
            and timeouts >= settings.wsl_selfheal_after_failures
            and timeouts != stalled_at
        ):
            stalled_at = timeouts
            await on_stall(timeouts)
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
