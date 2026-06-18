"""Bridge into the WSL2 distro that hosts vLLM.

The control app runs Windows-native; everything vLLM-side (serve.sh, systemctl
--user, nvidia-smi, pkill) is executed through `wsl.exe -d <distro> -u <user>`.
"""
from __future__ import annotations

import os
import subprocess

from .config import Settings, get_settings
from .proc import CmdResult, run_argv

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
