"""Bridge into the WSL2 distro that hosts vLLM.

The control app runs Windows-native; everything vLLM-side (serve.sh, systemctl
--user, nvidia-smi, pkill) is executed through `wsl.exe -d <distro> -u <user>`.
"""
from __future__ import annotations

import os

from .config import Settings, get_settings
from .proc import CmdResult, run_argv

# WSL_UTF8=1 makes wsl.exe emit its own messages as UTF-8 instead of UTF-16LE
# (otherwise distro-not-found etc. come back as garbled spaced-out text).
_WSL_ENV = {**os.environ, "WSL_UTF8": "1"}


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
