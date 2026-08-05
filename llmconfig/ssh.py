r"""Native OpenSSH transport for Spark node access (no WSL in the path).

Spark *telemetry* is nothing but `ssh <user>@<host> nvidia-smi …`. It ran inside
WSL only because that is where sparkrun's key pair happened to live — not for
any capability WSL provides. The cost of that accident was total: when the
distro's exec path wedged on 2026-08-04, all four Sparks reported `found=False`
with 0 MB VRAM even though every node was healthy and directly reachable.

Windows has had OpenSSH in `%ProgramFiles%\OpenSSH` for years, so telemetry can
talk to the nodes directly and survive a wedged distro. `sparkrun` lifecycle
(load/unload) still goes through WSL — it genuinely lives there.

A second, free win: the remote command is passed as ONE argv element to a real
`execve`, so it never meets a local shell. The quote mangling documented in
`backends/spark.py` (wsl.exe eats embedded double quotes, and `shlex.quote`
escapes for the *wrong* shell on the way through) cannot happen on this path.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from .config import Settings, get_settings
from .proc import CmdResult, run_argv

log = logging.getLogger(__name__)

# BatchMode: never prompt (this runs headless under a scheduled task).
# IdentitiesOnly: offer ONLY the key we name. Without it ssh walks its default
#   identities first and a node with MaxAuthTries can reject us before reaching
#   the right one — the classic "key is fine, auth still fails" trap.
# accept-new: trust an unseen node once, but still fail on a CHANGED host key.
DEFAULT_SSH_OPTS: tuple[str, ...] = (
    "BatchMode=yes",
    "ConnectTimeout=5",
    "IdentitiesOnly=yes",
    "StrictHostKeyChecking=accept-new",
)


def _ssh_exe() -> str:
    return "ssh.exe" if os.name == "nt" else "ssh"


def resolve_key(settings: Settings) -> str | None:
    """Absolute path to the Spark key, or None to let ssh use its own config.

    Degrading to None rather than passing a missing `-i` is deliberate: a typo'd
    or not-yet-provisioned key should fall back to whatever the user's ssh_config
    already does, not hard-fail every probe with 'no such identity'.
    """
    raw = (settings.spark_ssh_key or "").strip()
    if not raw:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(raw)))
    if not path.exists():
        log.warning("spark_ssh_key %s does not exist — falling back to ssh defaults", path)
        return None
    return str(path)


def ssh_argv(
    user: str,
    host: str,
    command: str,
    *,
    key: str | None = None,
    opts: "tuple[str, ...] | list[str] | None" = None,
    port: int | None = None,
) -> list[str]:
    argv: list[str] = [_ssh_exe()]
    for opt in (DEFAULT_SSH_OPTS if opts is None else opts):
        argv += ["-o", opt]
    if key:
        argv += ["-i", key]
    if port:
        argv += ["-p", str(port)]
    # The remote command is a SINGLE element: no local shell ever sees it.
    argv += [f"{user}@{host}", command]
    return argv


async def run_ssh(
    user: str,
    host: str,
    command: str,
    *,
    timeout: float = 20.0,
    settings: Settings | None = None,
) -> CmdResult:
    settings = settings or get_settings()
    argv = ssh_argv(user, host, command, key=resolve_key(settings))
    return await run_argv(argv, timeout=timeout)
