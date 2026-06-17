"""Control the Windows-native Ollama service via PowerShell.

`Restart-Service`/`Start-Service` may require the app to run elevated (or an ACL
grant on the service) — `llmconfig doctor` reports whether it currently has rights.
"""
from __future__ import annotations

from .proc import CmdResult, run_argv

# Status strings we normalize to. PowerShell's ServiceControllerStatus enum values.
RUNNING = "Running"
STOPPED = "Stopped"
NOT_FOUND = "NotFound"
UNKNOWN = "Unknown"


async def _ps(command: str, timeout: float = 25.0) -> CmdResult:
    return await run_argv(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        timeout=timeout,
    )


async def service_status(name: str) -> str:
    r = await _ps(f"(Get-Service -Name '{name}' -ErrorAction SilentlyContinue).Status")
    if not r.ok:
        return UNKNOWN
    return r.out.strip() or NOT_FOUND


async def is_running(name: str) -> bool:
    return (await service_status(name)) == RUNNING


async def start_service(name: str) -> CmdResult:
    return await _ps(f"Start-Service -Name '{name}'")


async def restart_service(name: str) -> CmdResult:
    return await _ps(f"Restart-Service -Name '{name}' -Force")


def looks_like_access_denied(r: CmdResult) -> bool:
    blob = (r.out + r.err).lower()
    return any(s in blob for s in ("access is denied", "permissiondenied", "cannot open", "requires elevation"))


async def is_elevated() -> bool:
    """Whether the app process is running with Administrator rights (needed to
    Start/Restart a system service like ollama)."""
    r = await _ps(
        "([Security.Principal.WindowsPrincipal]"
        "[Security.Principal.WindowsIdentity]::GetCurrent())"
        ".IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)"
    )
    return r.out.strip().lower() == "true"
