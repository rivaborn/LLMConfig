"""Tiny async subprocess helper shared by the wsl/winsvc/gpu bridges.

Every external command (`wsl.exe`, `powershell.exe`, `nvidia-smi`) goes through
`run_argv`, which never raises: a missing executable becomes rc 127 and a hang
becomes rc 124. That lets the app degrade gracefully when run off-box (e.g. on a
dev machine without WSL) instead of crashing a request handler.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

log = logging.getLogger(__name__)

# How long to wait for a killed process to actually die before abandoning it.
# Short on purpose: this only runs on a path that has already timed out, and the
# whole point is that the caller must get control back.
REAP_TIMEOUT_S = 5.0

# Budget for the `taskkill /T` sweep. It only has to outlive one process-tree
# walk; if it can't finish in this long we fall back to killing the direct child.
TREE_KILL_TIMEOUT_S = 5.0


async def kill_tree(pid: int) -> bool:
    """Kill a process AND its descendants. Returns True if the sweep ran.

    `Popen.kill()` on Windows is `TerminateProcess` on ONE handle, and it does
    not cascade. `wsl.exe` spawns a second `wsl.exe` (the relay to the utility
    VM), so killing the process we launched leaves its sibling running — and per
    `WslRecovery` step 1 those orphans are what hold a wedged distro open. On
    2026-08-04 twenty of them had accumulated, eight per `/api/status` call,
    while every timeout was reported as a clean reap.

    Non-Windows falls back to the caller's plain kill (returns False). Never
    raises: this runs on a path that has already failed.
    """
    if os.name != "nt":
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            "taskkill.exe", "/PID", str(pid), "/T", "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except (FileNotFoundError, NotImplementedError, OSError) as e:
        log.debug("kill_tree: cannot exec taskkill (%s)", e)
        return False
    try:
        await asyncio.wait_for(proc.wait(), timeout=TREE_KILL_TIMEOUT_S)
        return True
    except asyncio.TimeoutError:
        # taskkill itself wedged. Don't leave IT behind too.
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return False


@dataclass
class CmdResult:
    rc: int
    out: str
    err: str

    @property
    def ok(self) -> bool:
        return self.rc == 0

    def text(self) -> str:
        """Best-effort human text: stdout if present, else stderr."""
        return (self.out or self.err).strip()


async def run_argv(argv: list[str], timeout: float = 30.0, env: dict | None = None) -> CmdResult:
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except (FileNotFoundError, NotImplementedError) as e:
        # FileNotFoundError: executable absent. NotImplementedError: no subprocess
        # support on the running event loop (e.g. a non-Proactor loop on Windows).
        return CmdResult(127, "", f"cannot exec {argv[0]!r}: {e}")

    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        # Kill the whole TREE first. `proc.kill()` alone reaps the handle we own
        # and leaves any child running, which reads as a clean timeout while the
        # orphan keeps the distro wedged (2026-08-04). Fall back to the direct
        # kill when the sweep is unavailable (non-Windows) or itself times out.
        if not await kill_tree(proc.pid):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        # Bound the reap. `kill()` is a REQUEST: a process wedged in an
        # uninterruptible kernel call ignores it, and a bare `await proc.wait()`
        # then never returns — which silently breaks this module's whole "a hang
        # becomes rc 124" contract. On 2026-07-28 a wedged `wsl.exe` hung here
        # for 5 h holding a unit's swap lock, stacked 29 jobs behind it, and
        # stalled the Monitor loop (every sampler funnels through run_argv).
        # If the reap fails, abandon the handle and report the timeout anyway;
        # the orphan is cleaned up by the WSL recovery ladder.
        try:
            await asyncio.wait_for(proc.wait(), timeout=REAP_TIMEOUT_S)
        except asyncio.TimeoutError:
            return CmdResult(
                124, "",
                f"timeout after {timeout}s running {argv[0]!r}; "
                f"process survived kill() — abandoned (pid {proc.pid})",
            )
        return CmdResult(124, "", f"timeout after {timeout}s running {argv[0]!r}")

    rc = proc.returncode if proc.returncode is not None else -1
    return CmdResult(rc, out_b.decode("utf-8", "replace"), err_b.decode("utf-8", "replace"))
