"""Tiny async subprocess helper shared by the wsl/winsvc/gpu bridges.

Every external command (`wsl.exe`, `powershell.exe`, `nvidia-smi`) goes through
`run_argv`, which never raises: a missing executable becomes rc 127 and a hang
becomes rc 124. That lets the app degrade gracefully when run off-box (e.g. on a
dev machine without WSL) instead of crashing a request handler.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass


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
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        return CmdResult(124, "", f"timeout after {timeout}s running {argv[0]!r}")

    rc = proc.returncode if proc.returncode is not None else -1
    return CmdResult(rc, out_b.decode("utf-8", "replace"), err_b.decode("utf-8", "replace"))
