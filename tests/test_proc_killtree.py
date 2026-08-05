"""run_argv's timeout path must kill the process TREE, not just its own child.

`wsl.exe` spawns a second `wsl.exe`; TerminateProcess on the handle we own
leaves the sibling running. Those orphans are what hold a wedged distro open
(WslRecovery step 1), and on 2026-08-04 twenty had accumulated while every
timeout was reported as a clean reap.
"""
import asyncio

import pytest

import llmconfig.proc as proc_mod
from llmconfig.proc import run_argv


class FakeProc:
    def __init__(self, *, dies_on_kill=True):
        self.pid = 4321
        self.killed = False
        self.returncode = None
        self._dies_on_kill = dies_on_kill

    async def communicate(self):
        await asyncio.sleep(3600)      # hang until the caller's timeout fires

    def kill(self):
        self.killed = True

    async def wait(self):
        if self._dies_on_kill:
            self.returncode = -9
            return -9
        await asyncio.sleep(3600)      # survives kill() — the abandoned case


@pytest.fixture
def hung(monkeypatch):
    proc = FakeProc()

    async def fake_exec(*argv, **kw):
        return proc

    monkeypatch.setattr(proc_mod.asyncio, "create_subprocess_exec", fake_exec)
    return proc


async def test_timeout_kills_the_tree(hung, monkeypatch):
    swept: list[int] = []

    async def fake_kill_tree(pid):
        swept.append(pid)
        hung.returncode = -9
        return True

    monkeypatch.setattr(proc_mod, "kill_tree", fake_kill_tree)
    r = await run_argv(["wsl.exe", "-d", "d", "--", "true"], timeout=0.01)
    assert r.rc == 124
    assert swept == [hung.pid], "the whole tree must be swept, not just our child"
    assert hung.killed is False, "tree sweep succeeded; no need for the fallback"


async def test_falls_back_to_direct_kill_when_sweep_unavailable(hung, monkeypatch):
    async def no_sweep(pid):
        return False           # e.g. non-Windows, or taskkill itself wedged

    monkeypatch.setattr(proc_mod, "kill_tree", no_sweep)
    r = await run_argv(["wsl.exe"], timeout=0.01)
    assert r.rc == 124
    assert hung.killed is True, "must still kill the direct child"


async def test_surviving_process_is_reported_as_abandoned(monkeypatch):
    proc = FakeProc(dies_on_kill=False)

    async def fake_exec(*argv, **kw):
        return proc

    async def no_sweep(pid):
        return False

    monkeypatch.setattr(proc_mod.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(proc_mod, "kill_tree", no_sweep)
    monkeypatch.setattr(proc_mod, "REAP_TIMEOUT_S", 0.01)
    r = await run_argv(["wsl.exe"], timeout=0.01)
    assert r.rc == 124 and "abandoned" in r.err


async def test_kill_tree_is_a_noop_off_windows(monkeypatch):
    monkeypatch.setattr(proc_mod.os, "name", "posix")
    assert await proc_mod.kill_tree(1234) is False
