"""WslHealth: the breaker + runtime escalation added after the 2026-08-04 wedge.

The failure these cover: `WslRecovery` existed and worked, but its ONLY call
site was the boot gate, so a distro that wedged at runtime stayed wedged — and
every observer of the wedge made it worse (one /api/status = 4 Spark lanes = 4
exec timeouts and 8 abandoned wsl.exe).

Nothing here spawns a process; `run_argv` is faked at the wsl module boundary.
"""
import asyncio

import pytest

import llmconfig.wsl as wsl_mod
from llmconfig.config import Settings
from llmconfig.proc import CmdResult

TIMEOUT = CmdResult(124, "", "timeout after 20.0s running 'wsl.exe'")
NO_WSL = CmdResult(127, "", "cannot exec 'wsl.exe'")
OK = CmdResult(0, "ok", "")
REMOTE_FAIL = CmdResult(1, "", "nvidia-smi: command not found")


@pytest.fixture
def settings():
    return Settings(
        wsl_distro="d", wsl_user="u",
        wsl_selfheal_after_failures=3,
        wsl_breaker_retry_s=60.0,
    )


@pytest.fixture(autouse=True)
def _fresh_health():
    wsl_mod.reset_health()
    yield
    wsl_mod.reset_health()


@pytest.fixture
def fake_exec(monkeypatch):
    """Replace run_argv; `.result` decides what every exec returns."""
    box = type("Box", (), {"result": OK, "calls": 0})()

    async def fake_run_argv(argv, timeout=30.0, env=None):
        box.calls += 1
        return box.result

    monkeypatch.setattr(wsl_mod, "run_argv", fake_run_argv)
    return box


async def _run(settings, n=1):
    for _ in range(n):
        r = await wsl_mod.run_wsl("true", settings=settings)
    return r


async def test_breaker_opens_after_threshold_and_short_circuits(settings, fake_exec):
    fake_exec.result = TIMEOUT
    await _run(settings, 3)
    assert fake_exec.calls == 3
    health = wsl_mod.health(settings)
    assert health.wedged is True

    # The 4th caller must NOT spawn a process — that is the whole point.
    r = await wsl_mod.run_wsl("true", settings=settings)
    assert fake_exec.calls == 3, "breaker must short-circuit without exec'ing"
    assert r.rc == 124 and "short-circuited" in r.err


async def test_success_closes_the_breaker(settings, fake_exec):
    fake_exec.result = TIMEOUT
    await _run(settings, 3)
    assert wsl_mod.health(settings).wedged is True

    # probe() bypasses the breaker, so a recovered distro is observable.
    fake_exec.result = OK
    r = await wsl_mod.probe(settings=settings)
    assert r.ok
    assert wsl_mod.health(settings).wedged is False
    assert wsl_mod.health(settings).consecutive_timeouts == 0


async def test_remote_command_failure_is_not_a_wedge(settings, fake_exec):
    """rc != 0 from the REMOTE command proves the exec path works."""
    fake_exec.result = REMOTE_FAIL
    await _run(settings, 5)
    assert wsl_mod.health(settings).wedged is False
    assert fake_exec.calls == 5


async def test_missing_wsl_never_opens_the_breaker(settings, fake_exec):
    """rc 127 = dev box with no wsl.exe. Nothing to recover, nothing to break."""
    fake_exec.result = NO_WSL
    await _run(settings, 5)
    health = wsl_mod.health(settings)
    assert health.wedged is False and health.consecutive_timeouts == 0


async def test_probe_bypasses_the_breaker(settings, fake_exec):
    fake_exec.result = TIMEOUT
    await _run(settings, 3)
    before = fake_exec.calls
    await wsl_mod.probe(settings=settings)
    assert fake_exec.calls == before + 1, "probe must always really try"


async def test_wedge_escalates_to_recovery_in_background(settings, fake_exec):
    """The regression that mattered: a runtime wedge must CALL the ladder."""
    fired: list[int] = []

    async def fake_attempt(n):
        fired.append(n)
        return True

    wsl_mod.health(settings).register(fake_attempt)
    fake_exec.result = TIMEOUT
    await _run(settings, 3)
    await asyncio.sleep(0)          # let the background task run
    await asyncio.sleep(0)
    assert fired, "a runtime wedge must escalate to WslRecovery"
    assert fired[0] >= settings.wsl_selfheal_after_failures


async def test_escalation_is_not_stacked(settings, fake_exec):
    """Repeated timeouts must not pile up ladder invocations."""
    started = 0
    release = asyncio.Event()

    async def slow_attempt(n):
        nonlocal started
        started += 1
        await release.wait()
        return False

    wsl_mod.health(settings).register(slow_attempt)
    fake_exec.result = TIMEOUT
    await _run(settings, 6)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert started == 1, "one ladder at a time"
    release.set()


async def test_breaker_admits_a_trial_after_retry_window(settings, fake_exec, monkeypatch):
    fake_exec.result = TIMEOUT
    await _run(settings, 3)
    health = wsl_mod.health(settings)
    assert health.allow() is False

    # Age the breaker past the retry window instead of sleeping 60 s.
    health.opened_at -= settings.wsl_breaker_retry_s + 1
    assert health.allow() is True, "half-open trial must be admitted"
    assert health.allow() is False, "…but only one per window"
