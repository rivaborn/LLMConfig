"""Native ssh -> WSL fallback.

The point of the fallback is that a missing or untrusted native key must not
silently reproduce the outage it was introduced to fix. The point of the
DISCRIMINATION is that an offline node must not pay both transports' timeouts
on every poll.
"""
import time

import pytest

import llmconfig.backends.spark as spark_mod
import llmconfig.ssh as ssh_mod
from llmconfig.config import Settings, SparkConfig
from llmconfig.proc import CmdResult
from llmconfig.registry import SparkRegistry
from llmconfig.ssh import is_local_transport_failure

AUTH_FAIL = CmdResult(255, "", "fksogbetun@192.168.1.50: Permission denied (publickey).")
NO_SSH = CmdResult(127, "", "cannot exec 'ssh.exe'")
NODE_DOWN = CmdResult(255, "", "ssh: connect to host 192.168.1.50 port 22: Connection refused")
NODE_TIMEOUT = CmdResult(255, "", "ssh: connect to host 192.168.1.50 port 22: Operation timed out")
GOOD = CmdResult(0, "GPU-abc, [N/A], [N/A], [N/A], 0", "")
VIA_WSL = CmdResult(0, "from-wsl", "")


@pytest.fixture(autouse=True)
def _fresh():
    ssh_mod.reset_native_state()
    yield
    ssh_mod.reset_native_state()


def _backend(tmp_path, **overrides):
    cfg = SparkConfig(
        id="spark1", name="n", host="192.168.1.50", ssh_user="u", api_port=8000,
        registry_path=tmp_path / "r.yaml",
    )
    return spark_mod.SparkBackend(
        Settings(**overrides), cfg, SparkRegistry(cfg.registry_path),
    )


@pytest.fixture
def backend(tmp_path):
    return _backend(tmp_path, spark_ssh_native=True)


@pytest.fixture
def transports(monkeypatch):
    """Record which transport was used; `.native` sets the native result."""
    box = type("Box", (), {"native": GOOD, "used": None})()

    async def fake_run_ssh(user, host, command, *, timeout=20.0, settings=None):
        box.used = "native"
        return box.native

    async def fake_run_wsl(command, *, login=True, timeout=30.0, settings=None):
        box.used = "wsl"
        return VIA_WSL

    monkeypatch.setattr(spark_mod, "run_ssh", fake_run_ssh)
    monkeypatch.setattr(spark_mod, "run_wsl", fake_run_wsl)
    return box


@pytest.mark.parametrize("result", [AUTH_FAIL, NO_SSH])
def test_local_failures_are_recognised(result):
    assert is_local_transport_failure(result) is True


@pytest.mark.parametrize("result", [NODE_DOWN, NODE_TIMEOUT, GOOD])
def test_remote_failures_are_not_local(result):
    assert is_local_transport_failure(result) is False


async def test_auth_failure_falls_back_to_wsl(backend, transports):
    transports.native = AUTH_FAIL
    r = await backend._ssh("nvidia-smi")
    assert transports.used == "wsl"
    assert r.out == "from-wsl", "telemetry must survive a broken native key"


async def test_offline_node_does_not_retry_over_wsl(backend, transports):
    """A dead node fails the same way on both transports — don't pay twice."""
    transports.native = NODE_DOWN
    r = await backend._ssh("nvidia-smi")
    assert transports.used == "native"
    assert r.rc == 255 and "Connection refused" in r.err


async def test_healthy_native_never_touches_wsl(backend, transports):
    r = await backend._ssh("nvidia-smi")
    assert transports.used == "native" and r.out.startswith("GPU-")


async def test_demotion_is_sticky(backend, transports):
    transports.native = AUTH_FAIL
    await backend._ssh("nvidia-smi")            # demotes
    transports.native = GOOD                    # key "fixed", but window is open
    transports.used = None
    await backend._ssh("nvidia-smi")
    assert transports.used == "wsl", "must not re-attempt native every poll"


async def test_native_resumes_after_the_retry_window(backend, transports):
    transports.native = AUTH_FAIL
    await backend._ssh("nvidia-smi")
    state = ssh_mod.native_state()
    state.demoted_until = time.monotonic() - 1   # window lapsed
    transports.native = GOOD
    transports.used = None
    await backend._ssh("nvidia-smi")
    assert transports.used == "native", "a fixed key must resume without a restart"


async def test_native_disabled_uses_wsl_directly(tmp_path, transports):
    b = _backend(tmp_path, spark_ssh_native=False)
    await b._ssh("nvidia-smi")
    assert transports.used == "wsl"
