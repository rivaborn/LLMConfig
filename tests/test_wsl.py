"""WslKeepalive lifecycle — with a fake Popen so no real wsl.exe is spawned."""
import llmconfig.wsl as wsl_mod
from llmconfig.config import Settings
from llmconfig.wsl import WslKeepalive


class FakePopen:
    instances: list["FakePopen"] = []

    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        self._returncode = None  # None == still running
        self.terminated = False
        FakePopen.instances.append(self)

    def poll(self):
        return self._returncode

    def terminate(self):
        self.terminated = True
        self._returncode = -15


def _patch(monkeypatch):
    FakePopen.instances = []
    monkeypatch.setattr(wsl_mod.subprocess, "Popen", FakePopen)
    return WslKeepalive(Settings(wsl_distro="Ubuntu-24.04", wsl_user="folar"))


def test_ensure_spawns_sleep_infinity(monkeypatch):
    ka = _patch(monkeypatch)
    assert ka.alive() is False
    assert ka.ensure() is True
    assert ka.alive() is True
    assert len(FakePopen.instances) == 1
    argv = FakePopen.instances[0].argv
    assert argv[:5] == ["wsl.exe", "-d", "Ubuntu-24.04", "-u", "folar"]
    assert argv[-2:] == ["sleep", "infinity"]


def test_ensure_is_idempotent_while_alive(monkeypatch):
    ka = _patch(monkeypatch)
    ka.ensure()
    ka.ensure()
    ka.ensure()
    assert len(FakePopen.instances) == 1, "ensure() must not spawn a second keepalive"


def test_ensure_respawns_after_death(monkeypatch):
    ka = _patch(monkeypatch)
    ka.ensure()
    FakePopen.instances[0]._returncode = 0  # the process died
    assert ka.alive() is False
    ka.ensure()
    assert len(FakePopen.instances) == 2, "a dead keepalive must be replaced"


def test_stop_terminates(monkeypatch):
    ka = _patch(monkeypatch)
    ka.ensure()
    proc = FakePopen.instances[0]
    ka.stop()
    assert proc.terminated is True
    assert ka.alive() is False


def test_ensure_handles_missing_wsl(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("wsl.exe")

    monkeypatch.setattr(wsl_mod.subprocess, "Popen", boom)
    ka = WslKeepalive(Settings(wsl_distro="d", wsl_user="u"))
    assert ka.ensure() is False  # off-box: degrade, don't crash
    assert ka.alive() is False
