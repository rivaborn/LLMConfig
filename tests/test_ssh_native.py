"""Native OpenSSH transport for Spark nodes (no WSL in the telemetry path)."""
import pytest

import llmconfig.ssh as ssh_mod
from llmconfig.config import Settings
from llmconfig.ssh import DEFAULT_SSH_OPTS, resolve_key, ssh_argv

REMOTE = "nvidia-smi --query-gpu=name --format=csv,noheader; echo '#MEM#'; head -1 /proc/meminfo"


def test_remote_command_is_a_single_argv_element():
    """No local shell may ever see it — that is what broke quoting under wsl.exe."""
    argv = ssh_argv("u", "h", REMOTE)
    assert argv[-1] == REMOTE, "the command must survive verbatim"
    assert argv[-2] == "u@h"
    assert sum(1 for a in argv if a == REMOTE) == 1


def test_default_options_are_headless_safe():
    argv = ssh_argv("u", "h", "true")
    opts = [argv[i + 1] for i, a in enumerate(argv) if a == "-o"]
    assert set(DEFAULT_SSH_OPTS) <= set(opts)
    # IdentitiesOnly matters: without it ssh offers default keys first and a node
    # with MaxAuthTries can reject us before reaching the right one.
    assert "IdentitiesOnly=yes" in opts
    assert "BatchMode=yes" in opts, "must never prompt under a scheduled task"


def test_key_is_passed_when_present(tmp_path):
    key = tmp_path / "id_ed25519_sparkctl"
    key.write_text("x")
    argv = ssh_argv("u", "h", "true", key=str(key))
    assert "-i" in argv and argv[argv.index("-i") + 1] == str(key)


def test_missing_key_degrades_to_ssh_defaults(tmp_path):
    """A not-yet-provisioned key must not hard-fail every probe."""
    s = Settings(spark_ssh_key=str(tmp_path / "absent"))
    assert resolve_key(s) is None
    argv = ssh_argv("u", "h", "true", key=resolve_key(s))
    assert "-i" not in argv


def test_blank_key_setting_is_allowed():
    assert resolve_key(Settings(spark_ssh_key="")) is None


def test_resolve_key_expands_user(tmp_path, monkeypatch):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "id_ed25519_sparkctl").write_text("x")
    got = resolve_key(Settings(spark_ssh_key="~/.ssh/id_ed25519_sparkctl"))
    assert got is not None and got.endswith("id_ed25519_sparkctl")


async def test_run_ssh_does_not_touch_wsl(monkeypatch):
    seen = {}

    async def fake_run_argv(argv, timeout=30.0, env=None):
        seen["argv"] = argv
        from llmconfig.proc import CmdResult
        return CmdResult(0, "", "")

    monkeypatch.setattr(ssh_mod, "run_argv", fake_run_argv)
    await ssh_mod.run_ssh("u", "h", "true", settings=Settings(spark_ssh_key=""))
    assert not any("wsl" in a.lower() for a in seen["argv"]), "WSL must be out of this path"
    assert seen["argv"][0] in ("ssh", "ssh.exe")
