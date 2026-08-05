"""The runtime image is pinned by digest and passed as `sparkrun run --image`.

Without this, sparkrun resolves the recipe's `container:` through
build-and-copy.sh, which pulls the MOVING `eugr/spark-vllm:latest`. On
2026-08-04 that cost ~14 min of image pull mid-load and had left the four
nodes on four different digests. Verified on spark4 the same evening: with
`--image <digest>`, image prep went 14 min -> 0.4 s and nothing downloaded.
"""
import pytest

import llmconfig.backends.spark as spark_mod
from llmconfig.config import Settings, SparkConfig
from llmconfig.proc import CmdResult
from llmconfig.registry import SparkRegistry

DIGEST = ("eugr/spark-vllm@sha256:"
          "1d335d4fb3d1c5dce6e79f87a4019a04e98fa9ceb7b894d50d413159382ab6c6")


@pytest.fixture
def sent(monkeypatch):
    """Capture the sparkrun command line the backend builds."""
    box = type("Box", (), {"cmd": ""})()

    async def fake_run_wsl(command, *, login=True, timeout=30.0, settings=None):
        box.cmd = command
        return CmdResult(0, "ok", "")

    async def fake_run_ssh(user, host, command, *, timeout=20.0, settings=None):
        return CmdResult(0, "ok", "")

    monkeypatch.setattr(spark_mod, "run_wsl", fake_run_wsl)
    monkeypatch.setattr(spark_mod, "run_ssh", fake_run_ssh)
    return box


def _backend(tmp_path, **over):
    cfg = SparkConfig(
        id="spark4", name="n", host="192.168.1.53", ssh_user="u", api_port=8000,
        registry_path=tmp_path / "r.yaml",
    )
    return spark_mod.SparkBackend(Settings(**over), cfg, SparkRegistry(cfg.registry_path))


def test_default_settings_pin_the_image_by_digest():
    """The shipped default must be a DIGEST, never a moving tag."""
    img = Settings().spark_image
    assert img.startswith("eugr/spark-vllm@sha256:"), img
    assert ":latest" not in img


async def test_run_passes_image_flag(tmp_path, sent):
    b = _backend(tmp_path, spark_image=DIGEST)
    await b.run_recipe("@eugr/qwen3.5-122b-int4-autoround", served="qwen35-122b_8_3_26", port=8000)
    assert f"--image {DIGEST}" in sent.cmd
    assert "--served-model-name qwen35-122b_8_3_26" in sent.cmd


async def test_blank_setting_omits_the_flag(tmp_path, sent):
    """Escape hatch: fall back to whatever the recipe's `container:` resolves to."""
    b = _backend(tmp_path, spark_image="")
    await b.run_recipe("@eugr/qwen3.5-122b-int4-autoround", served="x", port=8000)
    assert "--image" not in sent.cmd


async def test_multi_node_run_is_pinned_too(tmp_path, sent):
    """A tp job must not silently use a different image from a solo load."""
    b = _backend(tmp_path, spark_image=DIGEST)
    await b.run_recipe("@eugr/qwen3.5-122b-int4-autoround", served="x", port=8000,
                hosts=["192.168.1.52", "192.168.1.53"], tp=2)
    assert f"--image {DIGEST}" in sent.cmd
