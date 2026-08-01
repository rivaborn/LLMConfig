"""Runtime reload: catalogs re-read in place, structural settings refused.

The regression these guard is the one that motivated the feature — a cluster
recipe edited on disk stayed invisible until the app was restarted, because
every registry is loaded once from `__init__` and the cluster one has no write
API.
"""

from llmconfig.config import Settings
from llmconfig.registry import Registry, SparkRegistry
from llmconfig.reload import (
    SECRET_FIELDS,
    is_structural,
    reload_catalogs,
    reload_settings,
)
from llmconfig.schemas import VllmAliasEntry


class _FakeUnit:
    def __init__(self, registry):
        self.registry = registry


class _FakeOrch:
    """Only the attributes `reload_catalogs` reaches for."""

    def __init__(self, units=None, cluster_registry=None):
        self.units = units or {}
        self.cluster_registry = cluster_registry
        self.defaults = None
        self.pins = None
        self.group_placements = None


def _write_cluster(path, alias, extra_args=None):
    rows = [
        f"- alias: {alias}",
        f"  recipe: /recipes/{alias}.yaml",
        f"  served_name: {alias}",
        "  tp: 2",
        "  min_nodes: 2",
        "  max_nodes: 2",
        "  status: ok",
    ]
    if extra_args:
        rows.append("  extra_args:")
        rows.extend(f"  - '{a}'" for a in extra_args)
    path.write_text("models:\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_cluster_registry_edit_is_picked_up_without_restart(tmp_path):
    """The motivating case: an edited cluster catalog, no restart."""
    path = tmp_path / "spark_cluster_models.yaml"
    _write_cluster(path, "ds4")
    reg = SparkRegistry(path, default_path=tmp_path / "missing-default.yaml")
    assert reg.get("ds4").extra_args == []

    # Someone hand-edits the file while the app is running.
    _write_cluster(path, "ds4", extra_args=["--max-model-len", "262144"])
    assert reg.get("ds4").extra_args == [], "in-memory copy must be stale until reload"

    out = reload_catalogs(_FakeOrch(cluster_registry=reg), registry=None)

    assert out["cluster_registry"] == 1
    assert reg.get("ds4").extra_args == ["--max-model-len", "262144"]


def test_unit_registries_reload_and_shared_holder_is_not_double_counted(tmp_path):
    """The primary lane shares the app's Registry instance — report it once."""
    vllm_path = tmp_path / "vllm_models.yaml"
    registry = Registry(vllm_path)
    registry.upsert(VllmAliasEntry(alias="zzz", served_name="zzz-x", managed_by="registry"))

    spark_path = tmp_path / "spark_models_spark1.yaml"
    _write_cluster(spark_path, "solo")
    spark_reg = SparkRegistry(spark_path, default_path=tmp_path / "missing.yaml")

    orch = _FakeOrch(units={
        "primary": _FakeUnit(registry),        # same object as the app's registry
        "spark1": _FakeUnit(spark_reg),
    })
    out = reload_catalogs(orch, registry=registry)

    assert "vllm_registry" in out
    assert "unit:primary" not in out, "shared holder must be de-duplicated by identity"
    assert out["unit:spark1"] == 1


def test_unit_without_registry_is_skipped(tmp_path):
    """Not every unit owns a catalog; a registry-less unit must not break the sweep."""
    class _Bare:
        kind = "lane"

    out = reload_catalogs(_FakeOrch(units={"bare": _Bare()}), registry=None)
    assert out == {}


def test_group_sharing_the_cluster_registry_keeps_the_stable_label(tmp_path):
    """A SparkGroup's `.registry` IS the cluster registry — label it once, stably.

    Regression from the live run on 2026-08-01: the unit sweep reached the shared
    object first, so the cluster catalog was reported as `unit:spark1_spark2` and
    `cluster_registry` vanished from the report — and only when a group happened
    to exist, so the report's shape depended on fabric state.
    """
    path = tmp_path / "spark_cluster_models.yaml"
    _write_cluster(path, "ds4")
    cluster = SparkRegistry(path, default_path=tmp_path / "missing.yaml")

    orch = _FakeOrch(units={"spark1_spark2": _FakeUnit(cluster)}, cluster_registry=cluster)
    out = reload_catalogs(orch, registry=None)

    assert out["cluster_registry"] == 1
    assert "unit:spark1_spark2" not in out


def test_structural_fields_are_refused_and_runtime_fields_applied():
    settings = Settings()

    # A runtime-safe scalar and a structural one, both diverging from .env/defaults.
    settings.swap_wait_timeout_s = settings.swap_wait_timeout_s + 123.0
    settings.gpu_uuid = "GPU-deadbeef-not-the-real-card"

    applied, restart_required = reload_settings(settings)

    assert "swap_wait_timeout_s" in applied
    assert settings.swap_wait_timeout_s == Settings().swap_wait_timeout_s

    assert "gpu_uuid" in restart_required, "a lane is BUILT from this — never hot-swap it"
    assert settings.gpu_uuid == "GPU-deadbeef-not-the-real-card", "must be left alone"


def test_secrets_are_redacted_in_the_report():
    settings = Settings()
    settings.llmconfig_api_key = "not-the-real-key"

    applied, restart_required = reload_settings(settings)
    report = {**applied, **restart_required}

    assert "llmconfig_api_key" in report
    assert "not-the-real-key" not in report["llmconfig_api_key"]


def test_no_diff_reports_nothing():
    applied, restart_required = reload_settings(Settings())
    assert applied == {} and restart_required == {}


def test_structural_classification():
    # Whole companion lane is structural by prefix; unknown-but-scalar is not.
    assert is_structural("companion_vllm_slots")
    assert is_structural("spark_fabric_links")
    assert is_structural("llmconfig_port")
    assert not is_structural("spark_mem_headroom")
    assert not is_structural("idle_unload_after_min")
    # Secrets are reportable, and reportable means they must be redacted.
    assert SECRET_FIELDS
