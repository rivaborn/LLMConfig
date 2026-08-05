"""GET /api/models on a SlotLane (the companion lane).

The bug this pins: the route fell through to the plain-Lane branch and called
`ln.ollama.list_models()`. A SlotLane has neither `.ollama` nor `.vllm` — it
owns one backend per configured slot — so it raised AttributeError, which the
route swallowed into `ollama_error`/`vllm_error`. Both lists came back empty,
the UI does not surface those fields, and the companion dropdowns were
therefore *always* empty with no visible cause (found 2026-08-05).

These drive the real route through the real app; a hand-rolled copy of the
branch would pass while the endpoint stayed broken.
"""
import pytest
from fastapi.testclient import TestClient

import llmconfig.lane as lane_mod
import llmconfig.main as main_mod
from llmconfig.config import Settings
from llmconfig.gpu import GpuInfo


@pytest.fixture
def client(monkeypatch, tmp_path):
    s = Settings(
        _env_file=None,
        monitor_enabled=False,
        idle_unload_enabled=False,
        registry_path=tmp_path / "reg.yaml",
        gpu_uuid="GPU-x",
        companion_enabled=True,
        companion_vllm_enabled=True,
        companion_gpu_uuid="GPU-C",
        companion_registry_path=tmp_path / "c.yaml",
        # Two slots; the registry seeded below deliberately holds a third alias
        # with no slot, which must NOT be offered.
        companion_vllm_slots="surya2=11437:4000,qwen25-relay=11438:3000",
    )
    monkeypatch.setattr(main_mod, "get_settings", lambda: s)

    async def fake_query_gpu(set_=None, uuid=None, **kw):
        return GpuInfo(found=True, uuid=uuid or "GPU-x", total_mb=8192,
                       used_mb=400, free_mb=7792)

    monkeypatch.setattr(lane_mod, "query_gpu", fake_query_gpu)

    app = main_mod.create_app()
    comp = app.state.orch.lane("companion")

    # Seed the companion registry: both slot aliases plus one that has no slot.
    from llmconfig.schemas import VllmAliasEntry
    for alias, served in (("surya2", "surya-ocr-2"),
                          ("qwen25-relay", "qwen2.5-1.5b"),
                          ("vl32", "qwen2.5-vl-32b")):
        comp.registry.upsert(VllmAliasEntry(alias=alias, served_name=served))

    # surya2 is serving; the other slot is idle.
    async def served_for(alias):
        return "surya-ocr-2" if alias == "surya2" else None

    for alias, backend in comp.backends.items():
        monkeypatch.setattr(backend, "served",
                            (lambda a: (lambda: served_for(a)))(alias))

    with TestClient(app) as c:
        yield c


def _companion(client):
    r = client.get("/api/models?lane=companion")
    assert r.status_code == 200
    return r.json()


def test_companion_dropdown_is_not_empty(client):
    """The regression itself."""
    d = _companion(client)
    assert d["vllm"], "companion lane must offer its slot aliases"


def test_no_attribute_error_leaks_into_the_error_fields(client):
    d = _companion(client)
    assert "AttributeError" not in (d.get("vllm_error") or "")
    assert "AttributeError" not in (d.get("ollama_error") or "")


def test_only_slot_aliases_are_offered(client):
    """`vl32` is in the registry but has no slot — `_load` would refuse it."""
    aliases = [a["alias"] for a in _companion(client)["vllm"]]
    assert "surya2" in aliases and "qwen25-relay" in aliases
    assert "vl32" not in aliases


def test_loaded_flag_tracks_the_served_name(client):
    by = {a["alias"]: a for a in _companion(client)["vllm"]}
    assert by["surya2"]["loaded"] is True
    assert by["qwen25-relay"]["loaded"] is False


def test_ollama_is_reported_unavailable_rather_than_crashing(client):
    d = _companion(client)
    assert d["ollama"] == []
    assert "not available" in (d.get("ollama_error") or "")


def test_primary_lane_still_lists_normally(client):
    """The new branch must not shadow the plain-Lane path."""
    r = client.get("/api/models?lane=primary")
    assert r.status_code == 200


def test_slotlane_still_has_no_lane_backends():
    """Guard the assumption: if SlotLane grows .ollama/.vllm, revisit the branch."""
    from llmconfig.slot_lane import SlotLane
    assert not hasattr(SlotLane, "ollama")
    assert not hasattr(SlotLane, "vllm")
