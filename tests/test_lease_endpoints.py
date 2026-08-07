"""Lease behaviour at the REST layer — the full app via TestClient.

Pins the fixes for two real bugs found in review:
* `GET /api/leases?unit=` (what an httpx client sends for `params={"unit": None}`)
  must mean "all units", not "units named ''".
* `/api/load` and `/api/unload` must honour a non-preemptible lease — otherwise
  the /v1 gate is hollow: chat traffic gets a 409 while anyone can load a
  different model right over the held unit.
"""
import time
from typing import Optional

import pytest
from fastapi.testclient import TestClient

import llmconfig.lane as lane_mod
import llmconfig.main as main_mod
from llmconfig.config import Settings
from llmconfig.gpu import GpuInfo
from llmconfig.schemas import GpuOut, Job, LaneStatus, StatusResponse


@pytest.fixture
def client(monkeypatch, tmp_path):
    s = Settings(
        _env_file=None,
        monitor_enabled=False,
        idle_unload_enabled=False,
        registry_path=tmp_path / "reg.yaml",
        gpu_uuid="GPU-x",
    )
    monkeypatch.setattr(main_mod, "get_settings", lambda: s)

    async def fake_query_gpu(set_=None, uuid=None, **kw):  # keep unit.status() off nvidia-smi
        return GpuInfo(found=True, uuid=uuid or "GPU-x", total_mb=24576,
                       used_mb=400, free_mb=24176)

    monkeypatch.setattr(lane_mod, "query_gpu", fake_query_gpu)

    app = main_mod.create_app()

    # /api/load and /api/unload must never reach the real backends in these tests.
    def fake_load(req):
        return Job(id="fake-job", kind=f"load:{req.lane}:{req.server}:{req.model}",
                   state="running", created_at=time.time())

    async def fake_unload(req):
        st = LaneStatus(id=req.lane, name=req.lane, owner="free",
                        ollama_up=False, vllm_up=False, gpu=GpuOut(found=False))
        return StatusResponse(owner="free", ollama_up=False, vllm_up=False,
                              gpu=GpuOut(found=False), lanes=[st])

    monkeypatch.setattr(app.state.orch, "load", fake_load)
    monkeypatch.setattr(app.state.orch, "unload", fake_unload)

    with TestClient(app) as c:
        yield c


def _claim(c, holder="alice", preemptible=False, **kw) -> str:
    r = c.post("/api/leases", json={"unit": "primary", "holder": holder,
                                    "preemptible": preemptible, **kw})
    assert r.status_code in (200, 201), r.text
    return r.json()["lease"]["id"]


# --------------------------------------------------------------------------- #
# list: the empty-unit-param bug
# --------------------------------------------------------------------------- #
def test_lease_list_with_empty_unit_param_returns_all(client):
    """`params={"unit": None}` reaches the server as `?unit=` — it must not
    silently filter every lease out (this is what `llmconfig lease list` sends)."""
    _claim(client)
    assert len(client.get("/api/leases").json()["leases"]) == 1
    assert len(client.get("/api/leases?unit=&active=false").json()["leases"]) == 1


def test_lease_list_with_a_real_unit_still_filters(client):
    _claim(client)
    assert len(client.get("/api/leases?unit=primary").json()["leases"]) == 1
    assert client.get("/api/leases?unit=companion").json()["leases"] == []


# --------------------------------------------------------------------------- #
# /api/load and /api/unload honour non-preemptible leases
# --------------------------------------------------------------------------- #
LOAD = {"server": "ollama", "model": "qwen3:4b", "lane": "primary"}


def test_api_load_refused_under_foreign_non_preemptible_lease(client):
    _claim(client, holder="nightly-eval", preemptible=False)
    r = client.post("/api/load", json=LOAD)
    assert r.status_code == 409
    d = r.json()["detail"]
    assert d["error"] == "lease_held" and "nightly-eval" in d["message"]


def test_api_load_allowed_for_the_holder_via_header(client):
    lid = _claim(client, holder="nightly-eval", preemptible=False)
    r = client.post("/api/load", json=LOAD, headers={"X-LLM-Lease": lid})
    assert r.status_code == 200 and r.json()["id"] == "fake-job"


def test_api_load_allowed_under_a_preemptible_lease(client):
    """Preemptible leases stay advisory on these endpoints, matching /v1."""
    _claim(client, holder="alice", preemptible=True)
    assert client.post("/api/load", json=LOAD).status_code == 200


def test_api_unload_refused_then_allowed_with_header(client):
    lid = _claim(client, holder="nightly-eval", preemptible=False)
    assert client.post("/api/unload", json={"lane": "primary"}).status_code == 409
    r = client.post("/api/unload", json={"lane": "primary"},
                    headers={"X-LLM-Lease": lid})
    assert r.status_code == 200


def test_api_load_free_unit_unaffected(client):
    assert client.post("/api/load", json=LOAD).status_code == 200


# --------------------------------------------------------------------------- #
# renew/revoke bodies are OPTIONAL (2026-08-07)
#
# Every field of both request models has a default, and requiring the body
# produced a silent failure mode: a bodyless renew 422s, a tolerant client
# reads 422 as transient API trouble and reports "still ours", and the lease
# quietly expires at TTL with nothing in either log. An entire embedding
# backfill ran leaseless this way.
# --------------------------------------------------------------------------- #
def test_lease_renew_with_no_body(client):
    lid = _claim(client)
    r = client.post(f"/api/leases/{lid}/renew")          # no body at all
    assert r.status_code == 200, r.text
    assert r.json()["renew_count"] == 1
    assert r.json()["state"] == "active"


def test_lease_renew_with_body_still_sets_ttl(client):
    lid = _claim(client)
    r = client.post(f"/api/leases/{lid}/renew", json={"ttl_s": 300})
    assert r.status_code == 200, r.text
    assert r.json()["ttl_s"] == 300


def test_lease_revoke_with_no_body(client):
    lid = _claim(client)
    r = client.post(f"/api/leases/{lid}/revoke")         # defaults: operator/admin
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["state"] == "revoked"
    assert body["revoked_by"] == "operator" and body["revoked_reason"] == "admin"


# --------------------------------------------------------------------------- #
# Lane.canonical_model — a lease claimed under a vLLM ALIAS must govern (and be
# kept alive by) the lane's SERVED name (2026-08-07)
#
# /api/load takes the alias while residency reports the served name. Without
# folding, the unused-reaper compared `harrier-embed` against `harrier-oss-06b`,
# concluded the leased model was absent, and revoked a live lease "unused" at
# ~305 s — while that model was loaded and serving the very holder it belonged
# to. SparkUnit always folded; Lane (and SlotLane) now do too.
# --------------------------------------------------------------------------- #
def test_lane_canonical_model_folds_alias_and_served_name(client):
    from llmconfig.schemas import VllmAliasEntry
    lane = client.app.state.orch.units["primary"]
    lane.registry.upsert(VllmAliasEntry(alias="harrier-embed",
                                        served_name="harrier-oss-06b"))
    assert lane.canonical_model("harrier-embed") == "harrier-embed"
    assert lane.canonical_model("harrier-oss-06b") == "harrier-embed"
    # Ollama tags and unknown names pass through unchanged.
    assert lane.canonical_model("qwen2.5:1.5b") == "qwen2.5:1.5b"


def test_lane_lease_claimed_by_served_name_stores_the_alias(client):
    """_canon at claim time now folds on lanes: a claim naming the SERVED name
    lands on the same key as one naming the alias, so the two can never split
    into a lease that fails to shield its own model."""
    from llmconfig.schemas import VllmAliasEntry
    lane = client.app.state.orch.units["primary"]
    lane.registry.upsert(VllmAliasEntry(alias="harrier-embed",
                                        served_name="harrier-oss-06b"))
    r = client.post("/api/leases", json={"unit": "primary", "holder": "embedder",
                                         "model": "harrier-oss-06b"})
    assert r.status_code == 201, r.text
    assert r.json()["lease"]["model"] == "harrier-embed"


# --------------------------------------------------------------------------- #
# server/unit kind mismatch is a clear 400, not a confusing job failure
# --------------------------------------------------------------------------- #
def test_api_load_rejects_spark_server_on_a_gpu_lane(client):
    r = client.post("/api/load", json={"server": "spark", "model": "x", "lane": "primary"})
    assert r.status_code == 400
    assert "'ollama' or 'vllm'" in r.json()["detail"]
