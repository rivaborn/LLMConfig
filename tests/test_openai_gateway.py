"""OpenAI `/v1` gateway: model resolution, fast-path forward, cold-load streaming,
the non-stream short-circuit, and the model list — all with fake lane backends and
a MockTransport upstream (no nvidia-smi / WSL / real Ollama or vLLM)."""
import json
import time

import httpx
from httpx import ASGITransport

import llmconfig.lane as lane_mod
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from llmconfig.config import Settings
from llmconfig.gpu import GpuInfo
from llmconfig.jobs import JobManager
from llmconfig.lane_state import LaneDefaults
from llmconfig.leases import LeaseManager
from llmconfig.openai_gateway import OpenAIGateway, build_gateway_router
from llmconfig.orchestrator import Orchestrator
from llmconfig.proc import CmdResult
from llmconfig.registry import Registry
from llmconfig.schemas import LaneStatus, LeaseClaimRequest, OllamaModel, ServedModel

GiB = 1024 ** 3


class World:
    def __init__(self):
        self.base = 400
        self.used_mb = 400
        self.ollama: dict[str, tuple[int, int]] = {}
        self.vllm = None
        self.tags = ["qwen3-coder:30b", "llama3:8b"]

    def gpu(self, uuid):
        return GpuInfo(found=True, uuid=uuid, total_mb=24576, used_mb=self.used_mb,
                       free_mb=24576 - self.used_mb)


class FakeOllama:
    def __init__(self, w):
        self.w = w

    async def up(self):
        return True

    async def ensure_running(self, wait_s=20.0):
        return True

    async def list_models(self):
        return [
            OllamaModel(name=t, size_bytes=2 * GiB, loaded=t in self.w.ollama,
                        size_vram_bytes=self.w.ollama.get(t, (0, 0))[1])
            for t in self.w.tags
        ]

    async def loaded(self):
        return [OllamaModel(name=n, size_bytes=s, loaded=True, size_vram_bytes=v)
                for n, (s, v) in self.w.ollama.items()]

    async def loaded_names(self):
        return list(self.w.ollama)

    async def unload_all(self):
        names = list(self.w.ollama)
        self.w.ollama.clear()
        if names:
            self.w.used_mb = self.w.base
        return names

    async def unload(self, m):
        self.w.ollama.pop(m, None)

    async def load(self, m, keep_alive=-1, num_gpu=None, timeout=900.0):
        self.w.ollama = {m: (2 * GiB, 2 * GiB)}
        self.w.used_mb = 2000

    async def block_count(self, m):
        return 32


class FakeVllm:
    def __init__(self, w, reg):
        self.w = w
        self.reg = reg

    async def served(self):
        return self.w.vllm

    async def served_info(self):
        # Mirrors a real relay's /v1/models: name, root (which distinguishes
        # same-named models on different units) and the served context window.
        m = self.w.vllm
        return ServedModel(name=m, root=(f"fake-org/{m}" if m else ""),
                           context_len=32768 if m else 0)

    async def up(self):
        return self.w.vllm is not None

    async def stop(self):
        self.w.vllm = None
        self.w.used_mb = self.w.base

    async def serve(self, alias):
        self.w.vllm = self.reg.served_name(alias)
        self.w.used_mb = 16000
        return CmdResult(0, "", "")

    async def wait_ready(self, served, timeout, on_log=None, alias=None):
        return self.w.vllm == served

    async def journal_tail(self, alias, n=40):
        return ""


class FakeKeepalive:
    def ensure(self):
        return True

    def alive(self):
        return True

    def stop(self):
        pass


def _upstream_app(captured):
    """A stand-in for the real vLLM relay / Ollama OpenAI endpoint. ASGITransport
    routes every backend URL here; `host` carries the port we forwarded to."""
    up = FastAPI()

    async def _handle(request: Request):
        body = await request.json()
        captured.append(f"{request.headers.get('host', '')}{request.url.path}")
        if body.get("stream"):
            async def gen():
                yield b'data: {"choices":[{"delta":{"content":"UPSTREAM_OK"}}]}\n\n'
                yield b"data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return JSONResponse({"marker": "UPSTREAM_OK", "model": body.get("model", "")})

    up.post("/v1/chat/completions")(_handle)
    up.post("/v1/completions")(_handle)
    # pooling runners — same passthrough, no streaming form
    up.post("/v1/embeddings")(_handle)
    up.post("/v1/rerank")(_handle)
    up.post("/v1/score")(_handle)
    return up


def _build(monkeypatch, tmp_path):
    s = Settings(_env_file=None, gpu_uuid="GPU-P", registry_path=tmp_path / "p.yaml",
                 evict_timeout_s=5, poll_interval_s=0.001, vllm_ready_grace_s=1)
    jobs = JobManager()
    orch = Orchestrator(s, Registry(s.registry_path), jobs)
    orch.defaults = LaneDefaults(s, path=tmp_path / "ld.yaml")
    world = World()
    lane = orch.primary
    lane.ollama = FakeOllama(world)
    lane.vllm = FakeVllm(world, lane.registry)
    lane.keepalive = FakeKeepalive()

    async def fake_query_gpu(set_=None, uuid=None):
        return world.gpu(uuid or "GPU-P")

    monkeypatch.setattr(lane_mod, "query_gpu", fake_query_gpu)

    captured: list[str] = []
    leases = LeaseManager(s, orch)
    gateway = OpenAIGateway(orch, jobs, s, leases)
    gateway._http = httpx.AsyncClient(transport=ASGITransport(app=_upstream_app(captured)))

    app = FastAPI()
    app.include_router(build_gateway_router(gateway))
    # Reachable from the lease tests without changing this helper's tuple arity.
    app.state.leases = leases
    return app, orch, jobs, world, captured


def _client(app):
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


async def test_models_lists_vllm_served_names_and_ollama_tags(monkeypatch, tmp_path):
    app, *_ = _build(monkeypatch, tmp_path)
    async with _client(app) as c:
        r = await c.get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    ids = {m["id"] for m in data}
    assert "qwen3-coder-30b" in ids   # a vLLM served_name (from the seeded registry)
    assert "qwen3-coder:30b" in ids   # an Ollama tag
    assert all(m["owned_by"] == "llmconfig" for m in data)
    # each entry carries a backend-tagged `name`; `id` stays the canonical handle
    by_id = {m["id"]: m for m in data}
    assert by_id["qwen3-coder-30b"]["name"] == "qwen3-coder-30b  (vLLM)"
    assert by_id["qwen3-coder:30b"]["name"] == "qwen3-coder:30b  (Ollama)"
    assert all("(vLLM)" in m["name"] or "(Ollama)" in m["name"] for m in data)


async def test_unknown_model_404(monkeypatch, tmp_path):
    app, *_ = _build(monkeypatch, tmp_path)
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json={"model": "does-not-exist", "stream": False})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


async def test_fast_path_vllm_forwards_to_relay(monkeypatch, tmp_path):
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    world.vllm = "qwen3-coder-30b"   # already serving exactly this model
    world.used_mb = 16000
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions",
                         json={"model": "qwen3-coder-30b", "stream": False,
                               "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json()["marker"] == "UPSTREAM_OK"
    assert any(":11437" in u for u in captured), "should forward to the primary vLLM relay"
    assert jobs.list() == [], "fast path must not create a load job"


async def test_fast_path_ollama_forwards_to_ollama(monkeypatch, tmp_path):
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    world.ollama = {"qwen3-coder:30b": (2 * GiB, 2 * GiB)}  # already loaded on Ollama
    world.used_mb = 2000
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions",
                         json={"model": "qwen3-coder:30b", "stream": False,
                               "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert any(":11434" in u for u in captured), "should forward to the lane's Ollama"
    assert jobs.list() == []


async def test_cold_load_streams_progress_then_forwards(monkeypatch, tmp_path):
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    # nothing loaded → a cold vLLM load must run, streaming progress, then forward
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions",
                         json={"model": "qwen3-coder-30b", "stream": True,
                               "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    text = r.text
    assert "⏳" in text, "cold load must stream progress chunks"
    assert "UPSTREAM_OK" in text, "after load it must relay the upstream completion"
    assert "[DONE]" in text
    assert world.vllm == "qwen3-coder-30b"
    assert any(":11437" in u for u in captured)


async def test_gateway_request_touches_lane(monkeypatch, tmp_path):
    # A /v1 request must reset the lane's idle-unload window (Lane.touch), so the
    # idle reaper never counts gateway traffic as inactivity.
    import time

    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    world.vllm = "qwen3-coder-30b"
    world.used_mb = 16000
    lane = orch.primary
    lane.last_activity = time.time() - 3600  # backdated: an hour "idle"
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions",
                         json={"model": "qwen3-coder-30b", "stream": False,
                               "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert time.time() - lane.last_activity < 60, "the request must advance last_activity"


async def test_nonstream_shortcircuits_during_a_different_load(monkeypatch, tmp_path):
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    lane = orch.primary
    # Simulate a *different* model mid-load on the lane.
    other = jobs.create(kind="load:primary:vllm:coder32")
    other.state = "running"
    real_status = lane.status

    async def fake_status(gpu=None):
        st = await real_status(gpu=gpu)
        st.swap_in_progress = True
        st.active_job_id = other.id
        return st

    monkeypatch.setattr(lane, "status", fake_status)

    async with _client(app) as c:
        r = await c.post("/v1/chat/completions",
                         json={"model": "qwen3-coder-30b", "stream": False,
                               "messages": [{"role": "user", "content": "title?"}]})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "", "must return an empty, immediate 200"
    assert captured == [], "must not forward upstream while a different load is in flight"


# --------------------------------------------------------------------------- #
# Lease admission on the inference path
# --------------------------------------------------------------------------- #
def _claim(app, holder="alice", unit="primary", **kw):
    return app.state.leases.claim(LeaseClaimRequest(unit=unit, holder=holder, **kw))[0]


def _chat(model="qwen3-coder:30b", **kw):
    return {"model": model, "messages": [{"role": "user", "content": "hi"}], **kw}


async def _load_ollama_tag(world, orch, tag="qwen3-coder:30b"):
    """Make the lane already serve `tag` so requests take the forward fast path."""
    world.ollama = {tag: (8 * GiB, 8 * GiB)}
    world.used_mb = 9000


async def test_unleased_traffic_allowed_when_nothing_is_claimed(monkeypatch, tmp_path):
    """opencode sends no lease header — it must keep working exactly as before."""
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    await _load_ollama_tag(world, orch)
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json=_chat())
    assert r.status_code == 200 and r.json()["marker"] == "UPSTREAM_OK"


async def test_unleased_traffic_allowed_under_a_preemptible_lease(monkeypatch, tmp_path):
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    await _load_ollama_tag(world, orch)
    _claim(app, preemptible=True)
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json=_chat())
    assert r.status_code == 200, "a preemptible lease is advisory for other callers"


async def test_unleased_traffic_refused_under_a_non_preemptible_lease(monkeypatch, tmp_path):
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    await _load_ollama_tag(world, orch)
    _claim(app, holder="nightly-eval", preemptible=False)
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json=_chat())
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "lease_required" and "nightly-eval" in err["message"]
    assert captured == [], "a refused request must never reach the backend"


async def test_valid_lease_header_forwards_and_records_use(monkeypatch, tmp_path):
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    await _load_ollama_tag(world, orch)
    lease = _claim(app, holder="alice", preemptible=False)
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json=_chat(),
                         headers={"X-LLM-Lease": lease.id})
    assert r.status_code == 200 and r.json()["marker"] == "UPSTREAM_OK"
    assert lease.requests == 1 and lease.last_seen_at is not None


async def test_revoked_lease_header_is_refused_and_names_the_new_holder(monkeypatch, tmp_path):
    """Requirement 4 on the inference path: the displaced holder learns in-band."""
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    await _load_ollama_tag(world, orch)
    first = _claim(app, holder="alice", priority=1)
    _claim(app, holder="bob", priority=9)          # preempts alice
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json=_chat(),
                         headers={"X-LLM-Lease": first.id})
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "lease_revoked" and "bob" in err["message"]
    assert err["lease"]["revoked_reason"] == "preempted"
    assert captured == []


async def test_unknown_lease_header_reports_server_restarted(monkeypatch, tmp_path):
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    await _load_ollama_tag(world, orch)
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json=_chat(),
                         headers={"X-LLM-Lease": "a" * 12})
    assert r.status_code == 409
    assert "server_restarted" in r.json()["error"]["message"]


async def test_lease_for_a_different_unit_is_refused(monkeypatch, tmp_path):
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    await _load_ollama_tag(world, orch)
    orch.units["companion"] = orch.units["primary"]  # give the manager a 2nd unit id
    lease = _claim(app, unit="companion")
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json=_chat(),
                         headers={"X-LLM-Lease": lease.id})
    assert r.status_code == 409 and r.json()["error"]["code"] == "lease_wrong_unit"


async def test_expired_lease_still_serves_its_holder_inside_the_grace_window(monkeypatch, tmp_path):
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    await _load_ollama_tag(world, orch)
    lease = _claim(app, holder="alice")
    lease._deadline = time.monotonic() - 1          # lapsed, but within the 30 s grace
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json=_chat(),
                         headers={"X-LLM-Lease": lease.id})
    assert r.status_code == 200, "a slipped renew must not drop the holder mid-session"


async def test_stream_rejection_returns_409_by_default(monkeypatch, tmp_path):
    """The gate decides before any bytes are written, so a real status code is
    available and strictly more correct than an in-band error chunk."""
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    await _load_ollama_tag(world, orch)
    _claim(app, holder="nightly-eval", preemptible=False)
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json=_chat(stream=True))
    assert r.status_code == 409 and r.json()["error"]["code"] == "lease_required"
    assert captured == []


async def test_stream_rejection_can_emit_an_sse_error_chunk(monkeypatch, tmp_path):
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    app.state.leases.s.lease_stream_reject_mode = "sse"
    await _load_ollama_tag(world, orch)
    _claim(app, holder="nightly-eval", preemptible=False)
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json=_chat(stream=True))
    assert r.status_code == 200
    assert "❌" in r.text and "[DONE]" in r.text
    assert captured == []


async def test_refused_request_does_not_touch_the_lane(monkeypatch, tmp_path):
    """A rejected request never reached a backend, so it must not extend the lane's
    idle window (the mirror of test_gateway_request_touches_lane)."""
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    await _load_ollama_tag(world, orch)
    _claim(app, holder="nightly-eval", preemptible=False)
    lane = orch.primary
    lane.last_activity = time.time() - 500
    before = lane.last_activity
    async with _client(app) as c:
        await c.post("/v1/chat/completions", json=_chat())
    assert lane.last_activity == before


async def test_block_unleased_kill_switch_restores_old_behaviour(monkeypatch, tmp_path):
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    app.state.leases.s.lease_block_unleased = False
    await _load_ollama_tag(world, orch)
    _claim(app, holder="nightly-eval", preemptible=False)
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json=_chat())
    assert r.status_code == 200


# --------------------------------------------------------------------------- #
# Pooling endpoints: /v1/embeddings, /v1/rerank, /v1/score
# --------------------------------------------------------------------------- #
async def test_embeddings_fast_path_forwards_to_relay(monkeypatch, tmp_path):
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    world.vllm = "qwen3-coder-30b"
    world.used_mb = 16000
    async with _client(app) as c:
        r = await c.post("/v1/embeddings", json={"model": "qwen3-coder-30b", "input": ["a", "b"]})
    assert r.status_code == 200
    assert r.json()["marker"] == "UPSTREAM_OK"
    assert any(u.endswith("/v1/embeddings") for u in captured), "must hit the embeddings sub-path"
    assert any(":11437" in u for u in captured)
    assert jobs.list() == [], "fast path must not create a load job"


async def test_rerank_and_score_reach_their_own_sub_paths(monkeypatch, tmp_path):
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    world.vllm = "qwen3-coder-30b"
    world.used_mb = 16000
    async with _client(app) as c:
        r1 = await c.post("/v1/rerank",
                          json={"model": "qwen3-coder-30b", "query": "q", "documents": ["d"]})
        r2 = await c.post("/v1/score", json={"model": "qwen3-coder-30b", "text_1": "a", "text_2": "b"})
    assert r1.status_code == 200 and r2.status_code == 200
    assert any(u.endswith("/v1/rerank") for u in captured)
    assert any(u.endswith("/v1/score") for u in captured)


async def test_rerank_against_ollama_is_501_not_a_confusing_404(monkeypatch, tmp_path):
    """Ollama's OpenAI surface has /v1/embeddings but no rerank. Say so plainly."""
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    world.ollama = {"qwen3-coder:30b": (2 * GiB, 2 * GiB)}
    world.used_mb = 2000
    async with _client(app) as c:
        r = await c.post("/v1/rerank",
                         json={"model": "qwen3-coder:30b", "query": "q", "documents": ["d"]})
    assert r.status_code == 501
    assert r.json()["error"]["code"] == "rerank_unsupported"
    assert captured == [], "must not forward a rerank to Ollama"


async def test_embeddings_never_return_an_empty_vector_mid_load(monkeypatch, tmp_path):
    """The chat path answers a mid-load request with an empty 200 so title-gen does
    not hang. Doing that here would write an empty embedding into a vector store —
    silent corruption. Must be an explicit 503 instead."""
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    lane = orch.primary
    other = jobs.create(kind="load:primary:vllm:coder32")
    other.state = "running"
    real_status = lane.status

    async def fake_status(gpu=None):
        st = await real_status(gpu=gpu)
        st.swap_in_progress = True
        st.active_job_id = other.id
        return st

    monkeypatch.setattr(lane, "status", fake_status)
    async with _client(app) as c:
        r = await c.post("/v1/embeddings", json={"model": "qwen3-coder-30b", "input": ["a"]})
    assert r.status_code == 503, "must refuse, not fabricate"
    assert r.json()["error"]["code"] == "unit_busy_loading"
    assert captured == [], "must not forward while a different load is in flight"


async def test_embeddings_unknown_model_404(monkeypatch, tmp_path):
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    async with _client(app) as c:
        r = await c.post("/v1/embeddings", json={"model": "nope-not-a-model", "input": ["a"]})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"
    assert captured == []


async def test_embeddings_respect_a_non_preemptible_lease(monkeypatch, tmp_path):
    """The lease gate is shared with the chat path — pooling traffic must not be a
    way around a non-preemptible claim."""
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    world.vllm = "qwen3-coder-30b"
    world.used_mb = 16000
    _claim(app, holder="someone-else", preemptible=False)
    async with _client(app) as c:
        r = await c.post("/v1/embeddings", json={"model": "qwen3-coder-30b", "input": ["a"]})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "lease_required"
    assert captured == [], "a refused request must never reach the backend"


# --------------------------------------------------------------------------- #
# Multi-model Sparks: route each model to ITS port, and never evict a neighbour
# --------------------------------------------------------------------------- #
from llmconfig.config import SparkConfig                    # noqa: E402
from llmconfig.registry import SparkRegistry                # noqa: E402
from llmconfig.schemas import LoadedModel, SparkModelEntry  # noqa: E402
from llmconfig.spark_unit import SparkUnit                  # noqa: E402

SPARK_HOST = "10.9.9.9"


def _add_spark(orch, tmp_path, settings):
    """Register a real SparkUnit on the orchestrator the gateway already holds."""
    cfg = SparkConfig(
        id="spark1", name="Spark 1", host=SPARK_HOST, ssh_user="u", api_port=8000,
        registry_path=tmp_path / "spark_models_spark1.yaml", load_timeout_s=5,
        max_models=4,
    )
    reg = SparkRegistry(cfg.registry_path)
    reg.upsert(SparkModelEntry(alias="a", recipe="ra", served_name="model-a",
                               mem_fraction=0.4, load_timeout_s=5))
    reg.upsert(SparkModelEntry(alias="b", recipe="rb", served_name="model-b",
                               mem_fraction=0.4, load_timeout_s=5))
    unit = SparkUnit(settings, cfg, reg, orch.jobs)
    orch.units["spark1"] = unit
    return unit


def _spark_status(*pairs):
    """A LaneStatus for a Spark holding (model, port) pairs."""
    models = [LoadedModel(server="spark", model=m, port=p) for m, p in pairs]
    return LaneStatus(
        id="spark1", name="Spark 1", kind="spark", owner="spark" if models else "free",
        ollama_up=False, vllm_up=False,
        loaded=models[0] if models else None, loaded_models=models,
    )


async def test_route_sends_each_model_to_its_own_port(monkeypatch, tmp_path):
    """Every Spark model used to resolve to the unit's single api_base, so a
    request for a model on slot 1 was delivered to slot 0's process — which
    answers 404 'model does not exist' with no hint that routing was wrong."""
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    gw = app.state.gateway if hasattr(app.state, "gateway") else None
    unit = _add_spark(orch, tmp_path, Settings(_env_file=None))
    from llmconfig.openai_gateway import OpenAIGateway
    gw = gw or OpenAIGateway(orch, jobs, Settings(_env_file=None))

    st = _spark_status(("model-a", 8000), ("model-b", 8001))
    default = unit.cfg.api_base

    assert gw._route(unit, "spark", "model-a", st, default) == f"http://{SPARK_HOST}:8000"
    assert gw._route(unit, "spark", "model-b", st, default) == f"http://{SPARK_HOST}:8001"
    # not resident yet -> slot 0, so a cold load still reaches a backend
    assert gw._route(unit, "spark", "model-c", st, default) == default
    # a GPU lane has one occupant; its URL must never be rewritten
    assert gw._route(orch.primary, "vllm", "anything", st, "http://relay") == "http://relay"


async def test_resident_reads_the_list_not_the_primary_occupant(monkeypatch, tmp_path):
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    from llmconfig.openai_gateway import OpenAIGateway
    gw = OpenAIGateway(orch, jobs, Settings(_env_file=None))

    st = _spark_status(("model-a", 8000), ("model-b", 8001))

    # `loaded` names only model-a; model-b is resident all the same.
    assert st.loaded.model == "model-a"
    assert gw._resident(st, "model-b") is not None
    assert gw._resident(st, "model-b").port == 8001
    assert gw._resident(st, "model-c") is None


async def test_spark_does_not_shortcircuit_on_a_different_in_flight_load(monkeypatch, tmp_path):
    """On a GPU lane a different load means this model is about to be evicted, so
    bailing is right. On a Spark the other load is for another slot entirely and
    says nothing about this request."""
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    unit = _add_spark(orch, tmp_path, Settings(_env_file=None))
    from llmconfig.openai_gateway import OpenAIGateway
    gw = OpenAIGateway(orch, jobs, Settings(_env_file=None))

    other = jobs.create(kind="load:spark1:spark:a")
    other.state = "running"
    st = _spark_status()
    st.swap_in_progress, st.active_job_id = True, other.id

    monkeypatch.setattr(orch, "load", lambda req: jobs.create(kind="load:spark1:spark:b"))
    job, short_circuit = gw._ensure_load_job(unit, st, "load:spark1:spark:b", "spark", "b",
                                             stream=False)
    assert short_circuit is False, "a Spark must queue, not bail"
    assert job is not None

    # the GPU lane keeps the old behaviour
    st2 = _spark_status()
    st2.swap_in_progress, st2.active_job_id = True, other.id
    job2, sc2 = gw._ensure_load_job(orch.primary, st2, "load:primary:vllm:x", "vllm", "x",
                                    stream=False)
    assert sc2 is True and job2 is None, "a GPU lane still short-circuits"


async def test_cold_load_reroutes_to_the_slot_the_model_landed_on(monkeypatch, tmp_path):
    """The pre-load backend is always slot 0 — usually some OTHER model's port on
    a multi-model node. Until the re-route, the forward after a cold load went to
    slot 0 and the co-resident answered 404 for a model that had just loaded."""
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    unit = _add_spark(orch, tmp_path, Settings(_env_file=None))
    from llmconfig.openai_gateway import OpenAIGateway
    gw = OpenAIGateway(orch, jobs, Settings(_env_file=None))

    # After the load, residency says model-b took slot 1.
    async def status_after_load(gpu=None):
        return _spark_status(("model-a", 8000), ("model-b", 8001))
    monkeypatch.setattr(unit, "status", status_after_load)

    reroute = gw._reroute(unit, "spark", "model-b", unit.cfg.api_base)
    assert await reroute() == f"http://{SPARK_HOST}:8001", \
        "the post-load forward must go to the slot the load actually took"

    # A model that never became resident keeps the default (cold-load fallback).
    reroute_missing = gw._reroute(unit, "spark", "model-c", unit.cfg.api_base)
    assert await reroute_missing() == unit.cfg.api_base


# --------------------------------------------------------------------------- #
# Auto-placement (no X-LLM-Lane header, or X-LLM-Lane: auto)
# --------------------------------------------------------------------------- #
from types import SimpleNamespace                            # noqa: E402

from llmconfig.monitor import Monitor                        # noqa: E402
from llmconfig.placement import Placer                       # noqa: E402


def _auto_build(monkeypatch, tmp_path, *, spark_models=(), **settings_overrides):
    """The standard _build plus a Spark unit and a wired Placer, so no-header
    requests genuinely have a CHOICE (two candidates disables the sole-pin).

    The Spark's status is ALWAYS stubbed (default: empty node) — the real
    SparkUnit.status would probe an unroutable host and spawn real wsl.exe
    telemetry from inside the placer's sweep, hanging the suite.
    """
    app, orch, jobs, world, captured = _build(monkeypatch, tmp_path)
    s = Settings(_env_file=None, **settings_overrides)
    unit = _add_spark(orch, tmp_path, s)
    st = _spark_status(*spark_models)

    async def spark_status(gpu=None):
        return st
    monkeypatch.setattr(unit, "status", spark_status)
    monkeypatch.setattr(unit, "_refresh_gpu_soon", lambda: None)

    leases = app.state.leases
    monitor = SimpleNamespace(util_for=lambda uuid: None)
    placer = Placer(s, orch, leases, monitor)
    # rebuild the gateway with the placer, reusing the captured upstream
    gateway = OpenAIGateway(orch, jobs, s, leases, placer)
    gateway._http = httpx.AsyncClient(transport=ASGITransport(app=_upstream_app(captured)))
    app2 = FastAPI()
    app2.include_router(build_gateway_router(gateway))
    app2.state.leases = leases
    return app2, orch, jobs, world, captured, unit, placer


async def test_no_header_places_on_the_resident_spark(monkeypatch, tmp_path):
    """The feature: a client that names only a model reaches the unit serving it,
    even though 'primary' never had it — and learns where via X-LLM-Unit."""
    app, orch, jobs, world, captured, unit, placer = _auto_build(
        monkeypatch, tmp_path, spark_models=(("model-a", 8000),))

    async with _client(app) as c:
        r = await c.post("/v1/chat/completions",
                         json={"model": "model-a",
                               "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    assert r.headers.get("x-llm-unit") == "spark1"
    assert captured and ":8000" in captured[-1], "forwarded to the spark's slot"


async def test_auto_sentinel_equals_absent_and_explicit_pin_still_pins(monkeypatch, tmp_path):
    app, orch, jobs, world, captured, unit, placer = _auto_build(
        monkeypatch, tmp_path, spark_models=(("model-a", 8000),))

    async with _client(app) as c:
        r_auto = await c.post("/v1/chat/completions", headers={"X-LLM-Lane": "auto"},
                              json={"model": "model-a",
                                    "messages": [{"role": "user", "content": "hi"}]})
        r_pin = await c.post("/v1/chat/completions", headers={"X-LLM-Lane": "primary"},
                             json={"model": "model-a",
                                   "messages": [{"role": "user", "content": "hi"}]})
    assert r_auto.status_code == 200 and r_auto.headers.get("x-llm-unit") == "spark1"
    assert r_pin.status_code == 404, "an explicit pin must NOT be second-guessed"


async def test_auto_with_valid_lease_pins_to_the_lease_unit(monkeypatch, tmp_path):
    """A lease is a claim on a unit; auto must not scatter its holder elsewhere."""
    app, orch, jobs, world, captured, unit, placer = _auto_build(
        monkeypatch, tmp_path, spark_models=(("model-a", 8000),))
    lease, _ = app.state.leases.claim(
        LeaseClaimRequest(unit="spark1", holder="alice", model="a"))

    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", headers={"X-LLM-Lease": lease.id},
                         json={"model": "model-a",
                               "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200 and r.headers.get("x-llm-unit") == "spark1"


async def test_auto_not_found_stays_404(monkeypatch, tmp_path):
    app, orch, jobs, world, captured, unit, placer = _auto_build(monkeypatch, tmp_path)
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions",
                         json={"model": "no-such-model",
                               "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


async def test_kill_switch_restores_primary_default(monkeypatch, tmp_path):
    """AUTO_PLACE_ENABLED=false → a no-header request behaves exactly as before:
    resolved against primary only, so a spark-only model 404s."""
    app, orch, jobs, world, captured, unit, placer = _auto_build(
        monkeypatch, tmp_path, auto_place_enabled=False,
        spark_models=(("model-a", 8000),))

    async with _client(app) as c:
        r = await c.post("/v1/chat/completions",
                         json={"model": "model-a",
                               "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404


async def test_models_union_under_auto_and_per_lane_with_header(monkeypatch, tmp_path):
    app, orch, jobs, world, captured, unit, placer = _auto_build(monkeypatch, tmp_path)
    world.ollama_tags = ["qwen3:4b"]

    async with _client(app) as c:
        union = (await c.get("/v1/models")).json()
        pinned = (await c.get("/v1/models", headers={"X-LLM-Lane": "spark1"})).json()
    union_ids = {m["id"] for m in union["data"]}
    assert "model-a" in union_ids and "model-b" in union_ids, "spark catalog present"
    pinned_ids = {m["id"] for m in pinned["data"]}
    # the spark registry seeds the packaged catalog too; the point is that the
    # pinned list has ONLY spark entries — nothing from primary's ollama/vllm.
    assert {"model-a", "model-b"} <= pinned_ids
    assert "qwen3:4b" not in pinned_ids, "explicit header keeps the per-unit list"


# --------------------------------------------------------------------------- #
# X-LLM-Hold — a static-config client leases the model it is using
# --------------------------------------------------------------------------- #
async def test_hold_header_claims_and_renews_a_preemptible_lease(monkeypatch, tmp_path):
    """opencode's config cannot carry a lease id (none exists until claimed), so
    the header is how a config-only client gets a lease at all."""
    app, orch, jobs, world, captured, unit, placer = _auto_build(
        monkeypatch, tmp_path, spark_models=(("model-a", 8000),))
    leases = app.state.leases

    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", headers={"X-LLM-Hold": "opencode"},
                         json={"model": "model-a",
                               "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    held = leases.active_for("spark1", "a")
    assert held is not None and held.holder == "opencode"
    assert held.preemptible is True, "must not 409 other callers' traffic"
    assert held.model == "a", "scoped to the model, folded to the catalog alias"
    first_deadline = held._deadline

    # A second request RENEWS in place rather than fragmenting into two leases.
    async with _client(app) as c:
        await c.post("/v1/chat/completions", headers={"X-LLM-Hold": "opencode"},
                     json={"model": "model-a",
                           "messages": [{"role": "user", "content": "again"}]})
    assert len(leases.active_all("spark1")) == 1
    assert leases.active_for("spark1", "a")._deadline >= first_deadline


async def test_hold_never_steals_another_holders_lease(monkeypatch, tmp_path):
    app, orch, jobs, world, captured, unit, placer = _auto_build(
        monkeypatch, tmp_path, spark_models=(("model-a", 8000),))
    leases = app.state.leases
    mine, _ = leases.claim(LeaseClaimRequest(unit="spark1", holder="alice", model="a"))

    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", headers={"X-LLM-Hold": "opencode"},
                         json={"model": "model-a",
                               "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, "traffic still flows — the hold is a convenience"
    assert leases.active_for("spark1", "a").id == mine.id, "alice keeps her lease"
    assert not any(l.holder == "opencode" for l in leases.active_all("spark1"))


async def test_hold_shields_the_model_from_placement_eviction(monkeypatch, tmp_path):
    """The whole point: a held model is not an eviction candidate."""
    from llmconfig.placement import ResidentFact, rank
    s = Settings(_env_file=None)
    from llmconfig.placement import CandidateFacts
    st = _spark_status(("model-a", 8000))

    def cand(uid, leased, order):
        return CandidateFacts(
            unit_id=uid, kind="spark", status=st, usage="idle", server="spark",
            load_arg="target",
            residents=[ResidentFact(model="model-a", alias="a", budget=0.9,
                                    idle_s=99999, leased=leased)],
            committed=0.9, want=0.3, order=order)

    free = rank("target", [cand("s1", False, 0), cand("s2", True, 1)], s)
    assert free.unit_id == "s1" and free.victims == ["a"], "unleased model is evictable"
    both_held = rank("target", [cand("s1", True, 0), cand("s2", True, 1)], s)
    assert both_held.outcome == "no_capacity", "a held model is never a victim"
