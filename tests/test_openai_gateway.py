"""OpenAI `/v1` gateway: model resolution, fast-path forward, cold-load streaming,
the non-stream short-circuit, and the model list — all with fake lane backends and
a MockTransport upstream (no nvidia-smi / WSL / real Ollama or vLLM)."""
import json

import httpx
from httpx import ASGITransport

import llmconfig.lane as lane_mod
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from llmconfig.config import Settings
from llmconfig.gpu import GpuInfo
from llmconfig.jobs import JobManager
from llmconfig.lane_state import LaneDefaults
from llmconfig.openai_gateway import OpenAIGateway, build_gateway_router
from llmconfig.orchestrator import Orchestrator
from llmconfig.proc import CmdResult
from llmconfig.registry import Registry
from llmconfig.schemas import OllamaModel

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
    gateway = OpenAIGateway(orch, jobs, s)
    gateway._http = httpx.AsyncClient(transport=ASGITransport(app=_upstream_app(captured)))

    app = FastAPI()
    app.include_router(build_gateway_router(gateway))
    return app, orch, jobs, world, captured


def _client(app):
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


async def test_models_lists_vllm_served_names_and_ollama_tags(monkeypatch, tmp_path):
    app, *_ = _build(monkeypatch, tmp_path)
    async with _client(app) as c:
        r = await c.get("/v1/models")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["data"]}
    assert "qwen3-coder-30b" in ids   # a vLLM served_name (from the seeded registry)
    assert "qwen3-coder:30b" in ids   # an Ollama tag
    assert all(m["owned_by"] == "llmconfig" for m in r.json()["data"])


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
