"""OpenAI-compatible `/v1` gateway — auto-loads the requested model, then proxies.

opencode's `/model` picker has no selection-time hook, so the model switch must
happen on the **inference path**: the first request for a model triggers the load.
This router resolves the requested `model` to a lane backend, ensures it's loaded via
the existing per-lane arbitration (`orch.load` → a Job), **streams the load progress**
to the client on a cold load, then reverse-proxies the request to the real backend.

No new arbitration: it reuses the lane lock / eviction / WSL-keepalive that
`Lane.load` already performs. It just moves the client-side `vllm-swap` poll loop
(resolve → load → poll → forward) server-side and adds SSE progress.

Lane selection: the `X-LLM-Lane` header (default `primary`). opencode's `companion`
provider sets `X-LLM-Lane: companion` so its models land on the RTX 3070 Ti.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import Settings
from .jobs import JobManager
from .lane import Lane
from .orchestrator import Orchestrator
from .schemas import LoadRequest


class OpenAIGateway:
    """Holds the long-lived forwarding client + the resolve/load/forward logic."""

    def __init__(self, orch: Orchestrator, jobs: JobManager, settings: Settings):
        self.orch = orch
        self.jobs = jobs
        self.s = settings
        self._http: httpx.AsyncClient | None = None

    # ---- forwarding client (no read timeout: chat generations can run long) ----
    def client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=10.0)
            )
        return self._http

    async def aclose(self) -> None:
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
        self._http = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def lane(self, lane_id: str) -> Lane:
        try:
            return self.orch.lane(lane_id)
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def resolve(self, lane: Lane, model: str) -> Optional[tuple[str, str, str]]:
        """Map an OpenAI `model` id → (server, load_arg, backend_base_url).

        - a vLLM `served_name` (uses `-`) → ("vllm", alias, lane relay url)
        - else an Ollama tag (has `:`)     → ("ollama", tag, lane ollama url)
        - else None (→ 404). vLLM/Ollama id formats don't collide.
        """
        if not model:
            return None
        # vLLM: match the served_name; prefer a non-blocked alias when several share it
        # (e.g. coder30-awq + coder30-fp8 both serve "qwen3-coder-30b").
        match: Optional[str] = None
        for e in lane.registry.entries():
            if (e.served_name or e.alias) == model:
                if e.status != "blocked":
                    return ("vllm", e.alias, lane.cfg.vllm_relay_url)
                match = match or e.alias
        if match is not None:
            return ("vllm", match, lane.cfg.vllm_relay_url)
        # Ollama: a tag present in the lane's catalog
        if ":" in model:
            try:
                names = {m.name for m in await lane.ollama.list_models()}
            except Exception:
                names = set()
            if model in names:
                return ("ollama", model, lane.cfg.ollama_url)
        return None

    def _ensure_load_job(self, lane: Lane, status, target_kind: str, server: str,
                         load_arg: str, stream: bool):
        """Return (job_or_None, short_circuit). Coalesces onto an identical in-flight
        load; for a *different* in-flight load, queues ours (stream) or signals a
        short-circuit (non-stream, so title-gen doesn't block for minutes)."""
        if status.swap_in_progress and status.active_job_id:
            active = self.jobs.get(status.active_job_id)
            if active and active.kind == target_kind and active.state in ("pending", "running"):
                return active, False  # identical target already loading → attach
            if not stream:
                return None, True     # different model loading + non-stream → bail fast
            # stream + different load: queue ours behind the lane lock (shows "waiting…")
        job = self.orch.load(LoadRequest(server=server, model=load_arg, lane=lane.cfg.id))
        return job, False

    def _fwd_headers(self, headers) -> dict:
        """Forward only what the upstream needs. Pass an Authorization through so a
        keyed backend still works; drop hop-by-hop / length headers (httpx resets)."""
        out = {"content-type": "application/json"}
        auth = headers.get("authorization")
        if auth:
            out["authorization"] = auth
        return out

    @staticmethod
    def _backend_url(backend: str, sub_path: str) -> str:
        return backend.rstrip("/") + "/v1" + sub_path

    # ---- non-streaming forward ----
    async def forward(self, backend: str, sub_path: str, body: dict, headers) -> Response:
        url = self._backend_url(backend, sub_path)
        try:
            resp = await self.client().post(url, json=body, headers=self._fwd_headers(headers))
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"upstream error: {type(e).__name__}: {e}")
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    # ---- streaming forward (raw passthrough, incl. upstream's `data: [DONE]`) ----
    async def _forward_stream(self, backend: str, sub_path: str, body: dict, headers):
        url = self._backend_url(backend, sub_path)
        async with self.client().stream(
            "POST", url, json=body, headers=self._fwd_headers(headers)
        ) as resp:
            if resp.status_code >= 400:
                raw = await resp.aread()
                detail = raw.decode("utf-8", "ignore")[:500] or f"HTTP {resp.status_code}"
                raise RuntimeError(f"upstream {resp.status_code}: {detail}")
            async for chunk in resp.aiter_raw():
                if chunk:
                    yield chunk

    # ------------------------------------------------------------------ #
    # SSE chunk builders (OpenAI chat.completion.chunk / text_completion)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sse(payload: dict) -> bytes:
        return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode("utf-8")

    def _progress_chunk(self, cid: str, created: int, model: str, text: str, is_chat: bool) -> bytes:
        if is_chat:
            choice = {"index": 0, "delta": {"content": text}, "finish_reason": None}
            obj = "chat.completion.chunk"
        else:
            choice = {"index": 0, "text": text, "finish_reason": None}
            obj = "text_completion"
        return self._sse({"id": cid, "object": obj, "created": created, "model": model,
                          "choices": [choice]})

    def _final_chunk(self, cid: str, created: int, model: str, is_chat: bool) -> bytes:
        if is_chat:
            choice = {"index": 0, "delta": {}, "finish_reason": "stop"}
            obj = "chat.completion.chunk"
        else:
            choice = {"index": 0, "text": "", "finish_reason": "stop"}
            obj = "text_completion"
        return self._sse({"id": cid, "object": obj, "created": created, "model": model,
                          "choices": [choice]})

    def _minimal_completion(self, model: str, is_chat: bool) -> dict:
        """A valid, empty 200 so a non-stream caller (e.g. opencode title-gen) returns
        immediately instead of blocking for minutes on a cold load it didn't ask for."""
        created = int(time.time())
        cid = ("chatcmpl-" if is_chat else "cmpl-") + uuid.uuid4().hex[:12]
        if is_chat:
            choice = {"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}
            obj = "chat.completion"
        else:
            choice = {"index": 0, "text": "", "finish_reason": "stop"}
            obj = "text_completion"
        return {"id": cid, "object": obj, "created": created, "model": model, "choices": [choice],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

    # ------------------------------------------------------------------ #
    # Job polling
    # ------------------------------------------------------------------ #
    async def _wait_job(self, job_id: str, timeout: float) -> tuple[bool, str]:
        """Block (non-stream path) until the load Job is terminal. → (ok, error)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            cur = self.jobs.get(job_id)
            if cur is None:
                return False, "load job vanished"
            if cur.state == "succeeded":
                return True, ""
            if cur.state == "failed":
                return False, cur.error or cur.message or "load failed"
            await asyncio.sleep(1.0)
        return False, f"load did not finish within {int(timeout)}s"

    async def _stream_load_then_forward(self, job_id: Optional[str], backend: str, sub_path: str,
                                        body: dict, model: str, headers, is_chat: bool):
        created = int(time.time())
        cid = ("chatcmpl-" if is_chat else "cmpl-") + uuid.uuid4().hex[:12]
        emitted = 0
        if job_id is not None:
            while True:
                cur = self.jobs.get(job_id)
                if cur is None:
                    yield self._progress_chunk(cid, created, model, "\nload job vanished\n", is_chat)
                    yield self._final_chunk(cid, created, model, is_chat)
                    yield b"data: [DONE]\n\n"
                    return
                while emitted < len(cur.log):
                    line = cur.log[emitted]
                    emitted += 1
                    yield self._progress_chunk(cid, created, model, f"⏳ {line}\n", is_chat)
                if cur.state == "succeeded":
                    break
                if cur.state == "failed":
                    err = cur.error or cur.message or "load failed"
                    yield self._progress_chunk(cid, created, model, f"❌ load failed: {err}\n", is_chat)
                    yield self._final_chunk(cid, created, model, is_chat)
                    yield b"data: [DONE]\n\n"
                    return
                await asyncio.sleep(1.5)
        # Loaded — relay the upstream completion verbatim (it emits its own [DONE]).
        try:
            async for chunk in self._forward_stream(backend, sub_path, body, headers):
                yield chunk
        except Exception as e:  # noqa: BLE001 — surface upstream failure into the stream
            yield self._progress_chunk(cid, created, model, f"\n❌ upstream error: {e}\n", is_chat)
            yield self._final_chunk(cid, created, model, is_chat)
            yield b"data: [DONE]\n\n"

    # ------------------------------------------------------------------ #
    # Request handler (shared by chat/completions and completions)
    # ------------------------------------------------------------------ #
    async def handle_completion(self, request: Request, sub_path: str, is_chat: bool):
        lane = self.lane(request.headers.get("x-llm-lane") or "primary")
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="request body must be JSON")
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="request body must be a JSON object")
        model = body.get("model") or ""
        stream = bool(body.get("stream"))

        resolved = await self.resolve(lane, model)
        if resolved is None:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": f"model '{model}' not found on lane '{lane.cfg.id}'",
                                    "type": "invalid_request_error", "code": "model_not_found"}},
            )
        server, load_arg, backend = resolved

        # Fast path: the lane already serves exactly this model → forward, no load.
        status = await lane.status()
        if status.loaded and status.loaded.server == server and status.loaded.model == model:
            if stream:
                return StreamingResponse(
                    self._stream_load_then_forward(None, backend, sub_path, body, model,
                                                   request.headers, is_chat),
                    media_type="text/event-stream",
                )
            return await self.forward(backend, sub_path, body, request.headers)

        # Need to load (cold / wrong model). Coalesce onto an identical in-flight load.
        target_kind = f"load:{lane.cfg.id}:{server}:{load_arg}"
        job, short_circuit = self._ensure_load_job(lane, status, target_kind, server, load_arg, stream)

        if stream:
            return StreamingResponse(
                self._stream_load_then_forward(job.id if job else None, backend, sub_path, body,
                                               model, request.headers, is_chat),
                media_type="text/event-stream",
            )

        # Non-stream: a different model is mid-load → return an empty 200 (don't hang).
        if short_circuit or job is None:
            return JSONResponse(content=self._minimal_completion(model, is_chat))

        if server == "vllm":
            entry = lane.registry.get(load_arg)
            base = float(entry.load_timeout_s if entry else self.s.default_vllm_load_timeout_s)
            timeout = base + self.s.vllm_ready_grace_s + 30.0
        else:
            timeout = 600.0
        ok, err = await self._wait_job(job.id, timeout)
        if not ok:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": f"failed to load '{model}': {err}",
                                    "type": "server_error", "code": "model_load_failed"}},
            )
        return await self.forward(backend, sub_path, body, request.headers)

    async def models(self, request: Request) -> dict:
        lane = self.lane(request.headers.get("x-llm-lane") or "primary")
        ids: list[str] = []
        for e in lane.registry.entries():
            ids.append(e.served_name or e.alias)
        try:
            for m in await lane.ollama.list_models():
                if m.name:
                    ids.append(m.name)
        except Exception:
            pass
        seen: set[str] = set()
        data = []
        for i in ids:
            if i and i not in seen:
                seen.add(i)
                data.append({"id": i, "object": "model", "owned_by": "llmconfig"})
        return {"object": "list", "data": data}


def build_gateway_router(gateway: OpenAIGateway) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["openai-gateway"])

    @router.get("/models")
    async def v1_models(request: Request) -> dict:
        return await gateway.models(request)

    @router.post("/chat/completions")
    async def v1_chat_completions(request: Request):
        return await gateway.handle_completion(request, "/chat/completions", is_chat=True)

    @router.post("/completions")
    async def v1_completions(request: Request):
        return await gateway.handle_completion(request, "/completions", is_chat=False)

    return router
