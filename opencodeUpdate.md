# opencodeUpdate — `/v1` auto-loading gateway for opencode `/model`

**Goal:** let opencode users switch models with just `/model <provider>/<name>` — no manual `/swap`.
opencode's `/model` picker has **no selection-time hook**, so the load must happen on the **inference
path**: opencode's first request for the picked model triggers the load. Build an **OpenAI-compatible
`/v1` gateway inside LLMConfig** (this app, port 11430) that resolves the model → ensures it's loaded
via the existing `/api/*` flow → forwards to the real backend, **streaming load progress** on cold
loads.

This is the client-side `vllm-swap` logic (resolve → `/api/load` → poll `/api/jobs`) moved
server-side, plus a reverse proxy and SSE progress. **No new arbitration logic** — reuse `/api/load`
(eviction, WSL-keepalive, per-lane lock already happen there).

## Endpoints (new routes on the existing FastAPI app, prefix `/v1`)
opencode sets each provider's `baseURL` to `http://192.168.1.40:11430/v1`.

### `GET /v1/models`
- Lane = `X-LLM-Lane` header (default `primary`).
- Read `GET /api/models?lane=<lane>`; return OpenAI `{object:"list", data:[{id, object:"model", owned_by:"llmconfig"}]}`.
- `id` = each vLLM `served_name` (e.g. `qwen3-coder-30b`) and each Ollama tag (e.g. `qwen3-coder:30b`).

### `POST /v1/chat/completions` (and `POST /v1/completions`, same logic)
1. `lane` = `X-LLM-Lane` header or `primary`. `stream` = body `stream` (bool).
2. **Resolve** `model` → (server, load_arg, backend):
   - `model` equals a vLLM `served_name` in `GET /api/vllm/aliases?lane=<lane>` → `server=vllm`,
     `load_arg = entry.alias` (e.g. `qwen3-coder-30b` → `coder30-awq`), backend = lane's vLLM relay
     (`primary` → `http://127.0.0.1:11437`, `companion` → `:11438`).
   - else `model` is an Ollama tag (has `:`, present in `/api/models` `ollama[].name`) → `server=ollama`,
     `load_arg = model`, backend = lane's Ollama (`primary` → `:11434`, `companion` → `:11435`).
   - else → `404 {"error":{"code":"model_not_found"}}`. (Formats don't collide: vLLM uses `-`, Ollama `:`.)
3. **Ensure loaded**: `GET /api/status`, find the lane; if `loaded.server==server` and
   `loaded.model==model` → **fast path**, jump to step 5.
4. **Load** (not loaded / wrong model):
   - **Coalesce**: if `lane.swap_in_progress` and `lane.active_job_id` is already loading this target,
     attach to that `job_id`; else `POST /api/load {server, model:load_arg, lane}` → `job_id`.
   - **If `stream`** — respond `text/event-stream`:
     - Poll `GET /api/jobs/{job_id}` (~1.5 s); emit each NEW `job.log` line as a chat chunk
       `delta.content` (e.g. `⏳ loading qwen3-coder-30b on RTX 3090 …\n`).
     - `state==failed` → emit the error as a final delta, `finish_reason:"stop"`, then `data: [DONE]`. STOP.
     - `state==succeeded` → STOP progress; **Forward** (step 5) streaming, relaying upstream chunks
       verbatim, then `[DONE]`.
   - **If not `stream`** (e.g. opencode title-gen):
     - If a load for a *different* model is already in progress on the lane → return a minimal `200`
       chat.completion (empty/space content) so the caller doesn't hang for minutes.
     - else `POST /api/load` + poll until `succeeded`/`failed` (bounded by the alias `load_timeout_s`);
       success → Forward non-stream; failure → `503` with the error.
5. **Forward**: reverse-proxy the original body to `backend/chat/completions` (or `/completions`),
   passing streaming through unchanged.

## Name resolution / state
- Cache `GET /api/vllm/aliases?lane=primary|companion` (served_name → alias) and `/api/models` ollama
  tags with a short TTL (or refresh on each `/v1/models`).
- `/api/status` reports `loaded.model` as the served_name (vLLM) or tag (Ollama), so the
  already-loaded check compares the OpenAI `model` id directly to `loaded.model`.

## Edge cases (must handle)
- **Concurrency**: opencode often fires a title-gen request + the user message near-simultaneously.
  The coalesce step (check `swap_in_progress` / `active_job_id`) prevents a duplicate `/api/load`.
- **Cold-load timeout**: a load past `load_timeout_s` returns job `state=failed` → stream the error,
  never hang.
- **Non-stream during a cold load**: short-circuit (above) so title-gen doesn't block minutes.
- **Auth**: LAN-only / keyless today; if `LLMCONFIG_API_KEY` is set, accept it and pass through to `/api/*`.

## Reuse (already in this codebase)
- `POST /api/load` (async → Job), `GET /api/jobs/{id}` (`state` ∈ pending/running/succeeded/failed,
  `log[]`), `GET /api/status` (`lanes[].loaded{server,model}`, `swap_in_progress`, `active_job_id`),
  `GET /api/vllm/aliases?lane=`, `GET /api/models?lane=`.
- Suggested layout: a new `llmconfig/openai_gateway.py` router `include_router`-ed into `main:app`;
  an httpx reverse-proxy helper; the poll loop mirrors `opencode-config/tools/vllm-swap`.

## opencode side (done separately in `rivaborn/opencode-config` — the compatibility contract)
- providers `ollama` / `vllm` / `companion` set `baseURL=http://192.168.1.40:11430/v1`; the
  `companion` provider adds `customHeaders: {"X-LLM-Lane":"companion"}`. Model ids are unchanged
  (vLLM served_names, Ollama tags). The gateway must accept those ids + the `X-LLM-Lane` header.

## Verify (curl the gateway directly)
1. 3090 `free`; stream a chat request `model=qwen3-coder-30b` → SSE shows loading progress then a real
   completion; `/api/status` = `vllm:qwen3-coder-30b` (primary). No `/swap` involved.
2. `model=qwen3-coder:30b` → gateway evicts vLLM, loads Ollama, answers.
3. `X-LLM-Lane: companion`, `model=qwen2.5:1.5b` → loads on the 3070 Ti; 3090 untouched.
4. Already-loaded model → fast path (immediate forward, no load).
5. Two near-simultaneous requests for the same cold model → single load (coalesced).
6. Non-stream request during a cold load → returns quickly, no hang.

---
*Authored by the opencode-config session as a handoff spec; implement in the LLMConfig repo. The
opencode.json provider rewire + wiki docs are handled on the opencode-config side once `/v1` is live.*
