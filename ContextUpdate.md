# ContextUpdate — raise vLLM (and Ollama) context windows so models are usable in opencode

## Why
opencode sends a large baseline context on every request — system prompt + `AGENTS.md` + the three MCP
tool schemas (context7 / serena / playwright) — measured at **~24 577 input tokens for a one-line
question**. So a model is only usable as an opencode session model if its served context comfortably
exceeds that (≈ **28–32 k+**, to leave room for files + output).

Today most vLLM aliases are served at `--max-model-len 4096` (placeholder-conservative — the serve.sh
comment blames the vLLM 0.20.2 CUDA-graph memory bug, not VRAM). That makes **8 of 12 models
unreachable from opencode** (vLLM 400s once prompt+output exceed the window). `coder30-awq` already
proves the headroom exists: FP8 KV → **65536** ctx in ~3.1 GB of KV cache.

This is the **box-side half**. The opencode side is already done: `opencode.json` sets each model's
`context = served --max-model-len − output` so opencode never overflows (`rivaborn/opencode-config`,
commit `5f343df`). When you raise a `--max-model-len`, tell the opencode-config session and they
re-sync.

## Served contexts — ✅ essentially DONE (audited against live serve.sh 2026-07-24)

Read from the running `/home/folar/vllm/serve.sh`, not from memory. Almost everything this
document asked for had already been done:

| alias / served_name             | this doc claimed | **actual** | KV   | note                                                    |
| ------------------------------- | ---------------- | ---------- | ---- | ------------------------------------------------------- |
| `coder30-awq` / qwen3-coder-30b | 65536            | **65536**  | fp8  | the reference build                                     |
| `coder14` / qwen2.5-coder-14b   | 32768            | **32768**  | fp8  | architectural cap (Qwen2.5 native, no YaRN)             |
| `gemma4` / gemma-4-26b          | 32768            | **32768**  | fp8  | native 256K — could go higher if ever needed            |
| `vl7` / qwen2.5-vl-7b           | 16384            | **32768**  | fp8  | raised                                                  |
| `coder32` / qwen2.5-coder-32b   | 4096             | **32768**  | fp8  | raised                                                  |
| `q3-32b` / qwen3-32b            | 4096             | **40960**  | fp8  | raised; dense 32B, the tightest KV budget               |
| `q35-27b` / qwen3.5-27b         | 4096             | **65536**  | fp8  | raised                                                  |
| `q36-moe` / qwen3.6-moe         | 4096             | **131072** | fp8  | raised                                                  |
| `q36-27b` / qwen3.6-27b         | 4096 (blocked)   | **65536**  | fp8  | raised; no longer blocked                               |
| `q36-27b-abl`                   | (not listed)     | **65536**  | fp8  | abliterated sibling, added later                        |
| `devstral`                      | 4096             | **98304**  | fp8  | raised                                                  |
| `vl32` / qwen2.5-vl-32b         | 4096             | **4096**   | fp16 | **constrained, not pending** — see below                |
| `surya2` / surya-ocr-2          | (not listed)     | 18000      | fp16 | OCR pipeline; sized for its job                         |
| `qwen3-embed`                   | (not listed)     | 8192       | —    | embeddings (`--runner pooling`); no KV growth           |
| `smoke`                         | 4096             | 4096       | —    | test model; leave                                       |

### `vl32` is a deliberate limit, not an oversight

Its `serve.sh` case documents the corner it is in, and the numbers do not leave room:

- `Qwen2.5-VL-32B-Instruct-AWQ` — ~19.5 GB language weights **plus** vision-encoder workspace
  on a 24 GB card.
- It already runs `--cpu-offload-gb 4` purely to free KV space, with `--max-num-seqs 1` and
  `--enforce-eager` (vLLM's cpu-offload `uva.py` is not Dynamo-traceable, so compile is out).
- **fp8 KV does not work for this build** — three routes were tried and all failed
  (`fp8` auto-E4M3 → Triton `fp8e4nv` conversion error; `fp8_e5m2` → rejected by vLLM;
  `fp8 + --enforce-eager` → compressed-tensors AWQ dequant kernel). So KV is fp16, i.e. **double**
  the bytes per token of every other alias here.

Raising it meaningfully would mean offloading substantially more weight to CPU, which trades a
large throughput loss for context on a model this doc itself rates lower priority. **Leave it at
4096** unless a specific vision workload needs more, and revisit if a future vLLM/AWQ build
accepts fp8 KV on Ampere.

### Remaining
Nothing on the vLLM side. The one real gap found in this audit was not a context value at all:
the live `qwen3-embed` case had never been committed to `deploy/serve.sh`, so a redeploy would
have silently dropped the embeddings server (hard invariant 13). Now committed.

## How (your existing recipe — per-alias in serve.sh)
Mirror what `coder30-awq` already does:
- **`--kv-cache-dtype fp8`** — halves KV/token; this is what bought coder30-awq 65536.
- **`--gpu-memory-utilization`** — per-alias; serve.sh notes a safe ceiling ~0.88, but coder30-awq runs
  0.93. Tune up as weights allow.
- **`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`** — already exported globally (the 0.20.2 CUDA-graph
  charge workaround); keep it.
- **VRAM math:** 24 GiB − weights − CUDA-graph/activation = KV budget. The dense 32B AWQ (~18 GB
  weights) is the tightest — FP8 KV should still reach 32768; the 27B / devstral (smaller weights) can
  likely go 49152–65536. Dense models have a larger KV/token than coder30-awq's MoE-A3B (more active
  params), so don't assume 65536 everywhere — tune per alias with a canary recall test (like the
  28K/50K canaries you already ran for coder30-awq).

## Ollama, too (separate, also box-side) — DONE 2026-07-24
`OLLAMA_CONTEXT_LENGTH` lives in each NSSM service's `AppEnvironmentExtra` and **silently
truncates** any model without a baked `num_ctx`. Verified and corrected on the box:

| Service           | Was                       | Now       | Verified                                                                       |
| ----------------- | ------------------------- | --------- | ------------------------------------------------------------------------------ |
| `Ollama` (3090)   | already `32768`           | unchanged | `qwen3-coder:30b` -> `context_length=32768`, **20.2 GB fully on GPU, no spill** |
| `OllamaCompanion` | **unset** -> Ollama's 4096 | `32768`   | `qwen2.5:1.5b` -> `context_length=32768` (was 4096)                            |

Two corrections to what this doc previously claimed: the primary was **already** at 32768, and
the companion's problem was that the variable was **absent entirely**, so it fell back to
Ollama's built-in 4096 default — which is why the `/swap` relay model served at 4 k.

The primary was checked against the **running process** (load a model, read `/api/ps`), not the
registry, since a registry edit only takes effect on service restart.

The effective window is now visible per unit in the UI and on `/api/status` ->
`lanes[].loaded.context_len`, sourced from `/api/ps` `context_length` — the *runtime* window, not
the architectural maximum that `/api/show` reports.

> Tagged models with a baked `num_ctx` (`qwen3.6:27b-64k`, `qwen2.5-coder:32b-12k`, …) are
> unaffected: the Modelfile wins over the service default, and baking a tag remains the way to
> exceed it (hard invariant 6 — never add a context arg to the load path).

## opencode.json contract (already in place)
- `context = served − output`, because opencode uses `output` as a fixed `max_tokens` and caps the
  prompt at `context` with **no** reservation. Keep that invariant.
- After each `--max-model-len` bump: send the opencode-config session the new served value; they set
  `context = served − output` and drop the "(4K ctx)" labels.

## Verify
Per raised alias: `serve.sh <alias>` → `/api/status` shows it loaded → a canary recall test near the
new ceiling → then `/model vllm/<served-name>` from opencode and send a message. opencode's ~24.5k
overhead is itself the smoke test: if a normal prompt answers without a 400, the context is big enough.

---
*Authored by the opencode-config session as a handoff. `opencode.json` already mirrors the current
served contexts; ping that session to re-sync after any `--max-model-len` change.*
