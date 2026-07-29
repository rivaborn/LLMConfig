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

> **Scope note (2026-07-29).** This doc was written when the lab had two context tiers, vLLM and
> Ollama. There are now **three** — the four DGX Sparks are a tier of their own, ceilinged per
> recipe in the catalog rather than in `serve.sh`. Sections added below. All three were re-audited
> against the live box on 2026-07-29 and `opencode.json` re-synced to match (66 entries).

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

**Re-audited 2026-07-29 against `deploy/serve.sh`: all 15 values above are unchanged.** The vLLM
tier is stable; every entry in `opencode.json`'s `vllm` provider already summed exactly to its
ceiling and needed no edit.

## Spark served contexts — the third tier (added 2026-07-29)

A Spark's ceiling is **not** in `serve.sh`. It is the `--max-model-len` pinned in each recipe's
`extra_args` in `llmconfig/data/spark_models.default.yaml`, because the upstream recipes are sized
for two nodes and must be capped down for one. **They are not one flat number** — the assumption
that every recipe served 65536 was true only at first:

| served_name             | max-model-len | mem_fraction | status                | note                                          |
| ----------------------- | ------------: | -----------: | --------------------- | --------------------------------------------- |
| `gemma-4-26b-fp8`       |     **65536** |         0.40 | ok (verified 07-24)   | `needs_empty_node` — ~74 GB load transient    |
| `gpt-oss-120b`          |     **65536** |     *(unset)* | unverified            | no budget = whole-node claim                  |
| `qwen35-35b-a3b`        |     **65536** |         0.50 | unverified            | SGLang                                        |
| `qwen3-coder-next`      |    **262144** |         0.60 | unverified            | KV only 12 GB at full window                  |
| `qwen36-35b-a3b`        |    **262144** |         0.52 | unverified            | +MTP speculative decoding                     |
| `qwen36-27b`            |    **262144** |         0.68 | unverified            | dense 27B, 32 GB KV                           |
| `qwen35-122b-int4`      |    **262144** |         0.80 | **ok (verified 07-29)** | needs the marlin env override — see below   |
| `qwen3-vl-embedding-8b` |         32768 |         0.33 | unverified            | pooling runner → `/v1/embeddings`             |
| `qwen3-vl-reranker-8b`  |         32768 |         0.35 | unverified            | pooling → `/v1/rerank`, `needs_empty_node`    |

### The 122B serves 262144, and the window was never the problem

Earlier notes in this repo and in `opencode-config` recorded it at 64K, then 131072. Both are
superseded. Verified 2026-07-29 straight off the node — `spark1:8000/v1/models` reports
`max_model_len: 262144` — and it is the model currently resident there.

The failure that produced the 64K figure (booted at 94.8 % pool, died on its **first token**,
RayWorker killed with `NVRM NV_ERR_NO_MEMORY`) was fixed by an **env override, not a smaller
window**:

```yaml
extra_args: ["--max-model-len", "262144", "-o", "env.VLLM_MARLIN_USE_ATOMIC_ADD=0"]
```

The `@eugr` recipe ships that env set to `1`; on GB10 the marlin split-K path dies during
CUDA-graph capture (capture 23 of 51, *"CUDA error: an illegal instruction was encountered"*).
Weights and KV were never the constraint — 62.87 GiB weights, 31.2 GiB KV free, and the measured
KV pool is 1,121,010 tokens, so the 256K cap costs no memory at all.

**Generalise from this:** `-o env.KEY=VAL` in `extra_args` (substituted into `SPARK_RUN_CMD`'s
`{extra}`) is load-bearing, not decoration. Before concluding a Spark model needs a smaller context,
check whether a recipe env default is what is actually killing it.

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

### Per-tag effective windows (measured 2026-07-29 via `/api/show`)

`served = baked num_ctx, else OLLAMA_CONTEXT_LENGTH (32768)`. Only five tags carry a bake:

| tag                       | baked `num_ctx` |    served | note                                     |
| ------------------------- | --------------: | --------: | ---------------------------------------- |
| `qwen2.5-coder:32b-8k`    |            8192 |      8192 | bake                                     |
| `qwen2.5-coder:32b-12k`   |           12288 |     12288 | bake                                     |
| `qwen3.6:27b-64k`         |           65536 |     65536 | bake                                     |
| `qwen3.6:27b-96k`         |           98304 |     98304 | bake — full-GPU                          |
| `qwen3.6:27b-128k`        |          131072 |    131072 | bake — ~16 tok/s, CPU spill              |
| *all other tags*          |        *(none)* | **32768** | service default truncates                |

"All other tags" is `qwen3-coder:30b`, `qwen2.5-coder:32b`, `qwen2.5-coder:14b`, `qwen2.5:1.5b`,
`devstral-small-2:latest`, `qwen3:32b`, `qwen3.6:35b-a3b`, `qwen3.6:27B`, `Qwen3.5:27B`,
`qwen2.5vl:32b`, `qwen2.5vl:7b`, `gemma4:26b`.

> ⚠️ **The `/api/show` trap.** Read `num_ctx` out of the **`parameters`** block. Do *not* read
> `model_info.*context_length*` — that is the **architectural** maximum and is wildly larger:
> `qwen3.6:27B` reports 262144, `devstral-small-2` 393216, `qwen2.5vl:7b` 128000. Sizing
> `opencode.json` from those would set limits up to 8× the real window, and the model would 400 on
> any long prompt while looking perfectly configured. A tag with no `num_ctx` in `parameters` serves
> 32768 no matter what `model_info` claims.

## opencode.json contract (already in place)
- `context = served − output`, because opencode uses `output` as a fixed `max_tokens` and caps the
  prompt at `context` with **no** reservation. Keep that invariant.
- It is **equality, not `<=`**. Anything below the ceiling is window you paid VRAM for and cannot
  use; anything above it is a 400 mid-session.
- After each `--max-model-len` bump: send the opencode-config session the new served value; they set
  `context = served − output` and drop the "(4K ctx)" labels.
- **This now covers three tiers, not two** — a Spark `extra_args` change or an Ollama re-bake
  obliges the same re-sync as a `serve.sh` edit.
- **Pooling and OCR models are deliberately excluded** from `opencode.json`: `qwen3-embed`,
  `surya-ocr-2`, `qwen3-vl-embedding-8b`, `qwen3-vl-reranker-8b`. They serve `/v1/embeddings`,
  `/v1/rerank` and `/v1/score`, not chat, so they would be dead entries in the `/model` picker.
  `/v1/models` lists them — expect that diff when auditing ids, and don't "fix" it.

### Re-sync 2026-07-29 (state after)

Audited every id against the live gateway and every limit against its ceiling: **66 entries across
8 providers, all satisfying `context + output == served`, zero stale ids.** What was wrong:

| Provider    | Fixed                                                                                   |
| ----------- | --------------------------------------------------------------------------------------- |
| `auto`      | 122B `65536` → **262144** (4× understated — the single biggest miss)                    |
| `sparkN` ×4 | flat `57344 + 8192` replaced by real per-recipe ceilings; **added** `qwen36-35b-a3b` and `qwen35-122b-int4` (5 → 7 models each) |
| `ollama`    | **added** `qwen3.6:27b-64k` (missing entirely); explicit limits on **11** tags that had none and were relying on opencode's own default; `32b-8k`/`32b-12k` corrected |
| `companion` | added `30720 + 2048` (its service default was once unset → Ollama's 4096)                |
| `vllm`      | nothing — already exact                                                                  |

## Verify
Per raised alias: `serve.sh <alias>` → `/api/status` shows it loaded → a canary recall test near the
new ceiling → then `/model vllm/<served-name>` from opencode and send a message. opencode's ~24.5k
overhead is itself the smoke test: if a normal prompt answers without a 400, the context is big enough.

---
*Authored by the opencode-config session as a handoff. `opencode.json` already mirrors the current
served contexts; ping that session to re-sync after any `--max-model-len` change.*
