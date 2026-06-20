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

## Current served contexts (serve.sh) → goal
| alias / served_name              | now      | target  | note                                                                 |
| -------------------------------- | -------- | ------- | ------------------------------------------------------------------- |
| `coder30-awq` / qwen3-coder-30b  | 65536    | —       | ✅ the reference; leave                                              |
| `coder14` / qwen2.5-coder-14b    | 32768    | 32768   | architectural cap (Qwen2.5 native 32768, no YaRN) — leave unless you add `--rope-scaling` (RoPE-NaN risk) |
| `gemma4` / gemma-4-26b           | 32768    | ≥32768  | native 256K — could go higher                                       |
| `vl7` / qwen2.5-vl-7b            | 16384    | ≥32768  | small weights, lots of room                                         |
| `coder32` / qwen2.5-coder-32b    | **4096** | ≥32768  | Qwen2.5 native 32768 (YaRN→131072)                                  |
| `q3-32b` / qwen3-32b             | **4096** | ≥32768  | dense 32B — tightest KV budget                                      |
| `q35-27b` / qwen3.5-27b          | **4096** | ≥32768  | 27B → more KV room than the 32B                                     |
| `vl32` / qwen2.5-vl-32b          | **4096** | ≥32768  | vision; lower opencode priority                                     |
| `q36-moe` / qwen3.6-moe          | **4096** | ≥32768  | MoE; verify chat first                                              |
| `q36-27b` / qwen3.6-27b          | **4096** | (blocked) | fix the FP8+offload bug first                                     |
| `devstral`                       | **4096** | ≥32768  | Mistral native 128K, ~14 GB weights → can go high                   |
| `smoke`                          | 4096     | —       | test model; leave                                                   |

(Aliases mapped from serve.sh; confirm against your `case "$ALIAS"`.)

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

## Ollama, too (separate, also box-side)
The primary `Ollama` NSSM service sets **`OLLAMA_CONTEXT_LENGTH=4096`** (in its `AppEnvironmentExtra`).
That **silently truncates** Ollama models to 4 k — including opencode's **default**
`ollama/qwen3-coder:30b`. Raise it (e.g. 32768+) so the default model actually sees opencode's context.
(Same for `OllamaCompanion` if you want big-context Ollama models on the 3070 Ti.)

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
