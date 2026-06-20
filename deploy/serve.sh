#!/usr/bin/env bash
# Model-swap wrapper. Replaces `ollama run <name>` semantics for vLLM.
# Usage:  bash serve.sh <alias>
set -euo pipefail

ALIAS="${1:-}"
PORT="${PORT:-11436}"
HOST="${HOST:-0.0.0.0}"

# Stop any running vLLM server first.
pkill -f 'vllm serve' 2>/dev/null || true
sleep 2

# Activate venv if not already.
if [ -z "${VIRTUAL_ENV:-}" ]; then
  # shellcheck disable=SC1091
  source "$HOME/vllm/.venv/bin/activate"
fi

# Select the RTX 3090 by UUID, resolved to its current integer index. vLLM 0.20.2
# only accepts an integer CUDA_VISIBLE_DEVICES (a raw UUID fails ModelConfig's
# int() parse), and the chassis 3070 Ti drops in/out of CUDA enumeration, which
# shifts the 3090 between index 0 and 1 (a stale fixed index gave the 3070 Ti or
# an NVML "Invalid Argument"). Resolving UUID->index at launch is robust to both.
# Override by exporting CUDA_VISIBLE_DEVICES yourself before invoking serve.sh.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  _GPU_IDX="$(python - <<'PYEOF'
import torch
for i in range(torch.cuda.device_count()):
    if "739bece9" in str(getattr(torch.cuda.get_device_properties(i), "uuid", "")):
        print(i); break
PYEOF
)"
  if [ -z "${_GPU_IDX}" ]; then
    echo "serve.sh: RTX 3090 (GPU-739bece9-...) not found in torch CUDA enumeration; refusing to launch on the wrong GPU." >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="${_GPU_IDX}"
fi
export HF_HUB_ENABLE_HF_TRANSFER=1
# vLLM 0.20.2 charges CUDA-graph profile memory against --gpu-memory-utilization,
# which can drive KV cache budget negative when --max-model-len is raised. The
# documented fix is to skip that estimate; works in tandem with the per-alias
# utilization values below. See "Known issues -> Available KV cache memory".
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
# nvcc lives in /usr/local/cuda-13.0/bin; ensure it's on PATH for torch.compile
# repro-strings and JIT backends (flashinfer etc.). /etc/profile.d/cuda.sh only
# fires for login shells, but serve.sh is often spawned non-login.
if [ -d /usr/local/cuda-13.0/bin ]; then
  export CUDA_HOME=/usr/local/cuda-13.0
  case ":$PATH:" in *":$CUDA_HOME/bin:"*) ;; *) export PATH="$CUDA_HOME/bin:$PATH" ;; esac
fi

# Memory tuning notes:
#   3090 has 24 GiB. Windows desktop holds ~2.3 GiB → ~21.7 GiB free at startup.
#   --gpu-memory-utilization is fraction of TOTAL, so safe ceiling is ~0.88.
#   torch.compile + CUDA graphs are now enabled (cuda-nvcc-13-0 installed in WSL);
#   first request takes ~30s extra to warm, subsequent requests get ~20-30% throughput.

case "$ALIAS" in
  smoke)
    exec vllm serve Qwen/Qwen2.5-0.5B-Instruct \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name smoke \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.50
    ;;
  vl7)
    # FP8-KV context rollout (2026-06-19): tiny KV/token (32768 needs only ~0.88 GiB),
    # but FP16 7B weights + vision encoder + compile workspace left just 0.28 GiB at
    # util 0.85. Raise util to 0.92 and --enforce-eager (frees the compile workspace)
    # → ~2+ GiB KV, 32768 fits easily. Qwen2.5-VL native 32768 cap (no YaRN) is the max.
    exec vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name qwen2.5-vl-7b \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.92 \
      --kv-cache-dtype fp8 \
      --enforce-eager \
      --limit-mm-per-prompt '{"image":4}' \
      --enable-auto-tool-choice --tool-call-parser hermes
    ;;
  coder14)
    # 0.70 was enough with --enforce-eager; torch.compile workspace needs more.
    # FP8-KV rollout (2026-05-14): --kv-cache-dtype fp8 added. Qwen2.5 has
    # native max_position_embeddings=32768 and no YaRN rope_scaling, so we
    # CANNOT extend past 32K without risking RoPE NaN. FP8 KV here just halves
    # the KV memory footprint at the existing context ceiling.
    exec vllm serve Qwen/Qwen2.5-Coder-14B-Instruct-AWQ \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name qwen2.5-coder-14b \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.80 \
      --kv-cache-dtype fp8 \
      --enable-prefix-caching \
      --enable-auto-tool-choice --tool-call-parser hermes \
      --chat-template /home/folar/vllm/templates/tool_chat_template_hermes.jinja
    ;;
  gemma4)
    # MoE 26B-A4B. Multimodal (image+video) but video preprocessing hangs/OOMs
    # WSL with 15 GB RAM during profile-run. Disable MM for text-only chat.
    # Remove --limit-mm-per-prompt if you actually need vision.
    # FP8 KV NOT enabled — incompatible on Ampere + this AWQ build in vLLM
    # 0.20.2. Three failure modes:
    #   1. --kv-cache-dtype fp8 (auto E4M3) → Triton emits fp8e4nv conversion
    #      that Ampere cc=86 doesn't support
    #   2. --kv-cache-dtype fp8_e5m2 → vLLM rejects "not supported with fp8
    #      checkpoints" (compressed-tensors INT4 misclassified)
    #   3. fp8 + --enforce-eager → compressed-tensors AWQ dequant kernel
    #      itself uses the unsupported fp8e4nv. Eager doesn't help.
    # Bumping max-model-len 16384 → 32768 anyway (Gemma-4 has native 256K
    # cap and sliding-window attn keeps KV smaller than dense models).
    exec vllm serve cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name gemma-4-26b \
      --max-model-len 32768 \
      --max-num-batched-tokens 4096 \
      --gpu-memory-utilization 0.85 \
      --limit-mm-per-prompt '{"image":0,"video":0}' \
      --enable-prefix-caching \
      --enable-auto-tool-choice --tool-call-parser gemma4 \
      --chat-template /home/folar/vllm/templates/tool_chat_template_gemma4.jinja
    ;;
  q35-27b)
    # Qwen3.5-27B is hybrid (attention + Mamba layers) and stealth-multimodal.
    # Mamba state eats GPU workspace, so we additionally cap max-num-seqs=1
    # and drop prefix caching (which forces 'align' Mamba mode with overhead).
    # FP8-KV context rollout (2026-06-19). Mamba-hybrid → tiny attention KV/token:
    # 65536 needs only ~2.15 GiB KV. But torch.compile workspace left just 0.12 GiB
    # at util 0.93, so the load failed. --enforce-eager frees that workspace, leaving
    # GBs for KV — 65536 fits easily, no CPU offload needed. (Eager is also already
    # noted as needing less util than compile here.) Verified with a ~58K canary.
    exec vllm serve QuantTrio/Qwen3.5-27B-AWQ \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name qwen3.5-27b \
      --max-model-len 65536 \
      --max-num-batched-tokens 2048 \
      --max-num-seqs 1 \
      --gpu-memory-utilization 0.93 \
      --kv-cache-dtype fp8 \
      --enforce-eager \
      --limit-mm-per-prompt '{"image":0,"video":0}' \
      --enable-auto-tool-choice --tool-call-parser qwen3_xml
    ;;
  q3-32b)
    # Dense 32B AWQ (~19 GB weights, ~128 KB/token KV even at FP8 — the tightest).
    # FP8-KV context rollout (2026-06-19). Measured on this box (24 GB, headless):
    #   compile+0.93        → 1.82 GiB KV (max len ~14.9K)
    #   eager+0.94          → 3.20 GiB KV (max len ~26.2K) — still short of 32768
    #   eager+0.94 +offload2→ 5.81 GiB KV → fits Qwen3's native 40960 (needs 5.0 GiB)
    # So: --enforce-eager (frees CUDA-graph workspace; also required for cpu-offload,
    # whose uva.py isn't Dynamo-traceable) + --cpu-offload-gb 2 → 40960 (native cap).
    # Offload trades a little latency for the context.
    exec vllm serve Qwen/Qwen3-32B-AWQ \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name qwen3-32b \
      --max-model-len 40960 \
      --max-num-batched-tokens 2048 \
      --max-num-seqs 1 \
      --gpu-memory-utilization 0.94 \
      --kv-cache-dtype fp8 \
      --enforce-eager \
      --cpu-offload-gb 2 \
      --limit-mm-per-prompt '{"image":0,"video":0}' \
      --enable-auto-tool-choice --tool-call-parser hermes \
      --chat-template /home/folar/vllm/templates/tool_chat_template_hermes.jinja
    ;;
  vl32)
    # Vision model is tight on 24 GB — language weights (19.5 GB) + vision encoder
    # workspace leaves no KV room. Offload 4 GB of weights to CPU to free KV space.
    # --enforce-eager required: vLLM's cpu-offload uva.py is not Dynamo-traceable
    # (OrderedDict setattr); torch.compile errors out otherwise.
    exec vllm serve Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name qwen2.5-vl-32b \
      --max-model-len 4096 \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 1 \
      --gpu-memory-utilization 0.93 \
      --cpu-offload-gb 4 \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --enforce-eager \
      --enable-auto-tool-choice --tool-call-parser hermes
    ;;
  coder32)
    # Dense 32B AWQ (~19 GB weights) — same tight KV budget as q3-32b. Qwen2.5 native
    # max_position_embeddings=32768 with no YaRN → 32768 is the architectural ceiling
    # (do NOT extend past 32K, RoPE-NaN risk). FP8-KV context rollout (2026-06-19):
    # --kv-cache-dtype fp8 + --enforce-eager (frees CUDA-graph workspace; also required
    # for cpu-offload) + --cpu-offload-gb 2 to clear 32768 (eager+0.94 alone left only
    # ~26K; offload 2 buys ~5.8 GiB KV — see q3-32b measurements).
    exec vllm serve Qwen/Qwen2.5-Coder-32B-Instruct-AWQ \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name qwen2.5-coder-32b \
      --max-model-len 32768 \
      --max-num-batched-tokens 2048 \
      --max-num-seqs 1 \
      --gpu-memory-utilization 0.94 \
      --kv-cache-dtype fp8 \
      --enforce-eager \
      --cpu-offload-gb 2 \
      --limit-mm-per-prompt '{"image":0,"video":0}' \
      --enable-auto-tool-choice --tool-call-parser hermes \
      --chat-template /home/folar/vllm/templates/tool_chat_template_hermes.jinja
    ;;
  coder30-fp8)
    # KNOWN-BROKEN on vLLM 0.20.2 + cu130: FP8 weights + --cpu-offload-gb hits
    # "RuntimeError: b_scales is not on GPU". Use `coder30-awq` instead (fits
    # without offload) or wait for a vLLM upgrade.
    # --enforce-eager required because cpu-offload uva.py is not Dynamo-traceable.
    exec vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name qwen3-coder-30b \
      --max-model-len 4096 \
      --max-num-batched-tokens 2048 \
      --gpu-memory-utilization 0.85 \
      --cpu-offload-gb 10 \
      --enable-prefix-caching \
      --enforce-eager \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      --chat-template /home/folar/vllm/templates/tool_chat_template_qwen3coder.jinja
    ;;
  coder30-awq)
    # Recommended path for Qwen3-Coder-30B.
    # FP8-KV rollout Stage B (2026-05-14): --kv-cache-dtype fp8 halves KV per
    # token from 96 KB to 48 KB. At max-model-len 65536 this is ~3.1 GB KV,
    # within the 4.5 GB budget on the 3090. Stage A (40960) was verified with
    # a 28K-token canary recall test; Stage B was verified with a 50K-token
    # canary recall test. See "Context budget for coding workflows" in wiki.
    exec vllm serve QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name qwen3-coder-30b \
      --max-model-len 65536 \
      --max-num-batched-tokens 4096 \
      --gpu-memory-utilization 0.93 \
      --kv-cache-dtype fp8 \
      --enable-prefix-caching \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      --chat-template /home/folar/vllm/templates/tool_chat_template_qwen3coder.jinja
    ;;
  q36-27b)
    # 27B dense FP8 ~21 GB on GPU + ~8 GB CPU offload. Same FP8+offload
    # caveat as coder30-fp8 — may hit "b_scales is not on GPU".
    # --enforce-eager required because cpu-offload uva.py is not Dynamo-traceable.
    exec vllm serve Qwen/Qwen3.6-27B-FP8 \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name qwen3.6-27b \
      --max-model-len 4096 \
      --max-num-batched-tokens 2048 \
      --gpu-memory-utilization 0.88 \
      --cpu-offload-gb 8 \
      --enable-prefix-caching \
      --enforce-eager \
      --enable-auto-tool-choice --tool-call-parser qwen3_xml
    ;;
  q36-moe)
    # 35B-A3B MoE FP8 ~36 GB → offload 16 GB. Only 3B active so offload cost
    # is small compared to a dense model. Same FP8+offload caveat.
    # --enforce-eager required because cpu-offload uva.py is not Dynamo-traceable.
    exec vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name qwen3.6-moe \
      --max-model-len 4096 \
      --max-num-batched-tokens 2048 \
      --gpu-memory-utilization 0.88 \
      --cpu-offload-gb 16 \
      --enable-prefix-caching \
      --enforce-eager \
      --enable-auto-tool-choice --tool-call-parser qwen3_xml
    ;;
  devstral)
    # 24B FP16 ~48 GB. Needs 24 GB GPU + 24 GB CPU. WSL defaults to 50% of
    # host RAM (~15 GB on 32 GB system) → won't fit. To enable: edit
    # %USERPROFILE%/.wslconfig and set:
    #   [wsl2]
    #   memory=28GB
    # then `wsl --shutdown` from PowerShell to apply.
    # --enforce-eager required because cpu-offload uva.py is not Dynamo-traceable.
    exec vllm serve mistralai/Devstral-Small-2507 \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name devstral \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.88 \
      --cpu-offload-gb 26 \
      --enforce-eager \
      --enable-auto-tool-choice --tool-call-parser mistral \
      --chat-template /home/folar/vllm/templates/tool_chat_template_mistral_parallel.jinja
    ;;
  ""|-h|--help)
    cat <<USAGE
serve.sh — start a vLLM OpenAI-compatible server on port \$PORT (default 8000).

Fits single 3090 (24 GB) no offload:
  smoke         Qwen2.5-0.5B-Instruct          (install check, ~1 GB)
  vl7           Qwen2.5-VL-7B-Instruct         (vision, ~16 GB)
  coder14       Qwen2.5-Coder-14B-Instruct-AWQ (~9 GB)
  gemma4        gemma-4-26B-A4B-it-AWQ-4bit    (Google MoE, ~15 GB)
  q35-27b       Qwen3.5-27B-AWQ                (~16 GB)
  q3-32b        Qwen3-32B-AWQ                  (~20 GB)
  vl32          Qwen2.5-VL-32B-Instruct-AWQ    (vision, ~19 GB)
  coder32       Qwen2.5-Coder-32B-Instruct-AWQ (coder, ~19 GB)
  coder30-awq   Qwen3-Coder-30B-A3B-AWQ        (MoE, ~17 GB, auto-DLs)

Needs CPU offload (slower, uses WSL RAM):
  coder30-fp8   Qwen3-Coder-30B-A3B-FP8        (~30 GB, 10 GB offload)
  q36-27b       Qwen3.6-27B-FP8                (~28 GB, 8 GB offload)
  q36-moe       Qwen3.6-35B-A3B-FP8            (~36 GB, 16 GB offload)
  devstral      Devstral-Small-2507 FP16       (~48 GB, requires WSL RAM ≥28 GB)

Kills any existing 'vllm serve' before launching.
Defaults: PORT=11435, HOST=0.0.0.0 (LAN-reachable via WSL mirrored networking).
Override:  PORT=9000 HOST=127.0.0.1 bash serve.sh <alias>
LAN endpoint:  http://192.168.1.126:11435/v1   (next to Ollama on :11434)
USAGE
    exit 1
    ;;
  *)
    echo "Unknown alias: $ALIAS" >&2
    exec "$0" --help
    ;;
esac
