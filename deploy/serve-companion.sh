#!/usr/bin/env bash
# Companion-lane model launcher — RTX 3070 Ti (8 GB), SLOT mode.
# Usage:  bash serve-companion.sh <alias>        (systemd: vllm-companion@<alias>)
#
# SLOT MODE — this card holds SEVERAL vLLM processes at once (LLMConfig's
# SlotLane, COMPANION_VLLM_SLOTS). Three rules fall out of that:
#   1. Each alias owns a FIXED internal port; the socat relay map mirrors it:
#        surya2        vllm :11439   relay :11438
#        qwen25-relay  vllm :11440   relay :11441
#   2. NO global pkill. serve.sh's "kill any vllm first" would take the sibling
#      slot down on every launch. systemd owns lifecycle (vllm-companion@<alias>);
#      the only defensive kill below is scoped to THIS alias's own port.
#   3. Budgets are per-alias --gpu-memory-utilization fractions of the 8 GB
#      TOTAL and must sum <= ~0.80 across resident slots (driver baseline ~600 MiB
#      + two CUDA contexts eat the rest). surya2 0.55 + relay 0.25 = 0.80.
set -euo pipefail

ALIAS="${1:-}"
HOST="${HOST:-0.0.0.0}"

# Activate venv if not already.
if [ -z "${VIRTUAL_ENV:-}" ]; then
  # shellcheck disable=SC1091
  source "$HOME/vllm/.venv/bin/activate"
fi

# Select the RTX 3070 Ti by UUID, resolved to its current integer index — the
# same dance as serve.sh's 3090 block: vLLM rejects a raw GPU-<uuid> (ModelConfig
# int() parse) and its worker ignores CUDA_DEVICE_ORDER (CUDA FASTEST_FIRST puts
# the 3090 first whenever it is visible), so the index must be looked up live.
# Hard-fail if absent: launching an 8 GB-budgeted model on the 24 GB card is how
# a 30B coder once ended up on the wrong GPU — refuse instead.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  _GPU_IDX="$(python - <<'PYEOF'
import torch
for i in range(torch.cuda.device_count()):
    if "2caf7863" in str(getattr(torch.cuda.get_device_properties(i), "uuid", "")):
        print(i); break
PYEOF
)"
  if [ -z "${_GPU_IDX}" ]; then
    echo "serve-companion.sh: RTX 3070 Ti (GPU-2caf7863-...) not found in torch CUDA enumeration; refusing to launch on the wrong GPU." >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="${_GPU_IDX}"
fi

export HF_HUB_ENABLE_HF_TRANSFER=1
# Same profiler workaround as serve.sh — CUDA-graph profile memory otherwise
# counts against the (small) per-slot utilization budgets.
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
if [ -d /usr/local/cuda-13.0/bin ]; then
  export CUDA_HOME=/usr/local/cuda-13.0
  case ":$PATH:" in *":$CUDA_HOME/bin:"*) ;; *) export PATH="$CUDA_HOME/bin:$PATH" ;; esac
fi

case "$ALIAS" in
  surya2)
    # Surya 2 OCR VLM for epubocr — must be vLLM (epubocr consumes logprobs for
    # per-block confidence; needs --mm-processor-kwargs pixel bounds). Copied
    # from serve.sh's 3090 case with the 8 GB retune: utilization 0.85 -> 0.55
    # (~4.5 GB — the 3090 figure bought ~18 GB of KV pool this model never
    # needs). OOM fallback ladder, in order: max_pixels 6291456 -> 2359296,
    # then utilization 0.60 with the relay dropped to a 0.5B model.
    PORT=11439
    pkill -f "vllm serve.*--port ${PORT}" 2>/dev/null || true
    exec vllm serve datalab-to/surya-ocr-2 \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name surya-ocr-2 \
      --max-model-len 18000 \
      --dtype bfloat16 \
      --gpu-memory-utilization 0.55 \
      --enforce-eager \
      --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":6291456}' \
      --enable-prefix-caching
    ;;
  qwen25-relay)
    # The opencode /swap echo relay, vLLM-served (replaces the Ollama tag
    # qwen2.5:1.5b — Ollama declares no budget and cannot join a budgeted
    # card). AWQ ~1.1 GB weights; 32k window so opencode's ~24.5k baseline
    # prompt fits (bf16 1.5B + 32k KV does NOT fit beside surya2 — that is why
    # this is the AWQ build).
    PORT=11440
    pkill -f "vllm serve.*--port ${PORT}" 2>/dev/null || true
    exec vllm serve Qwen/Qwen2.5-1.5B-Instruct-AWQ \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name qwen2.5-1.5b \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.25 \
      --enforce-eager \
      --enable-prefix-caching
    ;;
  smoke)
    # Sanity check / fallback relay (0.5B, ~1 GB). No fixed slot by default —
    # give it one in COMPANION_VLLM_SLOTS before LLMConfig can drive it.
    PORT=11442
    pkill -f "vllm serve.*--port ${PORT}" 2>/dev/null || true
    exec vllm serve Qwen/Qwen2.5-0.5B-Instruct \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name smoke \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.15 \
      --enforce-eager
    ;;
  ""|-h|--help)
    cat <<USAGE
serve-companion.sh — vLLM on the RTX 3070 Ti (8 GB), one process per SLOT.

Aliases (fixed ports; the socat relays mirror them):
  surya2        datalab-to/surya-ocr-2          :11439  util 0.55  (OCR for epubocr)
  qwen25-relay  Qwen2.5-1.5B-Instruct-AWQ       :11440  util 0.25  (opencode relay)
  smoke         Qwen2.5-0.5B-Instruct           :11442  util 0.15  (sanity/fallback)

Unlike serve.sh this NEVER kills other vllm processes — slots co-reside and
systemd (vllm-companion@<alias>) owns each one's lifecycle.
USAGE
    exit 0
    ;;
  *)
    echo "serve-companion.sh: unknown alias '$ALIAS' (see --help)" >&2
    exit 1
    ;;
esac
