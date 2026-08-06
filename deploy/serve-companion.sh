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
#      TOTAL. Measured 2026-07-31: WSL charges ~1.5 GiB of non-torch CUDA
#      context PER PROCESS, so small models are overhead-dominated here.
#      Fit: surya2 0.50 + relay 0.40 (the relay profiled -0.46 GiB at 0.35).
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
    # from serve.sh's 3090 case with the 8 GB retune. Weights are only 1.37 GiB
    # (measured) — the budget went to the 6.3 MP multimodal profile (-0.45 GiB
    # at util 0.50) and then to the 18k bf16 KV pool (-0.0 after the max_pixels
    # cut). Final fit 2026-07-31: max_pixels 2359296 (~1536x1536, ample for
    # book pages), 1 image/prompt, window 12288 (epubocr's real need is ~9k:
    # ~2.3k vision tokens + prompt + the 6144 output cap), util 0.54 against
    # the ~4.5 GiB the shrunk relay leaves free. KV stays bf16 — fp8 KV would
    # nudge the logprobs epubocr's confidence floor reads.
    PORT=11439
    pkill -f "vllm serve.*--port ${PORT}" 2>/dev/null || true
    exec vllm serve datalab-to/surya-ocr-2 \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name surya-ocr-2 \
      --max-model-len 12288 \
      --dtype bfloat16 \
      --gpu-memory-utilization 0.54 \
      --enforce-eager \
      --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":2359296}' \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --enable-prefix-caching
    ;;
  paddleocr-vl)
    # PaddleOCR-VL 1.6 (0.9B) — TOOK OVER surya2's slot on 2026-08-06.
    #
    # Same relay port (11438) and same internal PORT (11439) as the surya2 case
    # above, ON PURPOSE: the socat relay is a pure port forward, so reusing
    # 11439 means the relay wiring is unchanged. The unit was renamed
    # (vllm-companion-relay-paddleocr) rather than left saying "surya2", since
    # a unit whose description lies is a debugging trap.
    #
    # ⚠️ surya2 IS STILL DEFINED ABOVE but has NO SLOT — it was removed from
    # COMPANION_VLLM_SLOTS to make room. Slots are static config; the case is
    # kept so reverting is a one-line .env change. surya-ocr-2 remains reachable
    # on the 3090 (serve.sh `surya2`) and on all four Sparks (staged
    # 2026-08-06), which is what made this swap affordable.
    #
    # ⚠️ epubocr pointed SURYA_INFERENCE_URL at :11438 and read surya's
    # per-token confidence. This port now answers with a DIFFERENT model under
    # a different name — epubocr must be re-pointed at the 3090 or a Spark.
    #
    # 8 GB RETUNE vs the 3090's paddleocr-vl case:
    #   window 16384 -> 8192   this model reads ONE document region per call,
    #                          so a short window costs nothing real and buys KV
    #                          room on a card that has none to spare.
    #   --enforce-eager        as with every slot here: CUDA-graph memory is
    #                          overhead this card cannot afford.
    #   util 0.54              the number proven for THIS 4300 MB slot by the
    #                          surya2 fit. Note paddle's weights are ~0.4 GiB
    #                          BIGGER (1.79 vs 1.37 measured), so KV headroom is
    #                          correspondingly tighter — if it fails the memory
    #                          profile, cut max-model-len before touching util,
    #                          which is what the sibling relay's budget depends on.
    #
    # NO --mm-processor-kwargs: min_pixels/max_pixels are QWEN kwargs and belong
    # to the surya2 case above, not here (this model has its own processor).
    PORT=11439
    pkill -f "vllm serve.*--port ${PORT}" 2>/dev/null || true
    exec vllm serve PaddlePaddle/PaddleOCR-VL-1.6 \
      --revision cdc88f5feff0e4079e75863205053a68358e52f7 \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name paddleocr-vl-1.6 \
      --trust-remote-code \
      --max-model-len 8192 \
      --dtype bfloat16 \
      --gpu-memory-utilization 0.54 \
      --enforce-eager \
      --mm-processor-cache-gb 0 \
      --enable-prefix-caching
    ;;
  qwen25-relay)
    # The opencode /swap echo relay, vLLM-served (replaces the Ollama tag
    # qwen2.5:1.5b — Ollama declares no budget and cannot join a budgeted
    # card). AWQ ~1.1 GB weights; 32k window so opencode's ~24.5k baseline
    # prompt fits (bf16 1.5B + 32k KV does NOT fit beside surya2 — that is why
    # this is the AWQ build).
    #
    # Measured 2026-07-31 (slot cutover, three profile runs): 0.25 → -2.19 GiB
    # available KV; 0.35 + fp8 KV + 2048-token profile → -0.46; 0.40 → -0.04.
    # The blocker is structural: WSL charges ~2 GiB of non-torch context +
    # activations per process, and a 32k window needs ~450 MiB of KV in one
    # piece — the sum exceeds what surya2 leaves free at any legal util.
    # Fix from vl32's own playbook: OFFLOAD the weights (~1.1 GiB) to CPU.
    # This card is chassis PCIe (not the Thunderbolt eGPU the offload warning
    # is about) and a 1.5B relay tolerates the latency. GPU footprint becomes
    # context + KV only, which fits at 0.35 with room to spare.
    PORT=11440
    pkill -f "vllm serve.*--port ${PORT}" 2>/dev/null || true
    exec vllm serve Qwen/Qwen2.5-1.5B-Instruct-AWQ \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name qwen2.5-1.5b \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.30 \
      --cpu-offload-gb 1 \
      --kv-cache-dtype fp8_e5m2 \
      --max-num-batched-tokens 2048 \
      --max-num-seqs 4 \
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
