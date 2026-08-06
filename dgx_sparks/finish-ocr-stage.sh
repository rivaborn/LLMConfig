#!/usr/bin/env bash
# Runs ON spark3 (.52). Waits for the two OCR downloads to finish, then fans
# them out to the other three nodes and verifies.
#
# Same reasoning as finish-stage.sh: it lives on the NODE because a multi-hour
# watcher in .40's WSL would die to WSL2's idle-shutdown (invariant 4).
#
# Fan-out route from spark3 — mirror image of the GLM one, because spark3 is in
# the OTHER fabric pair:
#     spark3 --200G--> spark4          (192.168.0.53)
#     spark3 --1GbE--> spark1 --200G--> spark2
# Measured 2026-08-06: 200G intra-pair 585 MB/s, 1GbE cross-pair 105 MB/s. At
# ~11.2 GiB total both hops are ~2 min, so this is not worth optimising
# further — it is shaped this way only to avoid pushing the same bytes over
# 1GbE twice.
set -uo pipefail

CACHE=/home/fksogbetun/.cache/huggingface
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
RS_E="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
PIDF=/tmp/ocr-stage.pid
DONE=/tmp/ocr-stage.DONE
MODELS=(models--datalab-to--surya-ocr-2 models--datalab-to--chandra-ocr-2)
# expected sizes in bytes, from the HF tree API (1.30 GiB / 9.88 GiB)
MIN_BYTES=(1300000000 10000000000)
log() { echo "[$(date -u +%F' '%H:%M:%S)] $*"; }

if [ -f "$PIDF" ]; then
  pid=$(cat "$PIDF"); log "waiting on download pid $pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
  log "download process exited"
fi

# Refuse to fan out a partial copy — the whole point of a stage is that the
# other nodes get something known-good, not something plausible.
fail=0
for i in 0 1; do
  d="$CACHE/hub/${MODELS[$i]}"
  inc=$(find "$d" -name '*.incomplete' 2>/dev/null | wc -l)
  b=$(du -sb "$d" 2>/dev/null | cut -f1)
  log "${MODELS[$i]}: bytes=${b:-0} incomplete=$inc"
  if [ "$inc" -ne 0 ] || [ "${b:-0}" -lt "${MIN_BYTES[$i]}" ]; then
    log "  INCOMPLETE — not fanning this one out"; fail=1
  fi
done
if [ "$fail" -ne 0 ]; then
  echo "FAILED incomplete_or_short" > "$DONE"
  log "re-run the download (it resumes), then re-launch this script"; exit 1
fi

# -H preserves the cache's hardlinks (snapshots/ -> blobs/).
for m in "${MODELS[@]}"; do
  log "$m -> spark4 (200G) and spark1 (1GbE)"
  rsync -aH --partial -e "$RS_E" "$CACHE/hub/$m" fksogbetun@192.168.0.53:"$CACHE/hub/" & a=$!
  rsync -aH --partial -e "$RS_E" "$CACHE/hub/$m" fksogbetun@192.168.1.50:"$CACHE/hub/" & b=$!
  wait $a; ra=$?; wait $b; rb=$?
  log "  spark4 rc=$ra  spark1 rc=$rb"
  if [ "$rb" -eq 0 ]; then
    log "$m: spark1 -> spark2 (200G)"
    $SSH fksogbetun@192.168.1.50 \
      "rsync -aH --partial -e '$RS_E' '$CACHE/hub/$m' fksogbetun@192.168.0.51:'$CACHE/hub/'"
    log "  spark2 rc=$?"
  else
    log "$m: skipping spark2 — its source (spark1) did not complete"
  fi
done

log "verification"
{
  printf '%-14s %-30s %10s %8s %6s\n' NODE MODEL SIZE FILES INC
  for h in 192.168.1.50 192.168.1.51 192.168.1.52 192.168.1.53; do
    for m in "${MODELS[@]}"; do
      if [ "$h" = "192.168.1.52" ]; then
        s=$(du -sh "$CACHE/hub/$m" 2>/dev/null | cut -f1)
        f=$(find "$CACHE/hub/$m" -type f 2>/dev/null | wc -l)
        i=$(find "$CACHE/hub/$m" -name '*.incomplete' 2>/dev/null | wc -l)
      else
        read -r s f i < <($SSH fksogbetun@"$h" \
          "echo \"\$(du -sh $CACHE/hub/$m 2>/dev/null | cut -f1) \$(find $CACHE/hub/$m -type f 2>/dev/null | wc -l) \$(find $CACHE/hub/$m -name '*.incomplete' 2>/dev/null | wc -l)\"")
      fi
      printf '%-14s %-30s %10s %8s %6s\n' "$h" "$m" "${s:-MISSING}" "${f:-0}" "${i:-0}"
    done
  done
} | tee "$DONE"
log "OCR STAGE COMPLETE — summary in $DONE"
