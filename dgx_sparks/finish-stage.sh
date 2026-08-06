#!/usr/bin/env bash
# Runs ON THE SEED NODE (spark1). Waits for the detached `hf download` to
# finish, then completes the stage unattended: LAN fan-out, container pull,
# verification, and a DONE marker.
#
# Deployed and launched by hand as:
#   scp finish-stage.sh fksogbetun@192.168.1.50:/tmp/
#   ssh ... "setsid nohup bash /tmp/finish-stage.sh <repo> <image> > /tmp/finish.log 2>&1 &"
#
# It lives on the NODE rather than on .40 because .40's WSL idle-shuts-down
# seconds after the last wsl.exe exits (CLAUDE.md invariant 4) and would kill a
# multi-hour watcher. The Sparks are always-on Linux boxes.
#
# Why the fan-out is shaped this way — measured 2026-08-06, see stage-model.sh:
# the internet uplink is ~11.7 MB/s aggregate while the 1GbE management hop is
# 105 MB/s and the 200G intra-pair hop 585 MB/s. Pull once, then move it locally.
set -uo pipefail

REPO="${1:?repo}"
IMAGE="${2:-}"
SNAP="models--${REPO//\//--}"
CACHE="/home/fksogbetun/.cache/huggingface"
DIR="$CACHE/hub/$SNAP"
PIDF="/tmp/stage-$(echo "$REPO" | tr / _).pid"
[ -f "$PIDF" ] || PIDF=/tmp/glm52-stage.pid          # the ad-hoc launch's name
DONE=/tmp/glm52-stage.DONE
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
RS_E="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
log() { echo "[$(date -u +%F' '%H:%M:%S)] $*"; }

# ---- 1. wait for the download ---------------------------------------------
if [ -f "$PIDF" ]; then
  pid=$(cat "$PIDF")
  log "waiting on download pid $pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
  log "download process exited"
else
  log "no pidfile — assuming the download already finished"
fi

# ---- 2. sanity-check before copying 383 GiB of possibly-partial data -------
inc=$(find "$DIR" -name '*.incomplete' 2>/dev/null | wc -l)
files=$(find "$DIR" -type f 2>/dev/null | wc -l)
bytes=$(du -sb "$DIR" 2>/dev/null | cut -f1)
log "seed state: files=$files incomplete=$inc bytes=$bytes"
if [ "$inc" -ne 0 ]; then
  log "ABORT: $inc incomplete blob(s) — the download did not finish cleanly."
  log "re-run the download phase (it resumes), then re-launch this script."
  echo "FAILED incomplete=$inc" > "$DONE"; exit 1
fi
if [ "${bytes:-0}" -lt 400000000000 ]; then
  log "ABORT: only $bytes bytes, expected ~412 GB (383.7 GiB)."
  echo "FAILED short bytes=$bytes" > "$DONE"; exit 1
fi

# ---- 3. fan out over LAN ---------------------------------------------------
# -H preserves the cache's hardlinks (snapshots/ -> blobs/); without it the copy
# silently doubles on-disk size.
log "fan out: -> spark2 over 200G and -> spark3 over 1GbE, in parallel"
rsync -aH --partial -e "$RS_E" "$DIR" fksogbetun@192.168.0.51:"$CACHE/hub/" & p1=$!
rsync -aH --partial -e "$RS_E" "$DIR" fksogbetun@192.168.1.52:"$CACHE/hub/" & p2=$!
wait $p1; r1=$?
wait $p2; r2=$?
log "spark2 rsync rc=$r1, spark3 rsync rc=$r2"

if [ "$r2" -eq 0 ]; then
  log "spark3 -> spark4 over 200G"
  $SSH fksogbetun@192.168.1.52 \
    "rsync -aH --partial -e '$RS_E' '$DIR' fksogbetun@192.168.0.53:'$CACHE/hub/'"
  log "spark4 rsync rc=$?"
else
  log "skipping spark4 — its source (spark3) did not complete"
fi

# ---- 4. pre-pull the pinned container image --------------------------------
if [ -n "$IMAGE" ]; then
  for h in 192.168.1.50 192.168.1.51 192.168.1.52 192.168.1.53; do
    log "docker pull on $h"
    if [ "$h" = "192.168.1.50" ]; then docker pull "$IMAGE" >/dev/null 2>&1 && log "  ok" || log "  FAILED"
    else $SSH fksogbetun@"$h" "docker pull '$IMAGE'" >/dev/null 2>&1 && log "  ok" || log "  FAILED"; fi
  done
fi

# ---- 5. verify every node --------------------------------------------------
log "verification"
{
  printf '%-16s %10s %8s %12s\n' NODE SIZE FILES INCOMPLETE
  for h in 192.168.1.50 192.168.1.51 192.168.1.52 192.168.1.53; do
    if [ "$h" = "192.168.1.50" ]; then
      s=$(du -sh "$DIR" 2>/dev/null | cut -f1); f=$(find "$DIR" -type f 2>/dev/null | wc -l)
      i=$(find "$DIR" -name '*.incomplete' 2>/dev/null | wc -l)
    else
      read -r s f i < <($SSH fksogbetun@"$h" \
        "echo \"\$(du -sh $DIR 2>/dev/null | cut -f1) \$(find $DIR -type f 2>/dev/null | wc -l) \$(find $DIR -name '*.incomplete' 2>/dev/null | wc -l)\"")
    fi
    printf '%-16s %10s %8s %12s\n' "$h" "${s:-MISSING}" "${f:-0}" "${i:-0}"
  done
} | tee "$DONE"
log "STAGE COMPLETE — summary in $DONE"
