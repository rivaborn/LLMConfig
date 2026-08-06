#!/usr/bin/env bash
# Stage a HuggingFace model into the HF cache on every DGX Spark, plus the
# container image the recipe pins — so a launch is engine start only.
#
# RUN FROM .40's WSL:
#   stage-model.sh download <repo> <revision>          # phase 1, ~hours
#   stage-model.sh fanout   <repo>                     # phase 2, ~1.3 h
#   stage-model.sh image    <image@sha256:...>         # phase 3, minutes
#   stage-model.sh verify   <repo>
#
# Phases are separate because phase 1 runs for hours: it is launched DETACHED on
# the node (setsid + pidfile) so it survives WSL idle-shutdown — WSL2 tears the
# distro down seconds after the last wsl.exe exits, which would otherwise kill
# the job (CLAUDE.md invariant 4).
#
# ── TOPOLOGY AND WHY THIS DOWNLOADS ONCE, NOT ONCE PER PAIR ──────────────────
#
# Fabric (re-verified by ping matrix 2026-08-06):
#     spark1(.50) <-> spark2(.51)   200G, on BOTH fabric subnets
#     spark3(.52) <-> spark4(.53)   200G, on BOTH fabric subnets
#     pair  <->  pair               NO PATH — 1GbE management only
# Both 200G subnets (192.168.0.x / 192.168.2.x) carry all four node addresses
# and look mutually reachable. They are dual-rail WITHIN a pair.
#
# MEASURED 2026-08-06 (this is what decides the shape):
#     HuggingFace WAN   ~6 MB/s per node, ~11.7 MB/s AGGREGATE — a shared
#                       ~100 Mbit uplink. --max-workers 8 genuinely opens 8
#                       files / ~60 connections and still does not exceed it,
#                       so concurrency cannot buy anything here.
#     200G intra-pair   585 MB/s   (ssh cipher-bound; a floor, not a ceiling)
#     1GbE cross-pair   105 MB/s   (line rate)
#
# So the 1GbE management hop — the link the "obvious" design avoids — is ~9x
# faster than the ENTIRE internet uplink. Downloading once per pair would pull
# 2 x 383.7 GiB over a 100 Mbit line (~19.4 h). Downloading once and fanning out
# over LAN costs ~9.8 h + ~1.3 h. Halve the WAN transfer, halve the wall clock.
#
#   RULE: pull from the internet exactly ONCE, then move it over ANY LAN link.
#   Do not "optimise" this back into per-pair downloads to avoid the 1GbE hop.
#
set -uo pipefail

PHASE="${1:?usage: stage-model.sh {download|fanout|image|verify} ...}"
SEED="192.168.1.50"                       # the one node that talks to HF
FAST_PEER="192.168.0.51"                  # spark2 over 200G  (from spark1)
CROSS="192.168.1.52"                      # spark3 over 1GbE mgmt (from spark1)
CROSS_FAST_PEER="192.168.0.53"            # spark4 over 200G  (from spark3)
ALL=("192.168.1.50" "192.168.1.51" "192.168.1.52" "192.168.1.53")
USER_SP="fksogbetun"
CACHE="/home/${USER_SP}/.cache/huggingface"

# -n is load-bearing, not tidiness: without it ssh reads its own stdin, and when
# this script is piped in (`wsl -- bash -s < stage-model.sh`) the FIRST ssh
# swallows the remainder of the script. The symptom is a run that silently does
# only its first step and exits 0. Calls that deliberately feed a heredoc drop
# -n and supply stdin explicitly.
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
SSH_IN="ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

case "$PHASE" in

download)
  REPO="${2:?repo}"; REV="${3:?revision — always pin, never a moving branch}"
  DIR="$CACHE/hub/models--${REPO//\//--}"
  # HF_HUB_DISABLE_XET=1 on purpose: this repo IS Xet-backed and huggingface_hub
  # nags for HF_XET_HIGH_PERFORMANCE, but Xet transfers have HUNG on this
  # network and a hang costs more than the speedup on a multi-hundred-GiB pull.
  # HF_HUB_ENABLE_HF_TRANSFER is deliberately unset — a deprecated no-op that
  # only emits a warning.
  #
  # The guard is a PIDFILE, not pgrep: this launcher's own command line contains
  # "hf download", so every pgrep pattern that matches the job also matches the
  # shell starting it. That false positive reported "already running" on a node
  # where nothing was, and skipped the launch entirely (2026-08-06).
  $SSH_IN "${USER_SP}@${SEED}" "bash -s" <<REMOTE
set -u
PIDF=/tmp/stage-\$(echo '$REPO' | tr / _).pid
LOG=/tmp/stage-\$(echo '$REPO' | tr / _).log
if [ -f "\$PIDF" ] && kill -0 "\$(cat \$PIDF)" 2>/dev/null; then
  echo "already running (pid \$(cat \$PIDF))"; exit 0
fi
export PATH=\$HOME/.local/bin:\$PATH HF_HOME=$CACHE HF_HUB_DISABLE_XET=1
setsid nohup uvx --quiet --from 'huggingface_hub[cli]' \
  hf download '$REPO' --revision '$REV' --max-workers 8 > "\$LOG" 2>&1 < /dev/null &
echo \$! > "\$PIDF"
sleep 5
kill -0 "\$(cat \$PIDF)" 2>/dev/null && echo "launched pid \$(cat \$PIDF), log \$LOG" \
  || { echo "FAILED TO START:"; tail -5 "\$LOG"; exit 1; }
REMOTE
  log "poll with: stage-model.sh verify $REPO"
  ;;

fanout)
  REPO="${2:?repo}"
  SNAP="models--${REPO//\//--}"
  # -H preserves the HF cache's hardlinks (snapshots/ -> blobs/). Without it the
  # copy silently doubles on-disk size.
  RS="rsync -aH --partial --info=progress2 -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new'"
  log "spark1 -> spark2 (200G) and spark1 -> spark3 (1GbE) in parallel"
  $SSH "${USER_SP}@${SEED}" "$RS '$CACHE/hub/$SNAP' '${USER_SP}@${FAST_PEER}:$CACHE/hub/'" 2>&1 | tail -2 &
  p1=$!
  $SSH "${USER_SP}@${SEED}" "$RS '$CACHE/hub/$SNAP' '${USER_SP}@${CROSS}:$CACHE/hub/'" 2>&1 | tail -2 &
  p2=$!
  wait $p1; wait $p2
  log "spark3 -> spark4 (200G)"
  $SSH "${USER_SP}@${CROSS}" "$RS '$CACHE/hub/$SNAP' '${USER_SP}@${CROSS_FAST_PEER}:$CACHE/hub/'" 2>&1 | tail -2
  log "fanout done"
  ;;

image)
  IMAGE="${2:?image ref, pin by @sha256: digest}"
  for h in "${ALL[@]}"; do
    log "docker pull on $h"
    $SSH "${USER_SP}@${h}" "docker pull '$IMAGE'" 2>&1 | tail -1
  done
  ;;

verify)
  REPO="${2:?repo}"
  DIR="$CACHE/hub/models--${REPO//\//--}"
  printf '%-16s %10s %8s %12s %s\n' NODE SIZE FILES INCOMPLETE RUNNING
  for h in "${ALL[@]}"; do
    out=$($SSH "${USER_SP}@${h}" \
      "s=\$(du -sh $DIR 2>/dev/null | cut -f1); \
       f=\$(find $DIR -type f 2>/dev/null | wc -l); \
       i=\$(find $DIR -name '*.incomplete' 2>/dev/null | wc -l); \
       r=\$(ps -eo args | grep -c 'huggingface_hub' ); \
       echo \"\${s:-MISSING} \${f:-0} \${i:-0} \$r\"")
    printf '%-16s %10s %8s %12s %s\n' "$h" $out
  done
  echo "compare against dgx_sparks/models/<model>/file-inventory.tsv (183 files, 383.72 GiB)"
  ;;

*) echo "unknown phase '$PHASE'"; exit 2;;
esac
