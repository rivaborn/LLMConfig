#!/usr/bin/env bash
# Stage a HuggingFace model into the HF cache on every DGX Spark, plus the
# container image the recipe pins — so a launch is engine start only.
#
# RUN FROM .40's WSL:
#   bash /mnt/c/Coding/rivaborn/LLMConfig/dgx_sparks/stage-model.sh <repo> <revision> [image]
#
# WHY THIS SHAPE — the fabric dictates it (re-verified 2026-08-06 by ping matrix):
#
#     spark1(.50) <-> spark2(.51)   200G, on BOTH fabric subnets
#     spark3(.52) <-> spark4(.53)   200G, on BOTH fabric subnets
#     pair  <->  pair               NO PATH — 1GbE management only
#
#   Both 200G subnets (192.168.0.x / 192.168.2.x) carry all four node addresses
#   and look mutually reachable. They are dual-rail WITHIN a pair. Do not
#   "optimise" this into a single download plus a fan-out to all three peers:
#   two of those hops would silently fall back to the 1GbE management link.
#
#   So: download ONCE PER PAIR from HF (both pair heads in parallel), then fan
#   out INTRA-PAIR over 200G. Half the internet transfer of four independent
#   downloads, and no 383 GiB ever crosses 1GbE.
#
# Idempotent and resumable: `hf download` skips complete blobs, and the
# intra-pair copy is rsync. Re-run it after an interruption.
set -uo pipefail

REPO="${1:?usage: stage-model.sh <hf-repo> <revision> [container-image]}"
REV="${2:?missing revision — always stage a pinned revision, never a moving branch}"
IMAGE="${3:-}"

SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
USER_SP="fksogbetun"
# pair head -> peer, and the peer's address ON THE 200G FABRIC (not management)
PAIR_HEADS=("192.168.1.50" "192.168.1.52")
declare -A PEER_MGMT=( ["192.168.1.50"]="192.168.1.51" ["192.168.1.52"]="192.168.1.53" )
declare -A PEER_FAST=( ["192.168.1.50"]="192.168.0.51" ["192.168.1.52"]="192.168.0.53" )

CACHE="/home/${USER_SP}/.cache/huggingface"
SNAP_DIR="models--${REPO//\//--}"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# --- 0. preflight -----------------------------------------------------------
log "staging $REPO @ $REV"
for h in "${PAIR_HEADS[@]}" "${PEER_MGMT[@]}"; do
  if ! $SSH "${USER_SP}@${h}" true 2>/dev/null; then
    log "FATAL: cannot ssh to $h"; exit 1
  fi
done
log "all four nodes reachable"

# --- 1. download once per pair head, in parallel ----------------------------
# uvx keeps huggingface_hub ephemeral — nothing permanent is installed on the
# nodes.
#
# HF_HUB_DISABLE_XET=1 on purpose. This repo IS Xet-backed and current
# huggingface_hub nags to enable HF_XET_HIGH_PERFORMANCE, but Xet transfers have
# HUNG on this network before, and a hang costs more than the speedup is worth
# on a multi-hundred-GiB stage. (HF_HUB_ENABLE_HF_TRANSFER is deliberately NOT
# set: it is a deprecated no-op in this version and only produces a warning.)
#
# Throughput comes from --max-workers, i.e. parallel FILES, not parallel chunks
# per file: measured 2026-08-06, a single 4.78 GiB file pulls at only ~4.6 MB/s,
# so per-connection rate is the limit and concurrency is what recovers it. This
# checkpoint's per-layer layout (75 files ~4.8 GiB each) suits that well.
dl_pids=()
for head in "${PAIR_HEADS[@]}"; do
  log "download -> $head (background; progress: tail /tmp/stage-*.log on that node)"
  $SSH "${USER_SP}@${head}" \
    "export PATH=\$HOME/.local/bin:\$PATH HF_HOME=$CACHE HF_HUB_DISABLE_XET=1; \
     uvx --quiet --from 'huggingface_hub[cli]' hf download '$REPO' --revision '$REV' \
       --max-workers 8 > /tmp/stage-${REPO//\//_}.log 2>&1" &
  dl_pids+=($!)
done
rc=0
for p in "${dl_pids[@]}"; do wait "$p" || rc=1; done
if [ "$rc" -ne 0 ]; then
  log "WARNING: at least one download returned non-zero — check /tmp/stage-*.log on the heads"
fi
log "pair-head downloads finished (rc=$rc)"

# --- 2. fan out intra-pair over the 200G link -------------------------------
for head in "${PAIR_HEADS[@]}"; do
  peer_fast="${PEER_FAST[$head]}"
  log "fan out $head -> $peer_fast (200G)"
  # -H preserves the hardlink structure of the HF cache (blobs/ <- snapshots/);
  # without it the copy silently doubles on-disk size.
  $SSH "${USER_SP}@${head}" \
    "rsync -aH --info=progress2 --partial \
       -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new' \
       '$CACHE/hub/$SNAP_DIR' '${USER_SP}@${peer_fast}:$CACHE/hub/'" \
    2>&1 | tail -2
done

# --- 3. pre-pull the container image on all four ----------------------------
if [ -n "$IMAGE" ]; then
  for h in "${PAIR_HEADS[@]}" "${PEER_MGMT[@]}"; do
    log "docker pull on $h"
    $SSH "${USER_SP}@${h}" "docker pull '$IMAGE'" 2>&1 | tail -1
  done
fi

# --- 4. verify --------------------------------------------------------------
log "verification"
printf '%-16s %10s %8s %10s\n' NODE SIZE FILES INCOMPLETE
for h in "${PAIR_HEADS[@]}" "${PEER_MGMT[@]}"; do
  read -r sz files incomplete < <($SSH "${USER_SP}@${h}" \
    "d=$CACHE/hub/$SNAP_DIR; \
     s=\$(du -sh \$d 2>/dev/null | cut -f1); \
     f=\$(find \$d -type f 2>/dev/null | wc -l); \
     i=\$(find \$d -name '*.incomplete' 2>/dev/null | wc -l); \
     echo \"\${s:-MISSING} \${f:-0} \${i:-0}\"")
  printf '%-16s %10s %8s %10s\n' "$h" "$sz" "$files" "$incomplete"
done
log "done — compare against dgx_sparks/models/<model>/file-inventory.tsv"
