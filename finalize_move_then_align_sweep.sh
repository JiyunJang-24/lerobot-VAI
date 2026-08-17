#!/usr/bin/env bash

# 1. Waits for the in-flight rsync of dataset_git to /dataset/jiyun/ to finish.
# 2. Verifies the copy (file count + byte count) and only then removes the original.
# 3. Replaces dataset_git with a symlink to the new location.
# 4. Runs the alignment-loss weight sweep (0.1 then 0.5) back to back.
#
# The verify-before-delete order matters: /dataset is an NFS mount, so this is a cross-filesystem
# copy, not a rename. A partial transfer that got deleted anyway would lose 54 GB of prepared
# datasets that took hours of conversion and re-encoding to build.
#
# Training must not start until the symlink is in place, since every run reads dataset_git.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
SRC="${SCRIPT_DIR}/dataset_git"
DST="${DST:-/dataset/jiyun/dataset_git}"
SUMMARY="${SCRIPT_DIR}/outputs/logs/move_then_align_sweep.log"
WEIGHTS="${WEIGHTS:-0.1 0.5}"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${SUMMARY}"; }

log "=== waiting for rsync to finish ==="
# -x matches the process NAME, not the command line. `pgrep -f "rsync -a --info"` also matches any
# shell whose own command line happens to contain that string -- including the shell running this
# very check -- so the loop waited on itself indefinitely while the real rsync had long since exited.
while pgrep -x rsync >/dev/null 2>&1; do
  sleep 60
done
log "rsync finished"

if [[ -L "${SRC}" ]]; then
  log "dataset_git is already a symlink -> $(readlink "${SRC}"); skipping the move"
else
  log "=== verifying the copy before deleting anything ==="
  src_files="$(find "${SRC}" -type f | wc -l)"
  dst_files="$(find "${DST}" -type f | wc -l)"
  src_bytes="$(du -sb "${SRC}" | cut -f1)"
  dst_bytes="$(du -sb "${DST}" | cut -f1)"
  log "  source: ${src_files} files, ${src_bytes} bytes"
  log "  copy  : ${dst_files} files, ${dst_bytes} bytes"

  if [[ "${src_files}" -ne "${dst_files}" || "${src_bytes}" -ne "${dst_bytes}" ]]; then
    log "MISMATCH -- leaving the original untouched and NOT starting training."
    log "Re-run the rsync to fill the gap, then run this script again."
    exit 1
  fi
  log "verified: counts and bytes match exactly"

  log "=== replacing dataset_git with a symlink ==="
  rm -rf "${SRC}"
  ln -s "${DST}" "${SRC}"
  log "  $(ls -ld "${SRC}")"
fi

# Sanity: the symlink must resolve and still expose the datasets the runs need.
if [[ ! -f "${SRC}/barx_frontonly_p900_i1000_u1000/raw/panda_mg/meta/info.json" ]] \
   || [[ ! -f "${SRC}/visual_robust_new_barx/new_barx/PandaOmron_TurnOnSinkFaucet/lerobot/meta/info.json" ]]; then
  log "symlink does not expose the expected datasets; NOT starting training."
  exit 1
fi
log "symlink resolves and the training datasets are reachable"
log "free space on /: $(df -h --output=avail / | tail -1 | tr -d ' ')"

log "=================================================================================="
log "alignment weight sweep (${WEIGHTS})"
log "=================================================================================="

summary=()
for w in ${WEIGHTS}; do
  run_log="${SCRIPT_DIR}/outputs/logs/barx_vralign_w${w}_$(date +%Y%m%d_%H%M%S).log"
  started=$(date +%s)
  log "### starting alignment weight=${w} -> ${run_log}"

  VISUAL_ROBUST_FRONT_OBJECTIVE=alignment \
  VISUAL_ROBUST_HEAD_MODE=none \
  VISUAL_ROBUST_CONTRASTIVE_WEIGHT="${w}" \
    "${SCRIPT_DIR}/run_barx_visualrobust_frontonly.sh" >"${run_log}" 2>&1
  status=$?

  elapsed=$(( $(date +%s) - started ))
  human="$((elapsed / 3600))h$(( (elapsed % 3600) / 60 ))m"

  # The trainer's own output goes to the backoff wrapper's per-attempt log, not to this stdout, so
  # judge success by the exit status and pull the final numbers from the newest attempt log.
  attempt_log="$(ls -t "${SCRIPT_DIR}"/outputs/logs/vr_attempt1_*.log 2>/dev/null | head -1)"
  final="$(tr '\r' '\n' < "${attempt_log}" 2>/dev/null | grep -oE 'loss:[0-9.]+' | tail -1)"
  align="$(tr '\r' '\n' < "${attempt_log}" 2>/dev/null | grep -oE "'visual_robust_alignment_loss': [0-9.e-]+" | tail -1)"

  if [[ ${status} -eq 0 ]]; then
    log "### alignment w=${w} OK (${human})  ${final}  ${align}"
    summary+=("align w=${w}: OK    (${human})  ${final}  ${align}")
  else
    log "### alignment w=${w} FAILED exit ${status} (${human}) -- continuing"
    summary+=("align w=${w}: FAILED exit ${status} (${human})")
  fi
done

log "=================================================================================="
log "alignment sweep finished"
for line in "${summary[@]}"; do log "  ${line}"; done
log "=================================================================================="
