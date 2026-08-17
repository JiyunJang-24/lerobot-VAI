#!/usr/bin/env bash

# Runs the remaining barx front-only queue strictly one at a time:
#
#   1. (already running) frozen-encoder baseline   -- this script only WAITS for it
#   2. alignment weight 3.0
#   3. alignment weight 1.0
#
# Replaces the parallel arrangement: sharing the 8 GPUs cost the alignment run 53% of its throughput
# (1.86 -> 0.875 it/s) for a frozen run that only needs 9.2 GiB, so the wall-clock saving was not
# worth having two half-speed jobs.
#
# Waiting is done on the frozen run's PROCESS, not on a log marker: the run writes through `tee`, so a
# crash leaves the log without any terminal line at all and a marker-based wait would hang forever.
# The pattern is matched against the trainer's own --job_name, which no shell in this script carries,
# so the loop cannot match itself (a mistake that once made a wait loop block on its own cmdline).

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
SUMMARY="${SCRIPT_DIR}/outputs/logs/frozen_then_align_sweep.log"
WEIGHTS="${WEIGHTS:-3.0 1.0}"
FROZEN_PATTERN="${FROZEN_PATTERN:-job_name=smolvla_robocasa_x_barx_frontonly_frozenvis}"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${SUMMARY}"; }

log "=================================================================================="
log "queue: frozen baseline (running) -> alignment $(echo "${WEIGHTS}" | tr ' ' ',')"
log "=================================================================================="

if pgrep -f "${FROZEN_PATTERN}" >/dev/null 2>&1; then
  log "waiting for the frozen-encoder baseline to finish ..."
  while pgrep -f "${FROZEN_PATTERN}" >/dev/null 2>&1; do
    sleep 60
  done
  log "frozen-encoder baseline exited"
else
  log "no frozen-encoder run in flight; starting the sweep immediately"
fi

frozen_log="$(ls -t "${SCRIPT_DIR}"/outputs/logs/barx_frontonly_frozenvis_*.log 2>/dev/null | head -1)"
if [[ -n "${frozen_log}" ]]; then
  log "frozen run last line: $(tr '\r' '\n' < "${frozen_log}" | tail -1 | cut -c1-120)"
fi

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
log "queue finished"
for line in "${summary[@]}"; do log "  ${line}"; done
log "=================================================================================="
