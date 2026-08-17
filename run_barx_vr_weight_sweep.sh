#!/usr/bin/env bash

# Runs the barx front-only + visual-robust contrastive training once per contrastive weight, back to
# back in a single process so the next one starts the moment the previous finishes -- no polling gap
# and nothing to re-prepare (every dataset involved is already built, so each run's prep step is a
# few-second no-op).
#
# Only the weight changes between runs; data, cameras, batch, views and objective are identical, so
# the sweep isolates it. The job name carries the weight, so output dirs and wandb runs stay distinct.
#
# Usage:
#   ./run_barx_vr_weight_sweep.sh            # 0.1 then 0.2
#   WEIGHTS="0.1 0.2 0.5" ./run_barx_vr_weight_sweep.sh

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
WEIGHTS="${WEIGHTS:-0.1 0.2}"
SUMMARY="${SCRIPT_DIR}/outputs/logs/barx_vr_weight_sweep.log"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${SUMMARY}"; }

log "=================================================================================="
log "barx + visual robust contrastive: weight sweep  (${WEIGHTS})"
log "=================================================================================="

summary=()
for w in ${WEIGHTS}; do
  run_log="${SCRIPT_DIR}/outputs/logs/barx_vr_w${w}_$(date +%Y%m%d_%H%M%S).log"
  started=$(date +%s)
  log "### starting weight=${w} -> ${run_log}"

  VISUAL_ROBUST_CONTRASTIVE_WEIGHT="${w}" \
    "${SCRIPT_DIR}/run_barx_visualrobust_frontonly.sh" >"${run_log}" 2>&1
  status=$?

  elapsed=$(( $(date +%s) - started ))
  human="$((elapsed / 3600))h$(( (elapsed % 3600) / 60 ))m"

  if [[ ${status} -eq 0 ]] && grep -q "End of training" "${run_log}" 2>/dev/null; then
    final="$(tr '\r' '\n' < "${run_log}" | grep -oE 'loss:[0-9.]+' | tail -1)"
    log "### weight=${w} OK (${human})  ${final}"
    summary+=("w=${w}: OK    (${human})  ${final}  ${run_log}")
  else
    log "### weight=${w} FAILED exit ${status} (${human}) -- continuing with the rest"
    log "last lines:"
    tail -15 "${run_log}" | tee -a "${SUMMARY}"
    summary+=("w=${w}: FAILED exit ${status} (${human})  ${run_log}")
  fi
done

log "=================================================================================="
log "weight sweep finished"
for line in "${summary[@]}"; do
  log "  ${line}"
done
log "  free disk: $(df -h --output=avail "${SCRIPT_DIR}" | tail -1 | tr -d ' ')"
log "=================================================================================="
