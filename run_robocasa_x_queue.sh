#!/usr/bin/env bash

# Runs train_smolVLA_robocasa_x.sh back-to-back for several per-robot episode mixes, one after the
# other on all 8 GPUs. Each mix gets its own DATASET_ROOT because prepare_robocasa_x_dataset.py
# refuses to reuse a raw/ dir that was built for a different episode count (it would otherwise abort
# with "already exists with N episodes, but M are needed"), and its own log under outputs/logs/.
#
# A failing run does not stop the queue -- the remaining mixes still run, and the summary at the end
# reports each one's exit status.
#
# Usage:
#   ./run_robocasa_x_queue.sh                 # runs the MIXES listed below
#   MIXES="500:500:500" ./run_robocasa_x_queue.sh   # override (panda:iiwa:ur5e per entry)

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
LOG_DIR="${SCRIPT_DIR}/outputs/logs"
mkdir -p "${LOG_DIR}"

# panda:iiwa:ur5e, in the order they should run.
MIXES="${MIXES:-1000:250:250 500:500:500}"

echo "=================================================================================="
echo "robocasa_x queue starting ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "  mixes (panda:iiwa:ur5e): ${MIXES}"
echo "=================================================================================="

summary=()
for mix in ${MIXES}; do
  panda="${mix%%:*}"
  rest="${mix#*:}"
  iiwa="${rest%%:*}"
  ur5e="${rest##*:}"

  tag="p${panda}_i${iiwa}_u${ur5e}"
  log="${LOG_DIR}/robocasa_x_${tag}_$(date +%Y%m%d_%H%M%S).log"
  started=$(date +%s)

  echo
  echo "### [$(date '+%H:%M:%S')] starting ${tag} -> ${log}"

  PANDA_TOTAL_EPISODES="${panda}" \
  IIWA_EPISODES="${iiwa}" \
  UR5E_EPISODES="${ur5e}" \
  DATASET_ROOT="${SCRIPT_DIR}/dataset_git/robocasa_x_${tag}" \
    "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${log}"
  status=${PIPESTATUS[0]}

  elapsed=$(( $(date +%s) - started ))
  human="$((elapsed / 3600))h$(( (elapsed % 3600) / 60 ))m"
  if [[ ${status} -eq 0 ]]; then
    echo "### [$(date '+%H:%M:%S')] ${tag} OK (${human})"
    summary+=("${tag}: OK    (${human})  ${log}")
  else
    echo "### [$(date '+%H:%M:%S')] ${tag} FAILED exit ${status} (${human}) -- continuing with the rest."
    summary+=("${tag}: FAILED exit ${status} (${human})  ${log}")
  fi
done

echo
echo "=================================================================================="
echo "robocasa_x queue finished ($(date '+%Y-%m-%d %H:%M:%S'))"
for line in "${summary[@]}"; do
  echo "  ${line}"
done
echo "  free disk: $(df -h --output=avail "${SCRIPT_DIR}" | tail -1 | tr -d ' ')"
echo "=================================================================================="
