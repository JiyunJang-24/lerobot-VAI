#!/usr/bin/env bash

# Waits for BOTH GPU 0-3 and GPU 4-7 fusion runs (one tower frozen each way) to finish, then starts
# the "both towers trainable" fusion condition across all 8 GPUs.
#
#   GPU 0-3 done   main=stock (trainable),      aux=contrastive+EEF 6-view (frozen)
#   GPU 4-7 done   main=contrastive+EEF 6-view (trainable), aux=stock (frozen)
#         v
#   GPU 0-7        main=stock, aux=contrastive+EEF 6-view, BOTH trainable
#                  batch 48/rank x 8 = 384 effective, matching the two runs above (96 x 4)

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
TAG_A="${TAG_A:-pnpsink_fusion_aux_contrastive_frozen_ki_lap}"
TAG_B="${TAG_B:-pnpsink_fusion_aux_stock_frozen_ki_lap}"
LOG_A="${LOG_A:-${SCRIPT_DIR}/outputs/logs/pnpsink_dualenc_ki_lap_20260824_205615.log}"
LOG_B="${LOG_B:-${SCRIPT_DIR}/outputs/logs/pnpsink_dualenc_ki_lap_20260824_205620.log}"

running () {
  local tag="$1"
  for p in $(pgrep -f lerobot_train_with_visual_robust 2>/dev/null); do
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "JOB_TAG=${tag}" && return 0
  done
  return 1
}

echo "waiting for ${TAG_A} (GPU 0-3) and ${TAG_B} (GPU 4-7) to finish ..."
while running "${TAG_A}" || running "${TAG_B}"; do sleep 120; done

for log in "${LOG_A}" "${LOG_B}"; do
  if [[ ! -f "${log}" ]] || ! grep -aq "End of training" "${log}"; then
    echo "did not find 'End of training' in ${log}; not starting the 8-GPU both-trainable run."
    exit 1
  fi
done
echo "both runs finished cleanly; starting both-trainable fusion on all 8 GPUs."

WAIT=false GPU_IDS=0,1,2,3,4,5,6,7 BATCH_SIZE="${BATCH_SIZE:-48}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29596}" \
FREEZE_AUX=false \
AUX_TOWER="${AUX_TOWER:-${SCRIPT_DIR}/outputs/siglip_pretrain/both_cmean_eefattn_6views_b16/vision_tower.safetensors}" \
JOB_TAG="${JOB_TAG:-pnpsink_fusion_both_trainable_ki_lap}" \
  "${SCRIPT_DIR}/run_pnpsink_dualencoder_ki_lap.sh"
exit $?
