#!/usr/bin/env bash

# Baseline on ChiefJang/barx_panda_ur5e_iiwa: front camera only (robot0_agentview_right), no wrist,
# no visual-robust loss.
#
# This export is one task tree per embodiment, and they are DIFFERENT tasks:
#   PandaOmron/pretrain/TurnOnSinkFaucet    900 eps  283,567 frames   1 instruction
#   UR5eOmron/pretrain/PnPSinkToCounter    1000 eps  498,810 frames   8 instructions
#   IIWAOmron/pretrain/PnPCounterToSink    1000 eps  393,709 frames   8 instructions
# All v2.1, state[16] / action[12], and only two cameras (agentview_right + eye_in_hand) -- there is
# no agentview_left here, so "front only" means dropping eye_in_hand at prep time.
#
# Dropping the wrist camera has to happen in the data, not via --dataset.use_wrist_cam=false: the
# trainer implements that flag as `[key for key in features if "wrist" in key]`, and this project's
# wrist camera is `observation.images.robot0_eye_in_hand`, which contains no "wrist" substring. The
# flag is still passed (it is semantically right and disables an assertion path keyed to the
# observation.wrist_image naming), but CAMERAS is what actually removes the camera.
#
# Prerequisite: the three trees must be codebase_version v3.0 -- ./setup_barx_dataset.sh does that.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BARX="${BARX:-${SCRIPT_DIR}/dataset_git/barx_panda_ur5e_iiwa}"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/dataset_git/barx_frontonly_p900_i1000_u1000}"
FRONT_CAM="observation.images.robot0_agentview_right"
BATCH_SIZE="${BATCH_SIZE:-64}"
LOG="${SCRIPT_DIR}/outputs/logs/barx_baseline_frontonly_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

echo "barx baseline, front camera only -> ${LOG}"
echo "  panda: TurnOnSinkFaucet (900)   ur5e: PnPSinkToCounter (1000)   iiwa: PnPCounterToSink (1000)"
echo "  camera: ${FRONT_CAM}"
echo "  batch:  ${BATCH_SIZE}/GPU"

# panda here is a single tree, so it goes in via the panda_mg slot and panda_human is skipped
# (--panda-human-episodes 0). PANDA_TOTAL_EPISODES is 900 because that is all this export has; the
# prep script clamps and says so rather than failing if it is asked for more.
PANDA_TOTAL_EPISODES=900 \
IIWA_EPISODES=1000 \
UR5E_EPISODES=1000 \
USE_PANDA_HUMAN=false \
NORMALIZE_TASK_LANGUAGE=true \
CAMERAS="${FRONT_CAM}" \
USE_WRIST_CAM=false \
BATCH_SIZE="${BATCH_SIZE}" \
JOB_TAG=barx_frontonly_baseline \
SOURCE_PANDA_MG="${BARX}/PandaOmron/pretrain/TurnOnSinkFaucet/lerobot" \
SOURCE_IIWA="${BARX}/IIWAOmron/pretrain/PnPCounterToSink/lerobot" \
SOURCE_UR5E="${BARX}/UR5eOmron/pretrain/PnPSinkToCounter/lerobot" \
DATASET_ROOT="${DATASET_ROOT}" \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

if [[ ${status} -eq 0 ]]; then
  echo "barx baseline finished OK -- log: ${LOG}"
else
  echo "barx baseline FAILED exit ${status} -- see ${LOG}"
fi
exit "${status}"
