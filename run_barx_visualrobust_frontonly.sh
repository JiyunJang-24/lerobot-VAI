#!/usr/bin/env bash

# barx front-camera-only policy + the visual-robust FRONT contrastive loss, using the `new_barx`
# export (ChiefJang/visual_robust_robocasa_x, new_barx/).
#
# The auxiliary export has one tree per barx embodiment/task pair, and each tree renders that
# episode from ALL THREE embodiments:
#   new_barx/PandaOmron_TurnOnSinkFaucet/lerobot   108 eps
#   new_barx/IIWAOmron_PnPCounterToSink/lerobot    108 eps
#   new_barx/UR5eOmron_PnPSinkToCounter/lerobot    108 eps
# with 6 cameras each = 3 embodiments x {agentview_right, eye_in_hand}. No agentview_left and no
# background variation, so after the reshape there are exactly 3 front views per frame -- one per
# embodiment -- which is the original single-group contrastive setup.
#
# Everything is the plain contrastive objective at its original settings (weight 0.5, temperature
# 0.1, vr_batch 8), not the alignment variant: on this project's data the alignment objective sat at
# cos ~0.999 from step 0 and contributed ~0.04% of the loss, whereas contrastive lands at a healthy
# 1.4-2.4 because dividing by the temperature rescales those near-identical similarities.
#
# The policy side is the same front-only barx setup whose baseline finished at loss 0.069, at the
# same batch 64, so the pair isolates the auxiliary loss.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BARX="${BARX:-${SCRIPT_DIR}/dataset_git/barx_panda_ur5e_iiwa}"
VR_ROOT="${VR_ROOT:-${SCRIPT_DIR}/dataset_git/visual_robust_new_barx/new_barx}"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/dataset_git/barx_frontonly_p900_i1000_u1000}"
FRONT_CAM="observation.images.robot0_agentview_right"
BATCH_SIZE="${BATCH_SIZE:-64}"
LOG="${SCRIPT_DIR}/outputs/logs/barx_vr_frontonly_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

VR_REPO_IDS="${VR_REPO_IDS:-PandaOmron_TurnOnSinkFaucet/lerobot,IIWAOmron_PnPCounterToSink/lerobot,UR5eOmron_PnPSinkToCounter/lerobot}"

echo "barx + visual robust (contrastive), front camera only -> ${LOG}"

PANDA_TOTAL_EPISODES="${PANDA_TOTAL_EPISODES:-900}" \
IIWA_EPISODES="${IIWA_EPISODES:-1000}" \
UR5E_EPISODES="${UR5E_EPISODES:-1000}" \
VR_ROOT="${VR_ROOT}" \
VR_REPO_IDS="${VR_REPO_IDS}" \
DATASET_ROOT="${DATASET_ROOT}" \
CAMERAS="${FRONT_CAM}" \
USE_WRIST_CAM="${USE_WRIST_CAM:-false}" \
BATCH_SIZE="${BATCH_SIZE}" \
VISUAL_ROBUST_BATCH_SIZE="${VISUAL_ROBUST_BATCH_SIZE:-8}" \
VISUAL_ROBUST_FRONT_OBJECTIVE="${VISUAL_ROBUST_FRONT_OBJECTIVE:-contrastive}" \
VISUAL_ROBUST_CONTRASTIVE_WEIGHT="${VISUAL_ROBUST_CONTRASTIVE_WEIGHT:-0.5}" \
VISUAL_ROBUST_HEAD_MODE="${VISUAL_ROBUST_HEAD_MODE:-none}" \
SOURCE_PANDA_MG="${BARX}/PandaOmron/pretrain/TurnOnSinkFaucet/lerobot" \
SOURCE_IIWA="${BARX}/IIWAOmron/pretrain/PnPCounterToSink/lerobot" \
SOURCE_UR5E="${BARX}/UR5eOmron/pretrain/PnPSinkToCounter/lerobot" \
JOB_TAG_BASE=barx_frontonly \
JOB_TAG_SUFFIX= \
  "${SCRIPT_DIR}/run_visual_robust_with_oom_backoff.sh" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

if [[ ${status} -eq 0 ]]; then
  echo "barx + visual robust finished OK -- log: ${LOG}"
else
  echo "barx + visual robust FAILED exit ${status} -- see ${LOG}"
fi
exit "${status}"
