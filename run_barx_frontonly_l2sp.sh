#!/usr/bin/env bash

# barx front-camera-only policy, vision tower TRAINED but pulled back toward its pretrained weights
# with an L2-SP penalty:  loss = policy_loss + VISION_L2SP_WEIGHT * sum_i (w_i - w0_i)^2
#
# Why this and not the two runs it sits between (measured on this project, 50k steps each):
#
#   run                       action loss   ||w-w0||^2   relative drift   centered pos/neg gap
#   pretrained init                     -            0          0.0000               -0.050
#   frozen vision encoder           0.092            0          0.0000                    ?
#   baseline (tower trained)        0.069         2526          0.1029               -0.138
#
# Training the tower buys a 25% lower action loss but drags the representation the WRONG way -- the
# gap goes from -0.050 at initialisation to -0.138, i.e. policy training teaches the encoder to tell
# the robots apart rather than to read the scene. Freezing the tower keeps the gap but costs that
# 25%. L2-SP is the dial between the two: the tower still adapts, but every step pays for how far it
# has moved from the pretrained point.
#
# Choosing VISION_L2SP_WEIGHT: the penalty is a SUM, so its printed value is large (weight 1e-3 at
# the baseline's end-state drift would read 2.5, next to an action loss of 0.069) while its gradient,
# 2*weight*(w - w0), stays modest and parameter-count independent. Judge it by the logged
# `vision_l2sp_relative_drift` against the 0.1029 above, not by the loss number.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BARX="${BARX:-${SCRIPT_DIR}/dataset_git/barx_panda_ur5e_iiwa}"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/dataset_git/barx_frontonly_p900_i1000_u1000}"
FRONT_CAM="observation.images.robot0_agentview_right"
WEIGHT="${VISION_L2SP_WEIGHT:?set VISION_L2SP_WEIGHT}"
SCOPE="${VISION_L2SP_SCOPE:-vision}"
LOG="${LOG:-${SCRIPT_DIR}/outputs/logs/barx_frontonly_l2sp_w${WEIGHT}_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

echo "barx front-only + L2-SP (${SCOPE}, weight ${WEIGHT}) -> ${LOG}"

PANDA_TOTAL_EPISODES="${PANDA_TOTAL_EPISODES:-900}" \
IIWA_EPISODES="${IIWA_EPISODES:-1000}" \
UR5E_EPISODES="${UR5E_EPISODES:-1000}" \
USE_PANDA_HUMAN="${USE_PANDA_HUMAN:-false}" \
NORMALIZE_TASK_LANGUAGE="${NORMALIZE_TASK_LANGUAGE:-true}" \
SOURCE_PANDA_MG="${SOURCE_PANDA_MG:-${BARX}/PandaOmron/pretrain/TurnOnSinkFaucet/lerobot}" \
SOURCE_IIWA="${SOURCE_IIWA:-${BARX}/IIWAOmron/pretrain/PnPCounterToSink/lerobot}" \
SOURCE_UR5E="${SOURCE_UR5E:-${BARX}/UR5eOmron/pretrain/PnPSinkToCounter/lerobot}" \
DATASET_ROOT="${DATASET_ROOT}" \
CAMERAS="${FRONT_CAM}" \
USE_WRIST_CAM="${USE_WRIST_CAM:-false}" \
BATCH_SIZE="${BATCH_SIZE:-64}" \
NUM_WORKERS="${NUM_WORKERS:-12}" \
STEPS="${STEPS:-50000}" \
SAVE_FREQ="${SAVE_FREQ:-10000}" \
WANDB_MODE="${WANDB_MODE:-online}" \
FREEZE_VISION_ENCODER=false \
VISION_L2SP_WEIGHT="${WEIGHT}" \
VISION_L2SP_SCOPE="${SCOPE}" \
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}" \
JOB_TAG="${JOB_TAG:-barx_frontonly_l2sp${SCOPE}_w${WEIGHT}}" \
TRAIN_SCRIPT=src/lerobot/scripts/lerobot_train_with_visual_robust.py \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

if [[ ${status} -eq 0 ]]; then
  echo "L2-SP w=${WEIGHT} finished OK -- log: ${LOG}"
else
  echo "L2-SP w=${WEIGHT} FAILED exit ${status} -- see ${LOG}"
fi
exit "${status}"
