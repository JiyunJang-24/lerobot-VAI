#!/usr/bin/env bash

# barx front-camera-only baseline with the SigLIP vision tower FROZEN.
#
# Why: every barx front-only run so far fine-tunes the vision tower, and the SigLIP feature analysis
# showed that fine-tuning alone drives the positive/negative gap NEGATIVE (-0.138 centered, vs -0.050
# at initialisation) -- i.e. policy training pushes the encoder to separate embodiments, the opposite
# of what the visual-robust loss is trying to buy. This run removes that variable: identical data,
# identical batch, identical schedule, only `--policy.freeze_vision_encoder=true`.
#
# It is deliberately cheap enough to share GPUs with a live run. Measured peak per rank
# (tools/probe_frozen_encoder_memory.py, batch 64, front camera only):
#
#     freeze=true    9.2 GiB      freeze=false   ~32 GiB (OOMed in a 34 GiB budget)
#
# Freezing removes the tower's gradients, its Adam moments, and -- because no leaf under it requires
# grad any more -- its entire backward activation graph, which is where most of the saving is.
#
# Two things a parallel launch must not inherit from the normal path:
#   * run_visual_robust_with_oom_backoff.sh waits for every lerobot_train process to exit before it
#     starts, so it would block forever behind the run we are sharing GPUs with. Hence this script
#     calls train_smolVLA_robocasa_x.sh directly.
#   * accelerate's rendezvous port defaults to 29500, already bound by the other run -> MAIN_PROCESS_PORT.
#
# Dataloader workers are halved (12 -> 6/rank) because the co-tenant run is already using 8x12 of
# this box's 96 cores.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BARX="${BARX:-${SCRIPT_DIR}/dataset_git/barx_panda_ur5e_iiwa}"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/dataset_git/barx_frontonly_p900_i1000_u1000}"
FRONT_CAM="observation.images.robot0_agentview_right"
LOG="${SCRIPT_DIR}/outputs/logs/barx_frontonly_frozenvis_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

echo "barx front-only, vision encoder FROZEN -> ${LOG}"

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
NUM_WORKERS="${NUM_WORKERS:-6}" \
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}" \
FREEZE_VISION_ENCODER=true \
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29520}" \
JOB_TAG="${JOB_TAG:-barx_frontonly_frozenvis_baseline}" \
TRAIN_SCRIPT=src/lerobot/scripts/lerobot_train_with_visual_robust.py \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

if [[ ${status} -eq 0 ]]; then
  echo "barx front-only frozen-encoder baseline finished OK -- log: ${LOG}"
else
  echo "barx front-only frozen-encoder baseline FAILED exit ${status} -- see ${LOG}"
fi
exit "${status}"
