#!/usr/bin/env bash

# Pure baseline on the single-task corpus: all three embodiments doing PnPSinkToCounter.
#
# Different from every earlier barx run, where each robot did a DIFFERENT task (Panda
# TurnOnSinkFaucet, IIWA PnPCounterToSink, UR5e PnPSinkToCounter). Here task is held fixed and only
# the embodiment varies, so cross-embodiment transfer is no longer confounded with task transfer.
#
# Pure means pure: stock SigLIP tower (no --policy.vision_encoder_path), tower trainable, no
# auxiliary loss, no knowledge insulation.
#
# Waits for the dataset download to finish first -- HF rate-limits at 1000 API requests per five
# minutes, so the fetch runs in windows and can still be in flight when this is launched.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BARX="${SCRIPT_DIR}/dataset_git/barx_panda_ur5e_iiwa"
WANT_IIWA=6008
WANT_PANDA=5408

count () { find "$1" -type f 2>/dev/null | wc -l; }

echo "waiting for the PnPSinkToCounter download to complete ..."
while true; do
  i=$(count "${BARX}/IIWAOmron/pretrain/PnPSinkToCounter")
  p=$(count "${BARX}/PandaOmron/pretrain/PnPSinkToCounter")
  [[ ${i} -ge ${WANT_IIWA} && ${p} -ge ${WANT_PANDA} ]] && break
  echo "  [$(date +%H:%M:%S)] IIWA ${i}/${WANT_IIWA}, Panda ${p}/${WANT_PANDA}"
  sleep 180
done
echo "download complete: IIWA ${i}, Panda ${p}"

for tree in "${BARX}/IIWAOmron/pretrain/PnPSinkToCounter/lerobot" \
            "${BARX}/PandaOmron/pretrain/PnPSinkToCounter/lerobot" \
            "${BARX}/UR5eOmron/pretrain/PnPSinkToCounter/lerobot"; do
  if [[ ! -f "${tree}/meta/info.json" ]]; then
    echo "missing ${tree}/meta/info.json -- the export is incomplete, not starting."
    exit 1
  fi
done

LOG="${SCRIPT_DIR}/outputs/logs/pnpsink_baseline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${SCRIPT_DIR}/outputs/logs"
echo "starting the baseline -> ${LOG}"

PANDA_TOTAL_EPISODES="${PANDA_TOTAL_EPISODES:-900}" \
IIWA_EPISODES="${IIWA_EPISODES:-1000}" \
UR5E_EPISODES="${UR5E_EPISODES:-1000}" \
USE_PANDA_HUMAN=false NORMALIZE_TASK_LANGUAGE=true \
SOURCE_PANDA_MG="${BARX}/PandaOmron/pretrain/PnPSinkToCounter/lerobot" \
SOURCE_IIWA="${BARX}/IIWAOmron/pretrain/PnPSinkToCounter/lerobot" \
SOURCE_UR5E="${BARX}/UR5eOmron/pretrain/PnPSinkToCounter/lerobot" \
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/dataset_git/barx_pnpsink_p900_i1000_u1000}" \
CAMERAS=observation.images.robot0_agentview_right USE_WRIST_CAM=false \
FREEZE_VISION_ENCODER=false \
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}" BATCH_SIZE="${BATCH_SIZE:-64}" \
NUM_WORKERS="${NUM_WORKERS:-12}" \
STEPS="${STEPS:-50000}" SAVE_FREQ="${SAVE_FREQ:-10000}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29550}" \
JOB_TAG=pnpsink_baseline \
TRAIN_SCRIPT=src/lerobot/scripts/lerobot_train_with_visual_robust.py \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${LOG}"
exit "${PIPESTATUS[0]}"
