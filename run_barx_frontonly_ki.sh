#!/usr/bin/env bash

# barx front-camera-only with KNOWLEDGE INSULATION.
#
# KI_OBJECTIVE picks what the VLM is trained to predict:
#   fast (default) -- the FAST-tokenized action chunk, pi_0.5 style, ~172 ids per chunk
#   lap            -- an English sentence describing the chunk, LAP style, ~16 tokens
#
# Two halves, and neither works alone:
#   * the flow-matching gradient is stopped before it reaches the VLM, so action learning can no
#     longer overwrite the pretrained representation, and
#   * the VLM is trained instead to predict the action chunk through its own LM head, so it still
#     learns the task rather than sitting frozen.
#
# Only the first half would reproduce the frozen-encoder baseline (action loss 0.092) under a
# different name; `--policy.ki_token_loss_weight` is validated to be > 0 for that reason.
#
# Cost over the baseline: with fast the postfix adds ~172 tokens to the VLM sequence and eager
# attention squares that, so batch 48 already peaks near 79 GiB per rank on 8-way DDP (see
# CLAUDE.md section 8). lap adds only ~16 tokens and is close to the baseline.
#
# Verify the mechanism before spending GPU-days on it:
#     python tools/smoke_test_knowledge_insulation.py

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BARX="${BARX:-${SCRIPT_DIR}/dataset_git/barx_panda_ur5e_iiwa}"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/dataset_git/barx_frontonly_p900_i1000_u1000}"
FRONT_CAM="observation.images.robot0_agentview_right"
KI_WEIGHT="${KI_WEIGHT:-1.0}"
KI_OBJECTIVE="${KI_OBJECTIVE:-fast}"
LOG="${LOG:-${SCRIPT_DIR}/outputs/logs/barx_frontonly_ki_${KI_OBJECTIVE}_w${KI_WEIGHT}_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

echo "barx front-only + knowledge insulation, objective=${KI_OBJECTIVE} (CE weight ${KI_WEIGHT}) -> ${LOG}"

EXTRA_TRAIN_ARGS=(
  --policy.knowledge_insulation=true
  --policy.ki_objective="${KI_OBJECTIVE:-fast}"
  --policy.ki_token_loss_weight="${KI_WEIGHT}"
  --policy.ki_max_tokens="${KI_MAX_TOKENS:-256}"
)
# A bash array cannot cross a process boundary; the child reads this back with mapfile.
EXTRA_TRAIN_ARGS_STR="$(printf '%s\n' "${EXTRA_TRAIN_ARGS[@]}")"
export EXTRA_TRAIN_ARGS_STR

PANDA_TOTAL_EPISODES="${PANDA_TOTAL_EPISODES:-900}" \
IIWA_EPISODES="${IIWA_EPISODES:-1000}" \
UR5E_EPISODES="${UR5E_EPISODES:-1000}" \
USE_PANDA_HUMAN="${USE_PANDA_HUMAN:-false}" \
NORMALIZE_TASK_LANGUAGE="${NORMALIZE_TASK_LANGUAGE:-true}" \
SOURCE_PANDA_MG="${SOURCE_PANDA_MG:-${BARX}/PandaOmron/pretrain/TurnOnSinkFaucet/lerobot}" \
SOURCE_IIWA="${SOURCE_IIWA:-${BARX}/IIWAOmron/pretrain/PnPCounterToSink/lerobot}" \
SOURCE_UR5E="${SOURCE_UR5E:-${BARX}/UR5eOmron/pretrain/PnPSinkToCounter/lerobot}" \
DATASET_ROOT="${DATASET_ROOT}" \
CAMERAS="${CAMERAS:-${FRONT_CAM}}" \
USE_WRIST_CAM="${USE_WRIST_CAM:-false}" \
BATCH_SIZE="${BATCH_SIZE:-48}" \
NUM_WORKERS="${NUM_WORKERS:-12}" \
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}" \
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}" \
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29530}" \
STEPS="${STEPS:-50000}" \
SAVE_FREQ="${SAVE_FREQ:-10000}" \
JOB_TAG="${JOB_TAG:-barx_frontonly_ki_${KI_OBJECTIVE}_w${KI_WEIGHT}}" \
TRAIN_SCRIPT=src/lerobot/scripts/lerobot_train_with_visual_robust.py \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

if [[ ${status} -eq 0 ]]; then
  echo "barx front-only knowledge-insulation (${KI_OBJECTIVE}) run finished OK -- log: ${LOG}"
else
  echo "barx front-only knowledge-insulation (${KI_OBJECTIVE}) run FAILED exit ${status} -- see ${LOG}"
fi
exit "${status}"
