#!/usr/bin/env bash
# barx3: baseline vs visual-encoder contrastive regularisation, side by side.
#
#   baseline     GPUs 0-3, port 29510
#   contrastive  GPUs 4-7, port 29520
#
# BOTH runs use the SAME policy batch (32/GPU, effective 128). That is deliberate: sizing each run
# to its own memory ceiling would confound batch size with method. 32 is the ceiling for the
# contrastive run -- measured, 48 OOMs -- so the baseline is deliberately run below what it could
# fit alone.
#
# VISUAL_ROBUST_BATCH_SIZE=32 is also the measured ceiling (40 and 48 both OOM at 72.3 GB -> 81 GB).
# It gives a contrastive matrix of 32 frames x 3 embodiments = 96 samples, 4x the repo default of 8.
set -uo pipefail
cd "$(dirname "$0")"
source /home/gpuuser/miniforge3/etc/profile.d/conda.sh
conda activate smolvla
source ./run_barx3_front_common.sh

STEPS="${STEPS:-50000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
BATCH="${BATCH:-32}"
VB="${VB:-32}"
mkdir -p outputs/logs

GPU_IDS=4,5 MAIN_PROCESS_PORT=29530 BATCH_SIZE="$BATCH" \
STEPS="$STEPS" SAVE_FREQ="$SAVE_FREQ" NUM_WORKERS=8 \
JOB_TAG=barx3_front_baseline OUTPUT_DIR=outputs/barx3_front/baseline \
  ./train_smolVLA_robocasa_x.sh > outputs/logs/barx3_front_baseline.log 2>&1 &
BASE_PID=$!
echo "baseline pid $BASE_PID -> outputs/barx3_front/baseline"

sleep 90   # stagger: both runs rebuild the same dataset cache at startup

GPU_IDS=6,7 MAIN_PROCESS_PORT=29540 BATCH_SIZE="$BATCH" \
STEPS="$STEPS" SAVE_FREQ="$SAVE_FREQ" NUM_WORKERS=8 \
VISUAL_ROBUST_BATCH_SIZE="$VB" \
VR_ROOT=dataset_git/visual_robust_new_barx/new_barx \
VR_REPO_IDS="IIWAOmron_PnPCounterToSink/lerobot,PandaOmron_TurnOnSinkFaucet/lerobot,UR5eOmron_PnPSinkToCounter/lerobot" \
JOB_TAG_BASE=barx3_front OUTPUT_DIR=outputs/barx3_front/contrastive \
  ./train_smolVLA_robocasa_x_visual_robust.sh > outputs/logs/barx3_front_contrastive.log 2>&1 &
VR_PID=$!
echo "contrastive pid $VR_PID -> outputs/barx3_front/contrastive"

wait $BASE_PID; echo "BASELINE EXIT $?"
wait $VR_PID;   echo "CONTRASTIVE EXIT $?"
echo BARX3_FRONT_DONE
