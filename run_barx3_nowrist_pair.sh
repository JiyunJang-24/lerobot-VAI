#!/usr/bin/env bash
# SUPERSEDED -- THIS SCRIPT DOES NOT REMOVE THE WRIST CAMERA. Kept only because it produced the
# runs now stored under outputs/barx3_batch36_BOTHCAMS/.
#
# It sets --dataset.use_wrist_cam=false, which in this repo drops the key `observation.wrist_image`
# (lerobot_dataset.py) and keys whose name contains "wrist" (lerobot_train.py). The barx key is
# `observation.images.robot0_eye_in_hand` and matches neither, so both runs trained on front+wrist
# exactly like run_barx3_pair.sh -- only the batch differed (36 vs 32).
#
# The measurement that should have caught it at the time: bs=32/vb=32 peaked at 72,251 MiB "without
# wrist" against 72,275 MiB with it. A 24 MiB difference means nothing was removed.
#
# For a genuinely front-only run use run_barx3_front_pair.sh, which builds a dataset with
# CAMERAS="observation.images.robot0_agentview_right" -- the camera set is decided when the dataset
# is built, not by a runtime flag.

set -uo pipefail
cd "$(dirname "$0")"
source /home/gpuuser/miniforge3/etc/profile.d/conda.sh
conda activate smolvla
source ./run_barx3_common.sh
# The wrist-free arm of the 2x2. Everything else is held identical -- same prepared dataset (it
# still CONTAINS the wrist videos, the policy simply does not read them), same policy batch, same
# auxiliary batch -- so the only thing that differs from run_barx3_pair.sh is this flag.
#
# Note the auxiliary contrastive term was ALREADY front-only in both arms
# (VISUAL_ROBUST_WRIST_ALIGNMENT_WEIGHT=0), so what changes here is the POLICY's observation
# space, not what the regulariser acts on.
export USE_WRIST_CAM=false

STEPS="${STEPS:-50000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
BATCH="${BATCH:-36}"
VB="${VB:-32}"
mkdir -p outputs/logs

GPU_IDS=0,1,2,3 MAIN_PROCESS_PORT=29530 BATCH_SIZE="$BATCH" \
STEPS="$STEPS" SAVE_FREQ="$SAVE_FREQ" NUM_WORKERS=8 \
JOB_TAG=barx3_nowrist_baseline OUTPUT_DIR=outputs/barx3_nowrist/baseline \
  ./train_smolVLA_robocasa_x.sh > outputs/logs/barx3_nowrist_baseline.log 2>&1 &
BASE_PID=$!
echo "baseline pid $BASE_PID -> outputs/barx3_nowrist/baseline"

sleep 90   # stagger: both runs rebuild the same dataset cache at startup

GPU_IDS=4,5,6,7 MAIN_PROCESS_PORT=29540 BATCH_SIZE="$BATCH" \
STEPS="$STEPS" SAVE_FREQ="$SAVE_FREQ" NUM_WORKERS=8 \
VISUAL_ROBUST_BATCH_SIZE="$VB" \
VR_ROOT=dataset_git/visual_robust_new_barx/new_barx \
VR_REPO_IDS="IIWAOmron_PnPCounterToSink/lerobot,PandaOmron_TurnOnSinkFaucet/lerobot,UR5eOmron_PnPSinkToCounter/lerobot" \
JOB_TAG_BASE=barx3_nowrist OUTPUT_DIR=outputs/barx3_nowrist/contrastive \
  ./train_smolVLA_robocasa_x_visual_robust.sh > outputs/logs/barx3_nowrist_contrastive.log 2>&1 &
VR_PID=$!
echo "contrastive pid $VR_PID -> outputs/barx3_nowrist/contrastive"

wait $BASE_PID; echo "BASELINE EXIT $?"
wait $VR_PID;   echo "CONTRASTIVE EXIT $?"
echo BARX3_NOWRIST_DONE
