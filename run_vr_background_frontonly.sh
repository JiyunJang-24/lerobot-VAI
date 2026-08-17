#!/usr/bin/env bash

# Same run as run_vr_background_training.sh, but the POLICY is fed only the front camera
# (robot0_agentview_right) -- no wrist camera.
#
# Why this is not simply --dataset.use_wrist_cam=false:
#   lerobot_train_with_visual_robust.py implements that flag as
#       wrist_feature_keys = [key for key in ds_meta.features if "wrist" in key]
#   i.e. it drops features whose *name contains the substring "wrist"*. This project's wrist camera
#   is named `observation.images.robot0_eye_in_hand`, which contains no such substring, so the flag
#   matches nothing and the policy keeps receiving both cameras. Verified against the prepared
#   dataset: the filter removes 0 keys. Setting it alone would produce a run that looks like
#   "front only" and is not.
#
# So the camera is removed where it actually takes effect -- at data prep, via CAMERAS, which
# tools/prepare_robocasa_x_dataset.py turns into a remove_feature() on every subset. That needs its
# own DATASET_ROOT because the existing one is built with two cameras.
#
# use_wrist_cam=false is still passed: it is semantically right, and it disables the trainer's
# POLICY_CAMERA_KEYS assertion path that only applies to the observation.wrist_image naming.
#
# The visual-robust auxiliary loss is unchanged (front contrastive only, wrist alignment weight 0).

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BASE="${SCRIPT_DIR}/dataset_git/robocasa_x_atmoic"
VR_ROOT="${VR_ROOT:-${SCRIPT_DIR}/dataset_git/visual_robust_robocasa_x_bg}"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/dataset_git/robocasa_x_pnp_mgonly_frontonly_p1000_i1000_u1000}"
FRONT_CAM="observation.images.robot0_agentview_right"
WRIST_CAM="observation.images.robot0_eye_in_hand"
LOG="${SCRIPT_DIR}/outputs/logs/robocasa_x_pnp_vr_frontonly_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

echo "Training PnP + visual robust, FRONT CAMERA ONLY -> ${LOG}"
echo "  policy cameras: ${FRONT_CAM}"
echo "  dataset root:   ${DATASET_ROOT}"

CAMERAS="${FRONT_CAM}" \
USE_WRIST_CAM=false \
VR_ROOT="${VR_ROOT}" \
JOB_TAG_SUFFIX=frontonly \
DATASET_ROOT="${DATASET_ROOT}" \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x_visual_robust.sh" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

# Prove the wrist camera really did not reach the policy, rather than assuming the flag worked.
ckpt="$(ls -td "${SCRIPT_DIR}"/outputs/train/*/*frontonly*/checkpoints/*/pretrained_model 2>/dev/null | head -1)"
if [[ -n "${ckpt}" ]]; then
  echo
  echo "=== policy input features actually used ==="
  python - "${ckpt}/train_config.json" "${WRIST_CAM}" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
wrist = sys.argv[2]
feats = list(cfg["policy"]["input_features"])
for f in feats:
    print(f"   {f}")
if wrist in feats:
    print(f"WARNING: {wrist} IS present -- this run is NOT front-only")
else:
    print(f"OK: {wrist} absent -- front-only confirmed")
PY
fi

if [[ ${status} -eq 0 ]]; then
  echo "front-only run finished OK -- log: ${LOG}"
else
  echo "front-only run FAILED exit ${status} -- see ${LOG}"
fi
exit "${status}"
