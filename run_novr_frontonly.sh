#!/usr/bin/env bash

# Baseline for the currently running front-only visual-robust run: SAME data, SAME schedule, but the
# visual-robust auxiliary loss removed entirely.
#
# "Removed entirely" here means the plain trainer (src/lerobot/scripts/lerobot_train.py) with no
# --dataset.visual_robust_* flags at all -- not the visual-robust trainer with
# contrastive_weight=0. The latter would still build the auxiliary dataloader and still push the
# extra views through the vision encoder every step, so it would differ from this run in memory,
# step time and dataloader RNG consumption for no benefit.
#
# Everything else matches run_vr_background_frontonly.sh so the two runs are directly comparable:
#   - same DATASET_ROOT (the front-only prep: robot0_agentview_right only, wrist removed at prep
#     time via CAMERAS -- see run_vr_background_frontonly.sh for why the use_wrist_cam flag alone
#     does not drop this project's `robot0_eye_in_hand` camera)
#   - same sources, 1000/1000/1000 episodes, panda from mg only, normalised task language
#   - same batch size 64/GPU, 50k steps, save_freq 10k, bf16, 8 GPUs
#
# The prep step is a no-op here: DATASET_ROOT was already built by the visual-robust front-only run
# and prepare_robocasa_x_dataset.py skips a dest that already has the requested episode count.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BASE="${SCRIPT_DIR}/dataset_git/robocasa_x_atmoic"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/dataset_git/robocasa_x_pnp_mgonly_frontonly_p1000_i1000_u1000}"
FRONT_CAM="observation.images.robot0_agentview_right"
WRIST_CAM="observation.images.robot0_eye_in_hand"
LOG="${SCRIPT_DIR}/outputs/logs/robocasa_x_pnp_novr_frontonly_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

echo "Training PnP baseline (NO visual robust loss), FRONT CAMERA ONLY -> ${LOG}"
echo "  policy cameras: ${FRONT_CAM}"
echo "  dataset root:   ${DATASET_ROOT}"

PANDA_TOTAL_EPISODES=1000 \
IIWA_EPISODES=1000 \
UR5E_EPISODES=1000 \
USE_PANDA_HUMAN=false \
NORMALIZE_TASK_LANGUAGE=true \
JOB_TAG="multi_task_pnp_mgonly_langnorm_novr_frontonly" \
SOURCE_PANDA_MG="${BASE}/mg/PandaOmron/pretrain/PnPCounterToStove" \
SOURCE_IIWA="${BASE}/IIWAOmron/pretrain/PnPCounterToSink/lerobot" \
SOURCE_UR5E="${BASE}/UR5eOmron/pretrain/PnPSinkToCounter/lerobot" \
DATASET_ROOT="${DATASET_ROOT}" \
CAMERAS="${FRONT_CAM}" \
USE_WRIST_CAM=false \
BATCH_SIZE="${BATCH_SIZE:-64}" \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

# Same check the visual-robust front-only run does: prove from the saved config that the wrist
# camera never reached the policy, rather than trusting the flag.
ckpt="$(ls -td "${SCRIPT_DIR}"/outputs/train/*/*novr_frontonly*/checkpoints/*/pretrained_model 2>/dev/null | head -1)"
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
  echo "no-VR front-only run finished OK -- log: ${LOG}"
else
  echo "no-VR front-only run FAILED exit ${status} -- see ${LOG}"
fi
exit "${status}"
