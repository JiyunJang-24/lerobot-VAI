#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="${SCRIPT_DIR}/.."
VENV_PATH="${REPO_ROOT}/venv-lerobot-eval"

if [ ! -d "${VENV_PATH}" ]; then
  echo "Error: ${VENV_PATH} virtual environment folder not found."
  exit 1
fi

source ${VENV_PATH}/bin/activate

export PYTHONPATH="${REPO_ROOT}/third_party/LIBERO:${PYTHONPATH:-}"

# Do avoid EGL device display error
export MUJOCO_GL=osmesa

RENAME_MAP='{"observation.images.image": "observation.image", "observation.images.image2": "observation.wrist_image"}'
DEBUG_ROOT="${REPO_ROOT}/outputs/eval/viewpoint_debug"

POLICY_PATHS=(
  # "/data1/local/lerobot-VAI/outputs/train/2026-06-08/14-32-37_smolvla_vanilla_rma_02/checkpoints/050000/pretrained_model"
  # "/data1/local/lerobot-VAI/outputs/train/2026-06-08/14-32-44_smolvla_axisguide_rma_02/checkpoints/050000/pretrained_model"
  # "/data1/local/lerobot-VAI/outputs/train/2026-06-13/12-55-38_smolvla_kyc_rma_02_ex2/checkpoints/050000/pretrained_model"
  "/data1/local/lerobot-VAI/outputs/train/2026-06-15/12-24-55_smolvla_kyc_rma_02_ex2_plucker_encoder/checkpoints/045000/pretrained_model"

)

POLICY_TAGS=(
  # "RMA_ex02_smolvla_vanilla_50000"
  # "RMA_ex02_smolvla_axisguide_50000"
  # "RMA_ex02_02_smolvla_kyc_50000"
  "RMA_ex02_02_smolvla_kyc_encoder_45000"
)

# Format: suite task_id viewpoint_angle label
# EVAL_CASES=(
#   "libero_spatial 0 0 spatial0_angle0"
#   "libero_goal 0 15 goal0_angle15"
#   "libero_object 0 30 object0_angle30"
#   "libero_10 0 330 libero10_0_angle330"
#   "libero_spatial 2 345 spatial2_angle345"
# )

EVAL_CASES=(
  # "libero_10 0 330 libero10_0_angle330"
  # "libero_spatial 2 345 spatial2_angle345"
  "libero_object 0 355 object0_angle355"
  "libero_object 0 350 object0_angle350"
  "libero_object 0 345 object0_angle345"
  "libero_object 0 340 object0_angle340"
  "libero_object 0 335 object0_angle335"
  "libero_object 0 330 object0_angle330"
  "libero_object 0 325 object0_angle325"
  # "libero_10 1 330 libero10_1_angle330"
)

run_eval() {
  local policy_path="$1"
  local policy_tag="$2"
  local suite="$3"
  local task_id="$4"
  local angle="$5"
  local label="$6"

  python3 "${REPO_ROOT}/src/lerobot/scripts/lerobot_eval.py" \
    --policy.path="${policy_path}" \
    --env.type=libero \
    --env.task="${suite}" \
    --env.task_ids="${task_id}" \
    --env.viewpoint_rotate="${angle}" \
    --env.viewpoint_debug=true \
    --env.viewpoint_debug_dir="${DEBUG_ROOT}/${policy_tag}/${label}" \
    --eval.n_episodes=20 \
    --eval.batch_size=1 \
    --job_name="${policy_tag}_${label}" \
    --rename_map="${RENAME_MAP}"
}

for case in "${EVAL_CASES[@]}"; do
  read -r suite task_id angle label <<< "${case}"
  for idx in "${!POLICY_PATHS[@]}"; do
    run_eval "${POLICY_PATHS[$idx]}" "${POLICY_TAGS[$idx]}" "${suite}" "${task_id}" "${angle}" "${label}"
  done
done
