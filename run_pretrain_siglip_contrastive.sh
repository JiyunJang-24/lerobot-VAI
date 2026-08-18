#!/usr/bin/env bash

# Pre-train the SigLIP tower for embodiment invariance on the visual-robust export, then hand the
# result to SmolVLA.
#
#   ./run_pretrain_siglip_contrastive.sh                 # pooled contrastive, 5k steps
#   POOL=tokens ./run_pretrain_siglip_contrastive.sh     # per-patch contrastive
#   STEPS=20000 LR=3e-5 ./run_pretrain_siglip_contrastive.sh
#
# Then train a policy on top of it:
#
#   EXTRA_TRAIN_ARGS_STR=$'--policy.vision_encoder_path=outputs/siglip_pretrain/contrastive_mean_b16/vision_tower.safetensors' \
#     ./run_barx_frontonly_ki.sh          # or any other preset
#
# Read before trusting the number it prints: the pooled gap and the pooled contrastive objective
# are the same statistic, so `pool=mean` optimises exactly what it reports. The token gap is the
# honest one -- if it does not move, the policy will not see the invariance, which is precisely how
# the alignment objective failed (CLAUDE.md section 3).

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
POOL="${POOL:-mean}"
STEPS="${STEPS:-5000}"
BATCH="${BATCH:-16}"
LR="${LR:-1e-5}"
TAG="${TAG:-contrastive_${POOL}_b${BATCH}}"
OUT="${OUT:-${SCRIPT_DIR}/outputs/siglip_pretrain/${TAG}}"
LOG="${LOG:-${SCRIPT_DIR}/outputs/logs/siglip_pretrain_${TAG}_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

export LD_LIBRARY_PATH="${SCRIPT_DIR}/.ffmpeg_shim:${LD_LIBRARY_PATH:-}"

echo "SigLIP contrastive pre-training (pool=${POOL}, ${STEPS} steps) -> ${LOG}"

python "${SCRIPT_DIR}/src/lerobot/scripts/pretrain_siglip_visual_robust.py" \
  --steps "${STEPS}" \
  --batch-size "${BATCH}" \
  --lr "${LR}" \
  --pool "${POOL}" \
  --temperature "${TEMPERATURE:-0.1}" \
  --l2sp "${L2SP:-0.0}" \
  --eval-every "${EVAL_EVERY:-250}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --output-dir "${OUT}" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

if [[ ${status} -eq 0 ]]; then
  echo "done -- tower at ${OUT}/vision_tower.safetensors"
else
  echo "FAILED exit ${status} -- see ${LOG}"
fi
exit "${status}"
