#!/usr/bin/env bash

# Waits for the barx front-only baseline to finish, then runs the same baseline with the WRIST
# camera added (robot0_agentview_right + robot0_eye_in_hand).
#
# Everything else is held fixed so the pair isolates the wrist camera's effect: same 2900 episodes
# (panda 900 TurnOnSinkFaucet / iiwa 1000 PnPCounterToSink / ur5e 1000 PnPSinkToCounter), same task
# language normalisation, same batch 64/GPU, same plain lerobot_train.py with no visual-robust loss.
#
# The dataset was pre-built while the first run was still training, so there is no prep wait here --
# the barx export already contains exactly these two cameras, so building it was a straight copy
# (no episode split, no camera removal, no re-encode) and took ~20 s.
#
# Runs in `tmux new -s smolvla`.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BARX="${BARX:-${SCRIPT_DIR}/dataset_git/barx_panda_ur5e_iiwa}"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/dataset_git/barx_frontwrist_p900_i1000_u1000}"
FRONT_CAM="observation.images.robot0_agentview_right"
WRIST_CAM="observation.images.robot0_eye_in_hand"
BATCH_SIZE="${BATCH_SIZE:-64}"
TMUX_SESSION="${TMUX_SESSION:-smolvla}"
CONDA_ENV="${CONDA_ENV:-smolvla}"
POLL_S="${POLL_S:-60}"

CONDA_SH=""
for candidate in \
  "$(dirname "$(dirname "${CONDA_EXE:-/nonexistent}")")/etc/profile.d/conda.sh" \
  "/opt/conda/etc/profile.d/conda.sh" \
  "${HOME}/miniforge3/etc/profile.d/conda.sh" \
  "${HOME}/miniconda3/etc/profile.d/conda.sh" \
  "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
  [[ -f "${candidate}" ]] && { CONDA_SH="${candidate}"; break; }
done
[[ -n "${CONDA_SH}" ]] || { echo "could not locate conda.sh" >&2; exit 1; }
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}" || exit 1
fi
command -v accelerate >/dev/null 2>&1 || { echo "accelerate not on PATH" >&2; exit 1; }

echo "=================================================================================="
echo "queued: barx baseline WITH wrist camera  [$(date '+%Y-%m-%d %H:%M:%S')]"
echo "  conda env:    ${CONDA_DEFAULT_ENV:-unknown}"
echo "  dataset root: ${DATASET_ROOT}"
echo "  cameras:      ${FRONT_CAM} + ${WRIST_CAM}"
echo "  batch:        ${BATCH_SIZE}/GPU (same as the front-only run)"
echo "=================================================================================="

while pgrep -f "lerobot_train.*\.py" >/dev/null 2>&1 \
   || pgrep -f "run_visual_robust_with_oom_backoff.sh" >/dev/null 2>&1; do
  sleep "${POLL_S}"
done
echo "[$(date '+%H:%M:%S')] no training running any more"

while pgrep -f "prepare_robocasa_x_dataset.py" >/dev/null 2>&1; do
  echo "[$(date '+%H:%M:%S')] waiting for data prep"
  sleep 30
done

echo "[$(date '+%H:%M:%S')] verifying the prepared dataset carries BOTH cameras"
if ! python - "${DATASET_ROOT}/raw" "${FRONT_CAM}" "${WRIST_CAM}" <<'PY'
import json, sys
from pathlib import Path
raw, front, wrist = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
subsets = sorted(p.parent.parent for p in raw.glob("*/meta/info.json"))
if not subsets:
    print(f"  no subsets under {raw}")
    sys.exit(1)
ok = True
for root in subsets:
    info = json.loads((root / "meta" / "info.json").read_text())
    cams = sorted(k for k, v in info["features"].items() if v.get("dtype") == "video")
    good = cams == sorted([front, wrist])
    if not good:
        ok = False
    print(f"  {'OK ' if good else 'BAD'} {root.name:10s} eps={info['total_episodes']:5d} "
          f"cameras={[c.replace('observation.images.', '') for c in cams]}")
sys.exit(0 if ok else 1)
PY
then
  echo "verification FAILED -- not starting training." >&2
  exit 1
fi
echo "[$(date '+%H:%M:%S')] verified: both cameras present"

echo "[$(date '+%H:%M:%S')] launching in tmux session '${TMUX_SESSION}'"
tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TMUX_SESSION}" -c "${SCRIPT_DIR}" \
  "source '${CONDA_SH}' && conda activate ${CONDA_ENV} && \
   PANDA_TOTAL_EPISODES=900 IIWA_EPISODES=1000 UR5E_EPISODES=1000 \
   USE_PANDA_HUMAN=false NORMALIZE_TASK_LANGUAGE=true \
   CAMERAS='${FRONT_CAM} ${WRIST_CAM}' USE_WRIST_CAM=true \
   BATCH_SIZE=${BATCH_SIZE} \
   JOB_TAG=barx_frontwrist_baseline \
   SOURCE_PANDA_MG='${BARX}/PandaOmron/pretrain/TurnOnSinkFaucet/lerobot' \
   SOURCE_IIWA='${BARX}/IIWAOmron/pretrain/PnPCounterToSink/lerobot' \
   SOURCE_UR5E='${BARX}/UR5eOmron/pretrain/PnPSinkToCounter/lerobot' \
   DATASET_ROOT='${DATASET_ROOT}' \
   ./train_smolVLA_robocasa_x.sh 2>&1 | tee outputs/logs/barx_frontwrist_\$(date +%Y%m%d_%H%M%S).log; exec bash"

for _ in $(seq 1 60); do
  sleep 10
  if pgrep -f "lerobot_train.*\.py" >/dev/null 2>&1; then
    echo "[$(date '+%H:%M:%S')] training is up (attach: tmux attach -t ${TMUX_SESSION})"
    exit 0
  fi
  tmux has-session -t "${TMUX_SESSION}" 2>/dev/null || { echo "tmux session died" >&2; exit 1; }
done
echo "training did not come up within 10 minutes; pane output:" >&2
tmux capture-pane -p -t "${TMUX_SESSION}" -S -40 >&2
exit 1
