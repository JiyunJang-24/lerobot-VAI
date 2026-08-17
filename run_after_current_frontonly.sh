#!/usr/bin/env bash

# Waits for the running visual-robust training (and the front-only data prep) to finish, verifies
# the wrist camera is really gone, then runs the same visual-robust training with the policy fed
# only the front camera.
#
# The verification is the point of this script. `--dataset.use_wrist_cam=false` does NOT remove this
# project's wrist camera: the trainer implements it as
#     wrist_feature_keys = [key for key in ds_meta.features if "wrist" in key]
# and the camera here is `observation.images.robot0_eye_in_hand`, which contains no "wrist"
# substring -- so the flag matches nothing and the policy silently keeps both cameras. The camera is
# therefore removed at prep time via CAMERAS (a real remove_feature on every subset), and this script
# refuses to start training unless the prepared dataset actually has exactly one camera and it is the
# front one.
#
# The visual-robust auxiliary loss is unchanged: front contrastive over the 12 agentview_right
# renders (3 embodiments x 4 backgrounds), wrist alignment weight 0.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
VR_ROOT="${VR_ROOT:-${SCRIPT_DIR}/dataset_git/visual_robust_robocasa_x_bg}"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/dataset_git/robocasa_x_pnp_mgonly_frontonly_p1000_i1000_u1000}"
FRONT_CAM="observation.images.robot0_agentview_right"
WRIST_CAM="observation.images.robot0_eye_in_hand"
TMUX_SESSION="${TMUX_SESSION:-smolvla}"
POLL_S="${POLL_S:-60}"
CONDA_ENV="${CONDA_ENV:-smolvla}"

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
echo "queued: PnP + visual robust, FRONT CAMERA ONLY  [$(date '+%Y-%m-%d %H:%M:%S')]"
echo "  conda env:    ${CONDA_DEFAULT_ENV:-unknown}"
echo "  dataset root: ${DATASET_ROOT}"
echo "  VR root:      ${VR_ROOT}"
echo "=================================================================================="

# Watch the backoff wrapper too: between two OOM rungs it is alive while no trainer process exists,
# and starting here in that gap would put two trainings on the same GPUs.
while pgrep -f "lerobot_train.*\.py" >/dev/null 2>&1 \
   || pgrep -f "run_visual_robust_with_oom_backoff.sh" >/dev/null 2>&1; do
  sleep "${POLL_S}"
done
echo "[$(date '+%H:%M:%S')] no training running any more"

while pgrep -f "prepare_robocasa_x_dataset.py" >/dev/null 2>&1; do
  echo "[$(date '+%H:%M:%S')] waiting for the front-only data prep to finish"
  sleep 30
done

echo "[$(date '+%H:%M:%S')] verifying the prepared dataset is front-only"
if ! python - "${DATASET_ROOT}/raw" "${FRONT_CAM}" "${WRIST_CAM}" <<'PY'
import json, sys
from pathlib import Path

raw, front, wrist = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
ok = True
subsets = sorted(p.parent.parent for p in raw.glob("*/meta/info.json"))
if not subsets:
    print(f"  no subsets found under {raw}")
    sys.exit(1)
for root in subsets:
    info = json.loads((root / "meta" / "info.json").read_text())
    cams = sorted(k for k, v in info["features"].items() if v.get("dtype") == "video")
    status = "OK "
    if wrist in cams:
        status, ok = "BAD", False
    elif cams != [front]:
        status, ok = "BAD", False
    print(f"  {status} {root.name:12s} eps={info['total_episodes']:5d} cameras={cams}")
sys.exit(0 if ok else 1)
PY
then
  echo "front-only verification FAILED -- not starting training." >&2
  exit 1
fi
echo "[$(date '+%H:%M:%S')] verified: wrist camera absent from every subset"

echo "[$(date '+%H:%M:%S')] launching in tmux session '${TMUX_SESSION}'"
tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TMUX_SESSION}" -c "${SCRIPT_DIR}" \
  "source '${CONDA_SH}' && conda activate ${CONDA_ENV} && \
   VR_ROOT='${VR_ROOT}' DATASET_ROOT='${DATASET_ROOT}' \
   CAMERAS='${FRONT_CAM}' USE_WRIST_CAM=false JOB_TAG_SUFFIX=frontonly \
   ./run_visual_robust_with_oom_backoff.sh 2>&1 | tee outputs/logs/vr_frontonly_session.log; exec bash"

for _ in $(seq 1 60); do
  sleep 10
  if pgrep -f "lerobot_train_with_visual_robust.py" >/dev/null 2>&1; then
    echo "[$(date '+%H:%M:%S')] training is up (attach: tmux attach -t ${TMUX_SESSION})"
    exit 0
  fi
  tmux has-session -t "${TMUX_SESSION}" 2>/dev/null || { echo "tmux session died" >&2; exit 1; }
done
echo "training did not come up within 10 minutes; pane output:" >&2
tmux capture-pane -p -t "${TMUX_SESSION}" -S -40 >&2
exit 1
