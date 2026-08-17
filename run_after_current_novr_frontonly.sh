#!/usr/bin/env bash

# Waits for the running front-only visual-robust training (and its OOM-backoff wrapper) to finish,
# re-verifies the prepared dataset is really front-only, then starts the same run WITHOUT the
# visual-robust loss (run_novr_frontonly.sh) in a new window of the `smolvla` tmux session.
#
# The existing window is left alone rather than killed, so the finished visual-robust run's output
# stays attachable; the new window becomes the active one, so `tmux attach -t smolvla` lands on it.
#
# The backoff wrapper is polled as well as the trainer: between two OOM rungs it is alive while no
# trainer process exists, and starting in that gap would put two trainings on the same 8 GPUs.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/dataset_git/robocasa_x_pnp_mgonly_frontonly_p1000_i1000_u1000}"
FRONT_CAM="observation.images.robot0_agentview_right"
WRIST_CAM="observation.images.robot0_eye_in_hand"
TMUX_SESSION="${TMUX_SESSION:-smolvla}"
TMUX_WINDOW="${TMUX_WINDOW:-novr}"
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
echo "queued: PnP baseline, NO visual robust loss, FRONT CAMERA ONLY  [$(date '+%Y-%m-%d %H:%M:%S')]"
echo "  dataset root: ${DATASET_ROOT}"
echo "  tmux:         ${TMUX_SESSION}:${TMUX_WINDOW}"
echo "=================================================================================="

while pgrep -f "lerobot_train.*\.py" >/dev/null 2>&1 \
   || pgrep -f "run_visual_robust_with_oom_backoff.sh" >/dev/null 2>&1; do
  sleep "${POLL_S}"
done
echo "[$(date '+%H:%M:%S')] no training running any more"

# Dead ranks can hold GPU memory for a few seconds after the trainer exits.
for _ in $(seq 1 60); do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)"
  [[ "${used:-9999}" -lt 2000 ]] && break
  sleep 5
done
echo "[$(date '+%H:%M:%S')] GPUs free (max used: ${used:-unknown} MiB)"

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
    if wrist in cams or cams != [front]:
        status, ok = "BAD", False
    print(f"  {status} {root.name:12s} eps={info['total_episodes']:5d} cameras={cams}")
sys.exit(0 if ok else 1)
PY
then
  echo "front-only verification FAILED -- not starting training." >&2
  exit 1
fi
echo "[$(date '+%H:%M:%S')] verified: wrist camera absent from every subset"

echo "[$(date '+%H:%M:%S')] launching in tmux ${TMUX_SESSION}:${TMUX_WINDOW}"
LAUNCH="source '${CONDA_SH}' && conda activate ${CONDA_ENV} && \
  DATASET_ROOT='${DATASET_ROOT}' ./run_novr_frontonly.sh 2>&1 | tee outputs/logs/novr_frontonly_session.log; exec bash"
if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  tmux new-window -t "${TMUX_SESSION}" -n "${TMUX_WINDOW}" -c "${SCRIPT_DIR}" "${LAUNCH}"
else
  tmux new-session -d -s "${TMUX_SESSION}" -n "${TMUX_WINDOW}" -c "${SCRIPT_DIR}" "${LAUNCH}"
fi

for _ in $(seq 1 60); do
  sleep 10
  if pgrep -f "lerobot_train\.py" >/dev/null 2>&1; then
    echo "[$(date '+%H:%M:%S')] training is up (attach: tmux attach -t ${TMUX_SESSION})"
    exit 0
  fi
  tmux has-session -t "${TMUX_SESSION}" 2>/dev/null || { echo "tmux session died" >&2; exit 1; }
done
echo "training did not come up within 10 minutes; pane output:" >&2
tmux capture-pane -p -t "${TMUX_SESSION}:${TMUX_WINDOW}" -S -40 >&2
exit 1
