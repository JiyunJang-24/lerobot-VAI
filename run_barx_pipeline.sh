#!/usr/bin/env bash

# Waits for the barx download, converts to v3.0, verifies, then runs the front-camera-only baseline
# in `tmux new -s smolvla`.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BARX="${BARX:-${SCRIPT_DIR}/dataset_git/barx_panda_ur5e_iiwa}"
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
echo "barx baseline pipeline (front camera only)  [$(date '+%Y-%m-%d %H:%M:%S')]"
echo "  conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo "  barx root: ${BARX}"
echo "=================================================================================="

while pgrep -f "barx_panda_ur5e_iiwa" >/dev/null 2>&1; do
  echo "[$(date '+%H:%M:%S')] waiting for the barx download"
  sleep "${POLL_S}"
done
echo "[$(date '+%H:%M:%S')] download finished"

while pgrep -f "lerobot_train.*\.py" >/dev/null 2>&1 \
   || pgrep -f "run_visual_robust_with_oom_backoff.sh" >/dev/null 2>&1; do
  sleep "${POLL_S}"
done

echo "[$(date '+%H:%M:%S')] converting to v3.0"
if ! "${SCRIPT_DIR}/setup_barx_dataset.sh"; then
  echo "barx conversion failed; not starting training." >&2
  exit 1
fi

echo "[$(date '+%H:%M:%S')] launching baseline in tmux session '${TMUX_SESSION}'"
tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TMUX_SESSION}" -c "${SCRIPT_DIR}" \
  "source '${CONDA_SH}' && conda activate ${CONDA_ENV} && ./run_barx_baseline_frontonly.sh 2>&1 | tee outputs/logs/barx_session.log; exec bash"

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
