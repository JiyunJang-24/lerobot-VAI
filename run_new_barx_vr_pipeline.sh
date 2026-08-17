#!/usr/bin/env bash

# Waits for the new_barx download, converts it to v3.0, reshapes the camera keys, verifies, then
# trains the barx front-only policy with the visual-robust contrastive loss in `tmux new -s smolvla`.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
VR_ROOT="${VR_ROOT:-${SCRIPT_DIR}/dataset_git/visual_robust_new_barx/new_barx}"
TMUX_SESSION="${TMUX_SESSION:-smolvla}"
CONDA_ENV="${CONDA_ENV:-smolvla}"
POLL_S="${POLL_S:-60}"
# 3 embodiments x agentview_right after the reshape.
EXPECTED_FRONT_VIEWS="${EXPECTED_FRONT_VIEWS:-3}"

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
echo "barx + visual robust (new_barx) pipeline  [$(date '+%Y-%m-%d %H:%M:%S')]"
echo "  VR_ROOT: ${VR_ROOT}"
echo "=================================================================================="

while pgrep -f "visual_robust_new_barx" >/dev/null 2>&1; do
  echo "[$(date '+%H:%M:%S')] waiting for the new_barx download"
  sleep "${POLL_S}"
done
echo "[$(date '+%H:%M:%S')] download finished"

while pgrep -f "lerobot_train.*\.py" >/dev/null 2>&1 \
   || pgrep -f "run_visual_robust_with_oom_backoff.sh" >/dev/null 2>&1; do
  sleep "${POLL_S}"
done

if python "${SCRIPT_DIR}/tools/check_visual_robust_ready.py" \
     --root "${VR_ROOT}" --min-front-views "${EXPECTED_FRONT_VIEWS}" >/dev/null 2>&1; then
  echo "[$(date '+%H:%M:%S')] already prepared"
else
  echo "[$(date '+%H:%M:%S')] converting to v3.0"
  for info in "${VR_ROOT}"/*/lerobot/meta/info.json; do
    [[ -f "${info}" ]] || continue
    ds="$(dirname "$(dirname "${info}")")"
    version="$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('codebase_version'))" "${info}")"
    if [[ "${version}" == "v3.0" ]]; then
      echo "  $(basename "$(dirname "${ds}")"): already v3.0"
      continue
    fi
    echo "  $(basename "$(dirname "${ds}")"): ${version} -> v3.0"
    MAX_JOBS=1 "${SCRIPT_DIR}/convert_robocasa_to_v30.sh" "${ds}" || exit 1
  done

  echo "[$(date '+%H:%M:%S')] reshaping camera keys"
  python "${SCRIPT_DIR}/tools/prepare_visual_robust_x_dataset.py" --root "${VR_ROOT}" || exit 1
fi

echo "[$(date '+%H:%M:%S')] verifying (need >= ${EXPECTED_FRONT_VIEWS} front views)"
python "${SCRIPT_DIR}/tools/check_visual_robust_ready.py" \
  --root "${VR_ROOT}" --min-front-views "${EXPECTED_FRONT_VIEWS}" || {
    echo "verification FAILED -- not starting training." >&2; exit 1; }

echo "[$(date '+%H:%M:%S')] launching in tmux session '${TMUX_SESSION}'"
tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TMUX_SESSION}" -c "${SCRIPT_DIR}" \
  "source '${CONDA_SH}' && conda activate ${CONDA_ENV} && ./run_barx_visualrobust_frontonly.sh 2>&1 | tee outputs/logs/barx_vr_session.log; exec bash"

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
