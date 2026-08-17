#!/usr/bin/env bash

# Prepares the rebuilt ChiefJang/visual_robust_robocasa_x export (now with background variation) and
# runs train_smolVLA_robocasa_x_visual_robust.sh against it, in `tmux new -s smolvla`.
#
# What changed upstream, and what it forces:
#   - 36 cameras per episode = 3 embodiments (IIWA / Panda / UR5e) x 4 backgrounds
#     (dark / plain / warm / white) x 3 views. Jaco is NOT in this rebuild.
#   - After dropping agentview_left, that is 12 front views per frame, not 3-4.
#   - Encoding 12 views/step does not fit (measured ~2.7 GB/view at vr_batch=8, and 4 views already
#     sits at 77.4 of 81.5 GB), so only VISUAL_ROBUST_VIEW_BUDGET of them are used per step.
#   - Which 4 therefore has to be RANDOM. The selector keeps the alphabetically-first keys, and they
#     sort by embodiment, so a fixed cut takes IIWAOmron.{dark,plain,warm,white} -- one robot, zero
#     cross-embodiment positive pairs. train_smolVLA_robocasa_x_visual_robust.sh turns random
#     sampling on automatically whenever available > budget.
#
# Waits for the download to finish rather than racing it, and never re-downloads over an
# already-converted tree (that silently corrupts it -- see tools/check_visual_robust_ready.py).

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
VR_ROOT="${VR_ROOT:-${SCRIPT_DIR}/dataset_git/visual_robust_robocasa_x_bg}"
TMUX_SESSION="${TMUX_SESSION:-smolvla}"
EXPECTED_FRONT_VIEWS="${EXPECTED_FRONT_VIEWS:-12}"
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
echo "visual robust w/ background variation  [$(date '+%Y-%m-%d %H:%M:%S')]"
echo "  conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo "  VR_ROOT:   ${VR_ROOT}"
echo "=================================================================================="

while pgrep -f "download_visual_robust_x.py" >/dev/null 2>&1; do
  echo "[$(date '+%H:%M:%S')] waiting for the download to finish"
  sleep 60
done

echo "[$(date '+%H:%M:%S')] checking readiness (need >= ${EXPECTED_FRONT_VIEWS} front views)"
if python "${SCRIPT_DIR}/tools/check_visual_robust_ready.py" \
     --root "${VR_ROOT}" --min-front-views "${EXPECTED_FRONT_VIEWS}"; then
  echo "[$(date '+%H:%M:%S')] already prepared"
else
  echo "[$(date '+%H:%M:%S')] preparing (v3.0 convert + camera reshape)"
  if ! python "${SCRIPT_DIR}/tools/download_visual_robust_x.py" --dest "${VR_ROOT}" --cameras 36; then
    echo "download failed; not starting training." >&2
    exit 1
  fi
  if ! VR_ROOT="${VR_ROOT}" "${SCRIPT_DIR}/setup_visual_robust_x_dataset.sh"; then
    echo "dataset prep failed; not starting training." >&2
    exit 1
  fi
  if ! python "${SCRIPT_DIR}/tools/check_visual_robust_ready.py" \
        --root "${VR_ROOT}" --min-front-views "${EXPECTED_FRONT_VIEWS}"; then
    echo "still not ready after prep; not starting training." >&2
    exit 1
  fi
fi

echo "[$(date '+%H:%M:%S')] launching training in tmux session '${TMUX_SESSION}'"
tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TMUX_SESSION}" -c "${SCRIPT_DIR}" \
  "source '${CONDA_SH}' && conda activate ${CONDA_ENV} && VR_ROOT='${VR_ROOT}' ./run_visual_robust_with_oom_backoff.sh 2>&1 | tee outputs/logs/vr_bg_session.log; exec bash"

# tmux reports success even if the command inside dies instantly, so confirm the trainer came up.
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
