#!/usr/bin/env bash

# Waits for the currently-running visual-robust training (3 embodiments) to finish, then runs the
# same training against the 4-embodiment auxiliary dataset that now includes JacoOmron.
#
# Jaco was added upstream as extra *cameras inside the existing* <task>/lerobot trees (9 -> 12
# cameras), not as a new task tree. The already-converted 3-embodiment copy under
# dataset_git/visual_robust_robocasa_x is therefore NOT upgradable in place: re-downloading over a
# converted tree refills it with the original per-episode files and reverts meta/info.json to v2.1,
# which silently corrupts it (this happened once already). So the 4-embodiment export is fetched
# into its own directory, dataset_git/visual_robust_robocasa_x_jaco, and the 3-embodiment copy is
# left intact for comparison.
#
# The training itself runs inside `tmux new -s smolvla`, as requested.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
VR_ROOT="${VR_ROOT:-${SCRIPT_DIR}/dataset_git/visual_robust_robocasa_x_jaco}"
POLL_S="${POLL_S:-60}"
TMUX_SESSION="${TMUX_SESSION:-smolvla}"
EXPECTED_FRONT_VIEWS="${EXPECTED_FRONT_VIEWS:-4}"

CONDA_ENV="${CONDA_ENV:-smolvla}"
# Remembered so the tmux command below can source it too. `tmux new-session "conda activate ..."`
# runs in a non-interactive shell with no conda hook installed, which fails with
# "CondaError: Run 'conda init' before 'conda activate'" -- activating there needs this explicitly.
CONDA_SH=""
for candidate in \
  "$(dirname "$(dirname "${CONDA_EXE:-/nonexistent}")")/etc/profile.d/conda.sh" \
  "/opt/conda/etc/profile.d/conda.sh" \
  "${HOME}/miniforge3/etc/profile.d/conda.sh" \
  "${HOME}/miniconda3/etc/profile.d/conda.sh" \
  "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
  if [[ -f "${candidate}" ]]; then
    CONDA_SH="${candidate}"
    break
  fi
done
if [[ -z "${CONDA_SH}" ]]; then
  echo "could not locate conda.sh" >&2
  exit 1
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}" || exit 1
fi
command -v accelerate >/dev/null 2>&1 || { echo "accelerate not on PATH" >&2; exit 1; }

echo "=================================================================================="
echo "queued: PnP + visual robust with Jaco (4 embodiments)  [$(date '+%Y-%m-%d %H:%M:%S')]"
echo "  conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo "  VR_ROOT:   ${VR_ROOT}"
echo "  waiting for the current training to finish"
echo "=================================================================================="

# Wait on the backoff *wrapper* as well as the trainer itself. Between two OOM rungs the wrapper is
# alive but no lerobot_train process exists (it is inside wait_for_free_gpus), so watching only the
# trainer would let this run barge in during that gap and put two trainings on the same GPUs.
while pgrep -f "lerobot_train.*\.py" >/dev/null 2>&1 \
   || pgrep -f "run_visual_robust_with_oom_backoff.sh" >/dev/null 2>&1; do
  sleep "${POLL_S}"
done
echo "[$(date '+%H:%M:%S')] no training running any more"

# The download may still be going (it started while the previous training was running) -- wait it
# out rather than racing it.
while pgrep -f "download_visual_robust_x.py" >/dev/null 2>&1; do
  echo "[$(date '+%H:%M:%S')] waiting for the Jaco download to finish"
  sleep 30
done

echo "[$(date '+%H:%M:%S')] checking Jaco dataset readiness (need >= ${EXPECTED_FRONT_VIEWS} front views)"
if python "${SCRIPT_DIR}/tools/check_visual_robust_ready.py" \
     --root "${VR_ROOT}" --min-front-views "${EXPECTED_FRONT_VIEWS}"; then
  echo "[$(date '+%H:%M:%S')] already prepared -- skipping download and setup"
else
  echo "[$(date '+%H:%M:%S')] not ready; downloading + preparing"
  if ! python "${SCRIPT_DIR}/tools/download_visual_robust_x.py" --dest "${VR_ROOT}" --cameras 12; then
    echo "Jaco download failed; not starting training." >&2
    exit 1
  fi
  if ! VR_ROOT="${VR_ROOT}" "${SCRIPT_DIR}/setup_visual_robust_x_dataset.sh"; then
    echo "Jaco dataset prep failed; not starting training." >&2
    exit 1
  fi
  if ! python "${SCRIPT_DIR}/tools/check_visual_robust_ready.py" \
        --root "${VR_ROOT}" --min-front-views "${EXPECTED_FRONT_VIEWS}"; then
    echo "Jaco dataset still not ready after prep; not starting training." >&2
    exit 1
  fi
fi

# VISUAL_ROBUST_MAX_VIEWS is deliberately left unset: train_smolVLA_robocasa_x_visual_robust.sh
# derives it by counting the observation.image.* features in VR_ROOT, so it picks up 4 here without
# anyone having to remember to bump a constant.
echo "[$(date '+%H:%M:%S')] launching training in tmux session '${TMUX_SESSION}'"
tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TMUX_SESSION}" -c "${SCRIPT_DIR}" \
  "source '${CONDA_SH}' && conda activate ${CONDA_ENV} && VR_ROOT='${VR_ROOT}' ./run_visual_robust_with_oom_backoff.sh 2>&1 | tee outputs/logs/vr_jaco_session.log; exec bash"

# Creating the session always "succeeds" even if the command inside dies instantly, so confirm the
# training actually came up rather than trusting has-session.
for _ in $(seq 1 60); do
  sleep 10
  if pgrep -f "lerobot_train_with_visual_robust.py" >/dev/null 2>&1; then
    echo "[$(date '+%H:%M:%S')] training is up in tmux session '${TMUX_SESSION}' (attach: tmux attach -t ${TMUX_SESSION})"
    exit 0
  fi
  if ! tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    echo "tmux session '${TMUX_SESSION}' died before training started" >&2
    exit 1
  fi
done

echo "training did not come up within 10 minutes; last pane output:" >&2
tmux capture-pane -p -t "${TMUX_SESSION}" -S -40 >&2
exit 1
