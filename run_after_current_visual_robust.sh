#!/usr/bin/env bash

# Waits for the currently-running robocasa_x training to exit, then runs the PnP + visual-robust
# (front contrastive only) training. Kept separate from the training scripts themselves so the
# queueing decision is visible and easy to cancel (just kill this script; the running training is
# untouched).
#
# The wait is on "any lerobot_train* process", so it also covers the case where the current run
# finishes and something else starts in between -- it will simply keep waiting.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
POLL_S="${POLL_S:-60}"

# This runs detached (nohup/setsid) hours after being launched, so it cannot rely on the launching
# shell having had the env active -- train_smolVLA_robocasa_x.sh expects `accelerate` and the repo's
# deps on PATH. Activate explicitly, same as train_smolVLA_visual_robust.sh does.
CONDA_ENV="${CONDA_ENV:-smolvla}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
  for candidate in \
    "$(dirname "$(dirname "${CONDA_EXE:-/nonexistent}")")/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "${HOME}/miniforge3/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
    if [[ -f "${candidate}" ]]; then
      # shellcheck disable=SC1090
      source "${candidate}"
      break
    fi
  done
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found; cannot activate ${CONDA_ENV}." >&2
    exit 1
  fi
  conda activate "${CONDA_ENV}" || exit 1
fi

if ! command -v accelerate >/dev/null 2>&1; then
  echo "accelerate not on PATH after activating ${CONDA_ENV}; aborting." >&2
  exit 1
fi
echo "conda env: ${CONDA_DEFAULT_ENV:-unknown}  ($(command -v accelerate))"

echo "=================================================================================="
echo "queued: PnP + visual robust (front contrastive)  [$(date '+%Y-%m-%d %H:%M:%S')]"
echo "  waiting for the current training to finish before starting"
echo "=================================================================================="

while pgrep -f "lerobot_train.*\.py" >/dev/null 2>&1; do
  sleep "${POLL_S}"
done
echo "[$(date '+%H:%M:%S')] no training running any more"

# Readiness is decided from the *final* shape the trainer needs (v3.0 + the reshaped
# observation.image.* keys); download/setup only run if that check fails.
#
# Do not "just re-run the download to be safe": snapshot_download restores the original per-episode
# files *into the already-converted tree* -- including meta/info.json, which reverts
# codebase_version to v2.1 -- leaving data/ with both the packed file-000.parquet and 108
# episode_*.parquet. LeRobot globs data/*/*.parquet, so it would load both and train on a corrupted
# dataset without erroring. This happened once already and had to be restored from lerobot_old.
echo "[$(date '+%H:%M:%S')] checking visual robust dataset readiness"
if python "${SCRIPT_DIR}/tools/check_visual_robust_ready.py"; then
  echo "[$(date '+%H:%M:%S')] already prepared -- skipping download and setup"
else
  echo "[$(date '+%H:%M:%S')] not ready; downloading + preparing"
  if ! python "${SCRIPT_DIR}/tools/download_visual_robust_x.py"; then
    echo "visual robust download failed; not starting training." >&2
    exit 1
  fi
  if ! "${SCRIPT_DIR}/setup_visual_robust_x_dataset.sh"; then
    echo "visual robust dataset prep failed; not starting training." >&2
    exit 1
  fi
  if ! python "${SCRIPT_DIR}/tools/check_visual_robust_ready.py"; then
    echo "visual robust dataset still not ready after prep; not starting training." >&2
    exit 1
  fi
fi

echo "[$(date '+%H:%M:%S')] starting PnP + visual robust training (with OOM backoff)"
exec "${SCRIPT_DIR}/run_visual_robust_with_oom_backoff.sh"
