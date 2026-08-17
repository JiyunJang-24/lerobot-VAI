#!/usr/bin/env bash

# Two runs, back to back, both on the left+right agentview cameras (no wrist):
#
#   1. BASELINE  -- plain lerobot_train.py, no visual-robust loss at all.
#   2. VISUAL ROBUST -- same data, plus the front contrastive loss computed SEPARATELY per viewpoint
#      (left renders contrasted only against left, right only against right).
#
# Both use batch 48/GPU. That is the only configuration the pair is comparable at: the 8 auxiliary
# views the second run encodes (4 left + 4 right) do not fit alongside a 2-camera policy at batch 64
# (~88 GB projected vs 81.5 GB available), and running the baseline at a different batch size would
# confound the comparison with a batch-size effect. So the baseline is deliberately run at 48 too,
# even though it would fit at 64.
#
# Runs inside `tmux new -s smolvla` so progress can be watched with `tmux attach -t smolvla`.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
POLICY_ROOT="${POLICY_ROOT:-${SCRIPT_DIR}/dataset_git/robocasa_x_pnp_mgonly_lr_p1000_i1000_u1000}"
VR_ROOT="${VR_ROOT:-${SCRIPT_DIR}/dataset_git/visual_robust_robocasa_x_lr}"
LEFT_CAM="observation.images.robot0_agentview_left"
RIGHT_CAM="observation.images.robot0_agentview_right"
WRIST_CAM="observation.images.robot0_eye_in_hand"
BATCH_SIZE="${BATCH_SIZE:-48}"
POLL_S="${POLL_S:-60}"
SUMMARY="${SCRIPT_DIR}/outputs/logs/lr_pair_summary.log"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${SUMMARY}"; }

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

log "=================================================================================="
log "left+right pair: baseline -> visual robust   (batch ${BATCH_SIZE}/GPU)"
log "  policy data: ${POLICY_ROOT}"
log "  VR data:     ${VR_ROOT}"
log "=================================================================================="

# --- wait for anything still running / still preparing -------------------------------------------
while pgrep -f "lerobot_train.*\.py" >/dev/null 2>&1 \
   || pgrep -f "run_visual_robust_with_oom_backoff.sh" >/dev/null 2>&1; do
  sleep "${POLL_S}"
done
while pgrep -f "prepare_robocasa_x_dataset.py" >/dev/null 2>&1 \
   || pgrep -f "setup_visual_robust_lr_dataset.sh" >/dev/null 2>&1 \
   || pgrep -f "prepare_visual_robust_x_dataset.py" >/dev/null 2>&1; do
  log "waiting for data prep to finish"
  sleep 30
done

# --- verify both datasets before burning ~20 GPU-hours -------------------------------------------
log "verifying policy dataset is left+right with no wrist"
if ! python - "${POLICY_ROOT}/raw" "${LEFT_CAM}" "${RIGHT_CAM}" "${WRIST_CAM}" <<'PY'
import json, sys
from pathlib import Path
raw, left, right, wrist = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
subsets = sorted(p.parent.parent for p in raw.glob("*/meta/info.json"))
if not subsets:
    print(f"  no subsets under {raw}")
    sys.exit(1)
ok = True
for root in subsets:
    info = json.loads((root / "meta" / "info.json").read_text())
    cams = sorted(k for k, v in info["features"].items() if v.get("dtype") == "video")
    good = cams == sorted([left, right])
    if wrist in cams or not good:
        ok = False
    print(f"  {'OK ' if good else 'BAD'} {root.name:12s} eps={info['total_episodes']:5d} cameras={[c.replace('observation.images.','') for c in cams]}")
sys.exit(0 if ok else 1)
PY
then
  log "policy dataset verification FAILED -- not starting."
  exit 1
fi

log "verifying VR dataset has separate left/right groups"
if ! python - "${VR_ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
infos = sorted(root.glob("*/lerobot/meta/info.json"))
if not infos:
    print(f"  no <task>/lerobot under {root}")
    sys.exit(1)
ok = True
for info_path in infos:
    info = json.loads(info_path.read_text())
    vids = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    l = [k for k in vids if k.startswith("observation.image.left.")]
    r = [k for k in vids if k.startswith("observation.image.right.")]
    good = info.get("codebase_version") == "v3.0" and len(l) >= 2 and len(l) == len(r)
    if not good:
        ok = False
    print(f"  {'OK ' if good else 'BAD'} {info_path.parts[-4]:26s} v={info.get('codebase_version')} left={len(l)} right={len(r)}")
sys.exit(0 if ok else 1)
PY
then
  log "VR dataset verification FAILED -- not starting."
  exit 1
fi
log "both datasets verified"

run_in_tmux() {
  local name="$1" inner="$2" session="smolvla"
  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" -c "${SCRIPT_DIR}" \
    "source '${CONDA_SH}' && conda activate ${CONDA_ENV} && ${inner}; exec bash"
  for _ in $(seq 1 60); do
    sleep 10
    pgrep -f "lerobot_train.*\.py" >/dev/null 2>&1 && { log "${name}: training is up"; return 0; }
    tmux has-session -t "${session}" 2>/dev/null || { log "${name}: tmux session died"; return 1; }
  done
  log "${name}: did not come up within 10 min; pane:"
  tmux capture-pane -p -t "${session}" -S -40 | tee -a "${SUMMARY}"
  return 1
}

wait_for_training_end() {
  while pgrep -f "lerobot_train.*\.py" >/dev/null 2>&1 \
     || pgrep -f "run_visual_robust_with_oom_backoff.sh" >/dev/null 2>&1; do
    sleep "${POLL_S}"
  done
}

# --- run 1: baseline, no visual robust ----------------------------------------------------------
log "### RUN 1/2: baseline (left+right, NO visual robust), batch ${BATCH_SIZE}"
BASELINE_LOG="${SCRIPT_DIR}/outputs/logs/lr_baseline_$(date +%Y%m%d_%H%M%S).log"
run_in_tmux "baseline" "\
PANDA_TOTAL_EPISODES=1000 IIWA_EPISODES=1000 UR5E_EPISODES=1000 \
USE_PANDA_HUMAN=false NORMALIZE_TASK_LANGUAGE=true \
CAMERAS='${LEFT_CAM} ${RIGHT_CAM}' USE_WRIST_CAM=false \
BATCH_SIZE=${BATCH_SIZE} \
JOB_TAG=multi_task_pnp_mgonly_langnorm_lr_baseline \
SOURCE_PANDA_MG='${SCRIPT_DIR}/dataset_git/robocasa_x_atmoic/mg/PandaOmron/pretrain/PnPCounterToStove' \
SOURCE_IIWA='${SCRIPT_DIR}/dataset_git/robocasa_x_atmoic/IIWAOmron/pretrain/PnPCounterToSink/lerobot' \
SOURCE_UR5E='${SCRIPT_DIR}/dataset_git/robocasa_x_atmoic/UR5eOmron/pretrain/PnPSinkToCounter/lerobot' \
DATASET_ROOT='${POLICY_ROOT}' \
./train_smolVLA_robocasa_x.sh 2>&1 | tee ${BASELINE_LOG}" || exit 1

wait_for_training_end
if grep -q "End of training" "${BASELINE_LOG}" 2>/dev/null; then
  log "### RUN 1/2 baseline finished OK -- ${BASELINE_LOG}"
else
  log "### RUN 1/2 baseline did NOT reach 'End of training' -- see ${BASELINE_LOG}"
  log "stopping; run 2 not started."
  exit 1
fi

# --- run 2: same data + visual robust, left/right groups separate --------------------------------
log "### RUN 2/2: left+right WITH visual robust (left<->left, right<->right), batch ${BATCH_SIZE}"
run_in_tmux "visual-robust" "\
VR_ROOT='${VR_ROOT}' DATASET_ROOT='${POLICY_ROOT}' \
CAMERAS='${LEFT_CAM} ${RIGHT_CAM}' USE_WRIST_CAM=false \
BATCH_SIZE=${BATCH_SIZE} \
VISUAL_ROBUST_FRONT_PREFIXES='observation.image.left.,observation.image.right.' \
JOB_TAG_SUFFIX=lr \
./run_visual_robust_with_oom_backoff.sh 2>&1 | tee outputs/logs/lr_visualrobust_session.log" || exit 1

log "both runs launched; run 2 in progress (attach: tmux attach -t smolvla)"
