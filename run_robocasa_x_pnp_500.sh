#!/usr/bin/env bash

# Trains the PnP multi-task cross-embodiment mix at 500 episodes per robot:
#   panda = PnPCounterToStove  (108 human + 392 mg = 500)
#   iiwa  = PnPCounterToSink   (500 of 1000)
#   ur5e  = PnPSinkToCounter   (500 of 1000)
#
# The four sources under dataset_git/robocasa_x_atmoic were already downloaded and converted to
# v3.0 by run_robocasa_x_atomic_pipeline.sh, so this skips straight to prep + training. Unlike the
# 1000-per-robot run, every source here needs a real episode subset (mg 392/9638, iiwa 500/1000,
# ur5e 500/1000), so prep re-encodes at the split boundaries and takes longer than that run's did.
#
# JOB_TAG puts "multi_task_pnp" into the job name -- and therefore into lerobot's output dir and the
# wandb run name -- so these runs stay distinguishable from the TurnOnSinkFaucet mixes, which
# otherwise produce identically-named directories for the same per-robot episode counts.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BASE="${SCRIPT_DIR}/dataset_git/robocasa_x_atmoic"
LOG="${SCRIPT_DIR}/outputs/logs/robocasa_x_pnp_p500_i500_u500_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

echo "Training PnP multi-task mix at 500/500/500 -> ${LOG}"

PANDA_TOTAL_EPISODES=500 \
IIWA_EPISODES=500 \
UR5E_EPISODES=500 \
JOB_TAG=multi_task_pnp \
SOURCE_PANDA_HUMAN="${BASE}/PandaOmron/pretrain/PnPCounterToStove" \
SOURCE_PANDA_MG="${BASE}/mg/PandaOmron/pretrain/PnPCounterToStove" \
SOURCE_IIWA="${BASE}/IIWAOmron/pretrain/PnPCounterToSink/lerobot" \
SOURCE_UR5E="${BASE}/UR5eOmron/pretrain/PnPSinkToCounter/lerobot" \
DATASET_ROOT="${SCRIPT_DIR}/dataset_git/robocasa_x_pnp_p500_i500_u500" \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

if [[ ${status} -eq 0 ]]; then
  echo "PnP 500/500/500 finished OK -- log: ${LOG}"
else
  echo "PnP 500/500/500 FAILED exit ${status} -- see ${LOG}"
fi
echo "free disk: $(df -h --output=avail "${SCRIPT_DIR}" | tail -1 | tr -d ' ')"
exit "${status}"
