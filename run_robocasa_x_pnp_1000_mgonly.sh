#!/usr/bin/env bash

# Re-runs the PnP multi-task cross-embodiment mix at 1000 episodes per robot, but with the panda
# share taken entirely from panda_mg instead of 108 panda_human + 892 panda_mg.
#
# Why: panda_human is the only one of the four subsets whose action carries mobile-base / torso
# motion (dims 0, 1, 2, 4 reach +-1.0). Those dims are exactly constant in panda_mg, iiwa and ur5e,
# so pooling all four into a single MEAN_STD normalizer collapses their std to 0.0006 - 0.022 and
# rescales panda_human's values to |z| up to 278 while every other sample sits at 0. Measured on the
# previous run, those four dims accounted for 36.4% of the total action MSE despite panda_human
# being only 2.1% of frames -- which is why that run's loss floored at 0.095 while the
# TurnOnSinkFaucet mixes (where no subset moves the base, so the same dims normalize to exactly 0)
# reached 0.009.
#
# Dropping panda_human makes all four subsets fixed-base, so those dims normalize to 0 and stop
# competing with the arm/gripper dims that actually matter. panda_mg has 9638 episodes, so it covers
# the full 1000 on its own. The cost is losing the 108 real human teleop demos.
#
# Also restyles every instruction to one convention (NORMALIZE_TASK_LANGUAGE=true): the exports
# phrase them per-robot -- panda "Pick the onion ... in the pan." vs iiwa/ur5e "pick the can ... in
# the sink" -- and since each robot also does a different task, capitalisation alone identifies the
# embodiment, giving the language encoder a shortcut around the instruction's actual content.
#
# Episode counts stay 1000 / 1000 / 1000 as requested. Note this still leaves the per-frame mix
# uneven (panda episodes are shorter), which is a separate, deliberately unaddressed issue.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BASE="${SCRIPT_DIR}/dataset_git/robocasa_x_atmoic"
LOG="${SCRIPT_DIR}/outputs/logs/robocasa_x_pnp_mgonly_p1000_i1000_u1000_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

echo "Training PnP multi-task mix, panda from mg only, 1000/1000/1000 -> ${LOG}"

PANDA_TOTAL_EPISODES=1000 \
IIWA_EPISODES=1000 \
UR5E_EPISODES=1000 \
USE_PANDA_HUMAN=false \
NORMALIZE_TASK_LANGUAGE=true \
JOB_TAG=multi_task_pnp_mgonly_langnorm \
SOURCE_PANDA_MG="${BASE}/mg/PandaOmron/pretrain/PnPCounterToStove" \
SOURCE_IIWA="${BASE}/IIWAOmron/pretrain/PnPCounterToSink/lerobot" \
SOURCE_UR5E="${BASE}/UR5eOmron/pretrain/PnPSinkToCounter/lerobot" \
DATASET_ROOT="${SCRIPT_DIR}/dataset_git/robocasa_x_pnp_mgonly_p1000_i1000_u1000" \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

if [[ ${status} -eq 0 ]]; then
  echo "PnP mg-only 1000/1000/1000 finished OK -- log: ${LOG}"
else
  echo "PnP mg-only 1000/1000/1000 FAILED exit ${status} -- see ${LOG}"
fi
echo "free disk: $(df -h --output=avail "${SCRIPT_DIR}" | tail -1 | tr -d ' ')"
exit "${status}"
