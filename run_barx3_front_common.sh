#!/usr/bin/env bash
# Shared configuration for the barx3 baseline vs visual-robust comparison.
#
# Three embodiments, three DIFFERENT tasks -- the task/embodiment correlation this whole
# programme is about:
#     IIWAOmron  -> PnPCounterToSink
#     PandaOmron -> TurnOnSinkFaucet
#     UR5eOmron  -> PnPSinkToCounter
# First 100 episodes of each, matched to the 108-episode visual-robust export so the auxiliary
# corpus is not the smaller side of the comparison.
#
# NORMALIZE_TASK_LANGUAGE is on and is NOT cosmetic here: each robot does a different task, so
# the per-corpus instruction styling (capitalisation, trailing full stop) is perfectly correlated
# with the embodiment, and the language encoder could identify the robot from punctuation instead
# of from the instruction. Both runs share the setting, so it cannot skew the comparison either way.
BASE=dataset_git/barx_panda_ur5e_iiwa
export SOURCE_PANDA_MG="${BASE}/PandaOmron/pretrain/TurnOnSinkFaucet/lerobot"
export SOURCE_IIWA="${BASE}/IIWAOmron/pretrain/PnPCounterToSink/lerobot"
export SOURCE_UR5E="${BASE}/UR5eOmron/pretrain/PnPSinkToCounter/lerobot"
export USE_PANDA_HUMAN=false
export NORMALIZE_TASK_LANGUAGE=true
export PANDA_TOTAL_EPISODES=100 IIWA_EPISODES=100 UR5E_EPISODES=100
export DATASET_ROOT=dataset_git/barx3_front_i100_p100_u100
export CAMERAS="observation.images.robot0_agentview_right"
export USE_WRIST_CAM=false
export USE_STATE=true
