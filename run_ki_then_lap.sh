#!/usr/bin/env bash

# The knowledge-insulation comparison, run back to back on the same box:
#
#   1. objective=fast -- the VLM predicts the FAST-tokenized chunk (~172 ids), pi_0.5 style
#   2. objective=lap  -- the VLM predicts an English sentence (~16 tokens), LAP style
#
# Everything else is held fixed (same corpus, batch, steps, schedule, insulation), so the only
# variable is what the VLM is asked to say about the action. Both use BATCH_SIZE=48: lap alone
# would fit 64, but then the comparison would confound the objective with the batch.
#
# The second run starts only if the first exits 0. Watch with:
#     tmux attach -t smolvla_ki
#     tail -f outputs/logs/barx_frontonly_ki_*.log

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BATCH_SIZE="${BATCH_SIZE:-48}"
STEPS="${STEPS:-50000}"

echo "=== [1/2] knowledge insulation, objective=fast ==="
KI_OBJECTIVE=fast BATCH_SIZE="${BATCH_SIZE}" STEPS="${STEPS}" MAIN_PROCESS_PORT=29530 \
  "${SCRIPT_DIR}/run_barx_frontonly_ki.sh"
fast_status=$?

if [[ ${fast_status} -ne 0 ]]; then
  echo "=== fast run FAILED (exit ${fast_status}); not starting the lap run ==="
  exit "${fast_status}"
fi

echo "=== [2/2] knowledge insulation, objective=lap ==="
KI_OBJECTIVE=lap BATCH_SIZE="${BATCH_SIZE}" STEPS="${STEPS}" MAIN_PROCESS_PORT=29531 \
  "${SCRIPT_DIR}/run_barx_frontonly_ki.sh"
lap_status=$?

echo "=== done: fast exit ${fast_status}, lap exit ${lap_status} ==="
exit "${lap_status}"
