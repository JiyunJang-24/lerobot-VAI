#!/usr/bin/env bash

# Converts the three ChiefJang/barx_panda_ur5e_iiwa trees to codebase_version v3.0, which is what
# tools/prepare_robocasa_x_dataset.py requires. Idempotent.
#
# convert_robocasa_to_v30.sh decides what to run from what is actually on disk rather than from the
# declared version, so it handles this export whether or not meta/episodes_stats.jsonl is present.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BARX="${BARX:-${SCRIPT_DIR}/dataset_git/barx_panda_ur5e_iiwa}"

TREES=(
  "PandaOmron/pretrain/TurnOnSinkFaucet/lerobot"
  "UR5eOmron/pretrain/PnPSinkToCounter/lerobot"
  "IIWAOmron/pretrain/PnPCounterToSink/lerobot"
)

echo "=== converting barx trees to v3.0 ==="
for t in "${TREES[@]}"; do
  ds="${BARX}/${t}"
  if [[ ! -f "${ds}/meta/info.json" ]]; then
    echo "  missing: ${ds}" >&2
    exit 1
  fi
  version="$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('codebase_version'))" "${ds}/meta/info.json")"
  if [[ "${version}" == "v3.0" ]]; then
    echo "  ${t%%/*}: already v3.0"
    continue
  fi
  echo "  ${t%%/*}: ${version} -> v3.0"
  MAX_JOBS=1 "${SCRIPT_DIR}/convert_robocasa_to_v30.sh" "${ds}" || exit 1
done

echo
echo "=== verification ==="
python - "${BARX}" <<'PY'
import json, sys
from pathlib import Path
barx = Path(sys.argv[1])
trees = {
    "panda(TurnOnSinkFaucet)": "PandaOmron/pretrain/TurnOnSinkFaucet/lerobot",
    "ur5e(PnPSinkToCounter)": "UR5eOmron/pretrain/PnPSinkToCounter/lerobot",
    "iiwa(PnPCounterToSink)": "IIWAOmron/pretrain/PnPCounterToSink/lerobot",
}
ok = True
for name, rel in trees.items():
    info = json.loads((barx / rel / "meta" / "info.json").read_text())
    cams = sorted(k for k, v in info["features"].items() if v.get("dtype") == "video")
    good = info.get("codebase_version") == "v3.0" and any("agentview_right" in c for c in cams)
    if not good:
        ok = False
    print(f"  {'OK ' if good else 'BAD'} {name:24s} v={info.get('codebase_version')} "
          f"eps={info['total_episodes']:5d} tasks={info.get('total_tasks')} "
          f"cams={[c.replace('observation.images.', '') for c in cams]}")
sys.exit(0 if ok else 1)
PY
status=$?
[[ ${status} -eq 0 ]] && echo && echo "barx dataset ready at ${BARX}" || echo "verification FAILED" >&2
exit "${status}"
