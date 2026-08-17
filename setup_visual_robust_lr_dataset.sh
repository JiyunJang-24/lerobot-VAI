#!/usr/bin/env bash

# Builds dataset_git/visual_robust_robocasa_x_lr: the same background-variation export, but keeping
# BOTH agentview cameras so the contrastive loss can run left-vs-left and right-vs-right separately.
#
# Source is the `lerobot_old` backups left behind inside dataset_git/visual_robust_robocasa_x_bg by
# the v2.1->v3.0 conversion. Those are the untouched 36-camera v2.1 trees (12 left / 12 right /
# 12 eye_in_hand, episodes_stats.jsonl already present), so nothing has to be re-downloaded.
#
# Built into a NEW directory on purpose: visual_robust_robocasa_x_bg has already been reshaped down
# to 24 keys and may still be read by a running job -- converting in place would both destroy the
# 3-embodiment/right-only variant used for the earlier runs and risk corrupting a live dataset.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
SRC_ROOT="${SRC_ROOT:-${SCRIPT_DIR}/dataset_git/visual_robust_robocasa_x_bg}"
DST_ROOT="${DST_ROOT:-${SCRIPT_DIR}/dataset_git/visual_robust_robocasa_x_lr}"
TASKS=(PickPlaceCounterToSink PickPlaceCounterToStove PickPlaceSinkToCounter)

echo "=== step 1/3: copying the untouched 36-camera backups ==="
mkdir -p "${DST_ROOT}"
for t in "${TASKS[@]}"; do
  src="${SRC_ROOT}/${t}/lerobot_old"
  dst="${DST_ROOT}/${t}/lerobot"
  if [[ ! -d "${src}" ]]; then
    echo "  missing backup: ${src}" >&2
    exit 1
  fi
  if [[ -f "${dst}/meta/info.json" ]]; then
    echo "  ${t}: already present, skipping copy"
    continue
  fi
  mkdir -p "${DST_ROOT}/${t}"
  cp -a "${src}" "${dst}"
  echo "  ${t}: copied ($(du -sh "${dst}" | cut -f1))"
done

echo
echo "=== step 2/3: converting to v3.0 ==="
for t in "${TASKS[@]}"; do
  ds="${DST_ROOT}/${t}/lerobot"
  version="$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('codebase_version'))" "${ds}/meta/info.json")"
  if [[ "${version}" == "v3.0" ]]; then
    echo "  ${t}: already v3.0"
    continue
  fi
  echo "  ${t}: ${version} -> v3.0"
  MAX_JOBS=1 "${SCRIPT_DIR}/convert_robocasa_to_v30.sh" "${ds}" || exit 1
done

echo
echo "=== step 3/3: reshaping cameras (keep left as its own group) ==="
python "${SCRIPT_DIR}/tools/prepare_visual_robust_x_dataset.py" --root "${DST_ROOT}" --keep-left || exit 1

echo
echo "=== verification ==="
python - "${DST_ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
ok = True
for info_path in sorted(root.glob("*/lerobot/meta/info.json")):
    info = json.loads(info_path.read_text())
    vids = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    left = sorted(k for k in vids if k.startswith("observation.image.left."))
    right = sorted(k for k in vids if k.startswith("observation.image.right."))
    wrist = sorted(k for k in vids if k.startswith("observation.wrist_image."))
    name = info_path.parent.parent.parent.name
    status = "OK " if (len(left) >= 2 and len(left) == len(right) and info.get("codebase_version") == "v3.0") else "BAD"
    if status == "BAD":
        ok = False
    print(f"  {status} {name:26s} v={info.get('codebase_version')} left={len(left)} right={len(right)} wrist={len(wrist)}")
sys.exit(0 if ok else 1)
PY
status=$?
if [[ ${status} -eq 0 ]]; then
  echo
  echo "left/right visual robust dataset ready at ${DST_ROOT}"
else
  echo "verification FAILED" >&2
fi
exit "${status}"
