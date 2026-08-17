#!/usr/bin/env bash

# One-time prep of the visual-robust auxiliary dataset for
# train_smolVLA_robocasa_x_visual_robust.sh:
#
#   1. v2.1 -> v3.0 conversion of each ChiefJang/visual_robust_robocasa_x <task>/lerobot tree
#      (also fixes this export's `videos/chunk-000/<Emb>.<view>/` dir naming, which the v2.1->v3.0
#      converter cannot glob -- see tools/normalize_video_dirs.py).
#   2. camera selection + rename into the keys the visual-robust losses look for
#      (tools/prepare_visual_robust_x_dataset.py).
#
# Assumes the dataset has already been downloaded to dataset_git/visual_robust_robocasa_x.
# Idempotent: both steps no-op once applied.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
VR_ROOT="${VR_ROOT:-${SCRIPT_DIR}/dataset_git/visual_robust_robocasa_x}"

if [[ ! -d "${VR_ROOT}" ]]; then
  echo "not found: ${VR_ROOT}" >&2
  echo "Download it first:  python tools/download_visual_robust_x.py" >&2
  exit 1
fi

echo "=== step 1/2: converting visual robust trees to v3.0 ==="
for info in "${VR_ROOT}"/*/lerobot/meta/info.json; do
  [[ -f "${info}" ]] || continue
  ds_root="$(dirname "$(dirname "${info}")")"
  version="$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('codebase_version'))" "${info}")"
  if [[ "${version}" == "v3.0" ]]; then
    echo "  $(basename "$(dirname "${ds_root}")"): already v3.0"
    continue
  fi
  echo "  $(basename "$(dirname "${ds_root}")"): ${version} -> v3.0"
  MAX_JOBS=1 "${SCRIPT_DIR}/convert_robocasa_to_v30.sh" "${ds_root}" || exit 1
done

echo
echo "=== step 2/2: selecting + renaming cameras ==="
python "${SCRIPT_DIR}/tools/prepare_visual_robust_x_dataset.py" --root "${VR_ROOT}" || exit 1

echo
echo "Visual robust dataset ready at ${VR_ROOT}"
