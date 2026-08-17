#!/usr/bin/env python
"""Fetch just the four RoboCasa subsets needed for the PnP cross-embodiment mix out of
ChiefJang/robocasa_x_atomic_ur5e_iiwa (13.8 GB total; these four are 9.5 GB):

  panda_human  PandaOmron/pretrain/PnPCounterToStove          108 eps  (v2.1)
  panda_mg     mg/PandaOmron/pretrain/PnPCounterToStove      9638 eps  (v2.0)
  iiwa         IIWAOmron/pretrain/PnPCounterToSink/lerobot   1000 eps  (v2.1)
  ur5e         UR5eOmron/pretrain/PnPSinkToCounter/lerobot   1000 eps  (v2.1)

Note the layout is not uniform: the IIWA/UR5e trees keep the dataset one level down in a
`lerobot/` subdir, while the two Panda trees have meta/data/videos at the top. The paths printed
at the end are the actual dataset roots to hand to train_smolVLA_robocasa_x.sh.

snapshot_download skips files already present, so re-running this resumes an interrupted fetch.
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "ChiefJang/robocasa_x_atomic_ur5e_iiwa"
DEST = Path(__file__).resolve().parent.parent / "dataset_git" / "robocasa_x_atmoic"

# label -> (download pattern prefix, dataset root relative to DEST)
SUBSETS = {
    "panda_human": ("PandaOmron/pretrain/PnPCounterToStove", "PandaOmron/pretrain/PnPCounterToStove"),
    "panda_mg": ("mg/PandaOmron/pretrain/PnPCounterToStove", "mg/PandaOmron/pretrain/PnPCounterToStove"),
    "iiwa": ("IIWAOmron/pretrain/PnPCounterToSink", "IIWAOmron/pretrain/PnPCounterToSink/lerobot"),
    "ur5e": ("UR5eOmron/pretrain/PnPSinkToCounter", "UR5eOmron/pretrain/PnPSinkToCounter/lerobot"),
}


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        REPO,
        repo_type="dataset",
        local_dir=str(DEST),
        allow_patterns=[f"{prefix}/*" for prefix, _ in SUBSETS.values()],
        max_workers=8,
    )

    missing = []
    print("\nDataset roots:")
    for label, (_, root) in SUBSETS.items():
        path = DEST / root
        ok = (path / "meta" / "info.json").is_file()
        print(f"  {label:12s} {'OK ' if ok else 'MISSING'} {path}")
        if not ok:
            missing.append(label)

    if missing:
        print(f"\nDOWNLOAD_INCOMPLETE: {', '.join(missing)}")
        return 1
    print("\nDOWNLOAD_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
