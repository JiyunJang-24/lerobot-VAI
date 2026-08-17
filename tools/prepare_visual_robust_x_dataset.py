#!/usr/bin/env python
"""Reshape the ChiefJang/visual_robust_robocasa_x trees into the camera keys the visual-robust
losses look for.

Each `<task>/lerobot` dataset renders the *same* episode from three embodiments, three views each:

    observation.images.{PandaOmron,UR5eOmron,IIWAOmron}.robot0_agentview_{left,right}
    observation.images.{PandaOmron,UR5eOmron,IIWAOmron}.robot0_eye_in_hand

`lerobot_train_with_visual_robust.py` does not look at those names. It collects every batch key
starting with `observation.image.` as the front contrastive views, and every key starting with
`observation.wrist_image.` as the wrist alignment views (see `_select_visual_robust_image_keys`),
and drops both families from the policy's own inputs (`_is_visual_robust_auxiliary_key`). Note
`observation.image.` is singular -- `observation.images.X` does *not* match it, so without this step
the loss would silently find fewer than two views and return None, i.e. train with no visual-robust
term at all.

So this script, per dataset:
  - drops the `robot0_agentview_left` cameras (the policy is only ever fed agentview_right, so the
    left view would make the contrastive term additionally enforce viewpoint invariance, which is
    not what we are after here),
  - renames  observation.images.<Emb>.robot0_agentview_right -> observation.image.<Emb>
             observation.images.<Emb>.robot0_eye_in_hand     -> observation.wrist_image.<Emb>

leaving 3 front views and 3 wrist views per frame -- one per embodiment of the same scene, which is
exactly the positive group the supervised-contrastive loss expects (all views of one sample share a
label; other samples in the batch are the negatives).

Renaming touches every place a v3.0 dataset records a video key: meta/info.json features,
meta/stats.json, the videos/<key>/ directories, and the `videos/<key>/*` + `stats/<key>/*` columns
in meta/episodes/**.parquet.

Sources must already be v3.0 -- run ./convert_robocasa_to_v30.sh on them first.

Usage:
    python tools/prepare_visual_robust_x_dataset.py --root dataset_git/visual_robust_robocasa_x
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.dataset_tools import remove_feature  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

FRONT_SUFFIX = ".robot0_agentview_right"
WRIST_SUFFIX = ".robot0_eye_in_hand"
DROP_SUFFIX = ".robot0_agentview_left"
SRC_PREFIX = "observation.images."


def log(msg: str) -> None:
    print(f"[prepare_visual_robust_x_dataset] {msg}", flush=True)


def video_keys(info: dict) -> list[str]:
    return [k for k, ft in info.get("features", {}).items() if ft.get("dtype") == "video"]


def target_name_keep_left(key: str) -> str | None:
    """Keep BOTH agentview cameras, each in its own contrastive group.

        observation.images.<Emb>.<bg>.robot0_agentview_right -> observation.image.right.<Emb>.<bg>
        observation.images.<Emb>.<bg>.robot0_agentview_left  -> observation.image.left.<Emb>.<bg>
        observation.images.<Emb>.<bg>.robot0_eye_in_hand     -> observation.wrist_image.<Emb>.<bg>

    The `left.` / `right.` component is what the trainer groups on: it is handed
    --dataset.visual_robust_front_prefixes=observation.image.left.,observation.image.right. and runs
    one contrastive term per prefix. Both still start with `observation.image.`, so they remain
    excluded from the policy's own inputs by _is_visual_robust_auxiliary_key without further changes.
    """
    if not key.startswith(SRC_PREFIX):
        return None
    body = key[len(SRC_PREFIX) :]
    if body.endswith(FRONT_SUFFIX):
        return f"observation.image.right.{body[: -len(FRONT_SUFFIX)]}"
    if body.endswith(DROP_SUFFIX):
        return f"observation.image.left.{body[: -len(DROP_SUFFIX)]}"
    if body.endswith(WRIST_SUFFIX):
        return f"observation.wrist_image.{body[: -len(WRIST_SUFFIX)]}"
    return None


def target_name(key: str) -> str | None:
    """Strip the view suffix and re-prefix by camera family, keeping whatever identifies the render.

    Handles both export shapes seen so far, because everything between the prefix and the view
    suffix is carried through untouched:

        observation.images.<Emb>.robot0_agentview_right              -> observation.image.<Emb>
        observation.images.<Emb>.<background>.robot0_agentview_right -> observation.image.<Emb>.<background>

    so the background-variation export yields 3 embodiments x 4 backgrounds = 12 distinct front
    views rather than collapsing them onto 3 colliding names.
    """
    if not key.startswith(SRC_PREFIX):
        return None
    body = key[len(SRC_PREFIX) :]
    if body.endswith(FRONT_SUFFIX):
        return f"observation.image.{body[: -len(FRONT_SUFFIX)]}"
    if body.endswith(WRIST_SUFFIX):
        return f"observation.wrist_image.{body[: -len(WRIST_SUFFIX)]}"
    return None


def rename_keys(root: Path, mapping: dict[str, str]) -> None:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())

    info["features"] = {mapping.get(k, k): v for k, v in info["features"].items()}
    info_path.write_text(json.dumps(info, indent=4))

    stats_path = root / "meta" / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        stats_path.write_text(json.dumps({mapping.get(k, k): v for k, v in stats.items()}, indent=4))

    videos_dir = root / "videos"
    for old, new in mapping.items():
        src, dst = videos_dir / old, videos_dir / new
        if src.is_dir() and not dst.is_dir():
            src.rename(dst)

    for path in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        df = pd.read_parquet(path)
        renames = {}
        for col in df.columns:
            for old, new in mapping.items():
                if col.startswith(f"videos/{old}/") or col.startswith(f"stats/{old}/"):
                    renames[col] = col.replace(old, new, 1)
                    break
        if renames:
            df.rename(columns=renames).to_parquet(path)


def prepare_one(dataset_root: Path, force: bool, keep_left: bool = False) -> bool:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        log(f"{dataset_root}: codebase_version={info.get('codebase_version')!r}, need v3.0 -- "
            f"run ./convert_robocasa_to_v30.sh on it first")
        return False

    keys = video_keys(info)
    if any(k.startswith("observation.image.") for k in keys) and not force:
        log(f"{dataset_root}: already reshaped ({len(keys)} video keys), skipping")
        return True

    namer = target_name_keep_left if keep_left else target_name
    to_drop = [] if keep_left else [k for k in keys if k.endswith(DROP_SUFFIX)]
    mapping = {k: t for k in keys if (t := namer(k)) is not None}

    if not mapping:
        log(f"{dataset_root}: no {SRC_PREFIX}*{FRONT_SUFFIX}/{WRIST_SUFFIX} cameras found; keys={keys}")
        return False

    log(f"{dataset_root}: dropping {len(to_drop)} left-view camera(s), renaming {len(mapping)}")

    if to_drop:
        tmp = dataset_root.parent / f".{dataset_root.name}_nodrop_tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        ds = LeRobotDataset(repo_id=f"visual_robust_{dataset_root.parent.name}", root=dataset_root)
        remove_feature(ds, feature_names=to_drop, repo_id="visual_robust_tmp", output_dir=tmp)
        shutil.rmtree(dataset_root)
        tmp.rename(dataset_root)

    rename_keys(dataset_root, mapping)

    final = video_keys(json.loads((dataset_root / "meta" / "info.json").read_text()))
    front = sorted(k for k in final if k.startswith("observation.image."))
    wrist = sorted(k for k in final if k.startswith("observation.wrist_image."))
    log(f"{dataset_root}: front views ({len(front)}): {front}")
    log(f"{dataset_root}: wrist views ({len(wrist)}): {wrist}")
    if len(front) < 2:
        log(f"{dataset_root}: ERROR -- fewer than 2 front views, contrastive loss would be skipped")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Dir holding the <task>/lerobot trees")
    parser.add_argument("--force", action="store_true", help="Reshape even if it looks already done")
    parser.add_argument(
        "--keep-left",
        action="store_true",
        help="Keep robot0_agentview_left as its own contrastive group (observation.image.left.*) "
        "instead of dropping it; right becomes observation.image.right.*",
    )
    args = parser.parse_args()

    # .../<task>/lerobot/meta/info.json -> .../<task>/lerobot
    roots = sorted(p.parent.parent for p in args.root.glob("*/lerobot/meta/info.json"))
    if not roots:
        log(f"no <task>/lerobot datasets under {args.root}")
        return 1

    ok = True
    for dataset_root in roots:
        ok &= prepare_one(dataset_root, args.force, keep_left=args.keep_left)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
