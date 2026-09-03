#!/usr/bin/env python
"""Add the re-exported `eef_pixel` / `eef_in_frame` columns to the already-converted v3.0 trees.

The re-export is v2.1 and one of its info.json files declares FEWER image features than the local
copy (3 of the 6 embodiments -- Jaco among the missing). Downloading meta/ over the local tree would
therefore do two bad things at once: revert the v3.0 conversion, and drop Jaco, which is the whole
point of the experiment. Section 6's rule about never re-downloading over a converted tree applies
exactly here.

So this pulls only the new parquet, joins on (episode_index, frame_index), and writes the two new
columns into the local v3.0 parquet, leaving the local meta's structure intact and adding the two
feature declarations to it.

    python tools/merge_eef_pixel_columns.py --dry-run
    python tools/merge_eef_pixel_columns.py --write
"""

import argparse
import glob
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

DEST = Path("/dataset/jiyun/dataset_git")
NEW_COLUMNS = ["eef_pixel", "eef_in_frame"]
TARGETS = [
    ("ChiefJang/visual_robust_robocasa_x", "new_barx/UR5eOmron_PnPSinkToCounter/lerobot",
     DEST / "visual_robust_new_barx_ur5e/new_barx/UR5eOmron_PnPSinkToCounter/lerobot"),
    ("ChiefJang/barx_panda_ur5e_iiwa", "IIWAOmron/pretrain/PnPSinkToCounter/lerobot",
     DEST / "barx_panda_ur5e_iiwa/IIWAOmron/pretrain/PnPSinkToCounter/lerobot"),
    ("ChiefJang/barx_panda_ur5e_iiwa", "PandaOmron/pretrain/PnPSinkToCounter/lerobot",
     DEST / "barx_panda_ur5e_iiwa/PandaOmron/pretrain/PnPSinkToCounter/lerobot"),
    ("ChiefJang/barx_panda_ur5e_iiwa", "UR5eOmron/pretrain/PnPSinkToCounter/lerobot",
     DEST / "barx_panda_ur5e_iiwa/UR5eOmron/pretrain/PnPSinkToCounter/lerobot"),
]


_LISTING: dict = {}


def log(msg: str) -> None:
    print(f"[merge] {msg}", flush=True)


def fetch_remote(repo: str, prefix: str, workers: int) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download, list_repo_files

    # list_repo_files is itself an API call and is refused once the window is exhausted, so it
    # needs the same backoff as the downloads -- and the listing is cached per repo because it
    # returns 200k+ paths for the visual-robust repo and four targets share two repos.
    listing = _LISTING.get(repo)
    if listing is None:
        for attempt in range(10):
            try:
                listing = list_repo_files(repo, repo_type="dataset")
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 9:
                    raise
                log(f"  listing {repo}: {type(exc).__name__}, waiting 60 s")
                time.sleep(60)
        _LISTING[repo] = listing
    files = sorted(f for f in listing
                   if f.startswith(prefix + "/data/") and f.endswith(".parquet"))
    if not files:
        raise FileNotFoundError(f"{repo}:{prefix} has no parquet under data/")

    def one(name):
        # The hub allows 1000 API requests per 5 minutes and a 1000-file tree blows straight
        # through it, so 429 gets a long backoff rather than the usual quick retry.
        for attempt in range(8):
            try:
                path = hf_hub_download(repo, name, repo_type="dataset")
                return pd.read_parquet(path, columns=["episode_index", "frame_index", *NEW_COLUMNS])
            except Exception as exc:  # noqa: BLE001
                if attempt == 7:
                    raise
                time.sleep(60 if "429" in str(exc) else 2 * (attempt + 1))
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(one, files))
    table = pd.concat(frames, ignore_index=True)
    log(f"  remote: {len(files)} parquet, {len(table)} rows")
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    for repo, prefix, dest in TARGETS:
        log(f"{repo}:{prefix}")
        if not dest.exists():
            log(f"  local tree missing at {dest} -- skipping")
            continue
        remote = fetch_remote(repo, prefix, args.workers)
        remote = remote.drop_duplicates(["episode_index", "frame_index"]).set_index(
            ["episode_index", "frame_index"])

        local_files = sorted(glob.glob(str(dest / "data" / "**" / "*.parquet"), recursive=True))
        total, matched = 0, 0
        for path in local_files:
            local = pd.read_parquet(path)
            if all(c in local.columns for c in NEW_COLUMNS):
                log(f"  {Path(path).name}: already has the columns -- skipping")
                continue
            keys = pd.MultiIndex.from_arrays(
                [local.episode_index.to_numpy(), local.frame_index.to_numpy()])
            joined = remote.reindex(keys)
            hit = joined[NEW_COLUMNS[1]].notna().to_numpy()
            total += len(local)
            matched += int(hit.sum())
            for column in NEW_COLUMNS:
                local[column] = joined[column].to_numpy()
            if args.write:
                local.to_parquet(path, index=False)
        rate = matched / max(total, 1)
        log(f"  joined {matched}/{total} rows ({100 * rate:.1f}%)"
            + ("  WROTE" if args.write else "  (dry run)"))
        if total and rate < 0.999:
            log("  WARNING: the join did not cover every local row -- do not trust the new columns")

        info_path = dest / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        if args.write and NEW_COLUMNS[0] not in info["features"]:
            info["features"]["eef_pixel"] = {"dtype": "float32", "shape": [2],
                                             "names": ["u", "v"]}
            info["features"]["eef_in_frame"] = {"dtype": "bool", "shape": [1], "names": None}
            info_path.write_text(json.dumps(info, indent=4))
            log("  declared the two new features in the LOCAL v3.0 info.json "
                "(local meta otherwise untouched -- it declares embodiments the re-export does not)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
