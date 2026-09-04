#!/usr/bin/env python
"""Download the redesigned export once it is completely uploaded, and convert it to v3.0.

Exits non-zero until every declared video stream has a file per episode. That check exists because
a previous subset shipped with data/ and meta/ present and videos/ missing: info.json declared
231,678 frames and not one of them could be decoded, and nothing about the listing looked wrong.
"""

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "ChiefJang/visual_robust_barx"
DEST = Path("/dataset/jiyun/dataset_git/eef_pairs")


def log(msg: str) -> None:
    print(f"[fetch_v2] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default="56combo_8000_local_motion_27cam_kitchen_l8s0")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    prefix = f"eef_pairs/{args.subset}"
    info_path = hf_hub_download(REPO, f"{prefix}/meta/info.json", repo_type="dataset")
    info = json.loads(Path(info_path).read_text())
    episodes = int(info["total_episodes"])
    streams = [k for k, v in info["features"].items() if v["dtype"] == "video"]

    files = [f.path for f in api.list_repo_tree(REPO, prefix, repo_type="dataset", recursive=True)
             if getattr(f, "size", None) is not None]
    ready = True
    for stream in streams:
        # the stream name sits AFTER the chunk directory: videos/chunk-000/<stream>/episode_*.mp4
        n = sum(1 for f in files if f"/{stream}/" in f and "/videos/" in f)
        state = "ok" if n >= episodes else "INCOMPLETE"
        log(f"  {stream:<44} {n}/{episodes} {state}")
        ready = ready and n >= episodes
    parquet = sum(1 for f in files if "/data/" in f and f.endswith(".parquet"))
    log(f"  parquet {parquet}/{episodes}")
    ready = ready and parquet >= episodes
    if not ready:
        log("upload not finished")
        return 1
    if not args.write:
        log("complete (dry run)")
        return 0

    dest = DEST / args.subset
    log(f"downloading {len(files)} files -> {dest}")

    def one(name):
        for attempt in range(6):
            try:
                local = hf_hub_download(REPO, name, repo_type="dataset")
                out = dest / Path(name).relative_to(prefix)
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(local, out)
                return None
            except Exception as exc:  # noqa: BLE001
                if attempt == 5:
                    return f"{name}: {exc}"
                time.sleep(60 if "429" in str(exc) else 3 * (attempt + 1))
        return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        errors = [e for e in pool.map(one, files) if e]
    if errors:
        log(f"{len(errors)} downloads failed, e.g. {errors[0][:120]}")
        return 1

    import subprocess

    subprocess.run([sys.executable, "tools/add_episode_stats_count.py", str(dest)], check=False)
    if json.loads((dest / "meta/info.json").read_text()).get("codebase_version") != "v3.0":
        from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset

        convert_dataset(repo_id=args.subset, root=str(DEST), push_to_hub=False)  # root is the PARENT
    log(f"ready at {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
