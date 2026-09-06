#!/usr/bin/env python
"""Download ChiefJang/visual_robust_libero.

Everything lands under /dataset (NFS, 96 TB free), never the root disk: the HF cache and outputs
filled the 292 GB root once already and killed every running job, so HF_HOME is redirected too.

Reports each LeRobot tree's version and video-file completeness rather than assuming, because a
previous export shipped data/ and meta/ with videos/ absent -- info.json declared 231,678 frames
and not one could be decoded.
"""

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "ChiefJang/visual_robust_libero"
DEST = Path("/dataset/jiyun/dataset_git/visual_robust_libero")
os.environ.setdefault("HF_HOME", "/dataset/jiyun/hf_cache")


def log(msg: str) -> None:
    print(f"[libero] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include", default="", help="comma-separated top-level prefixes; all if empty")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = [(f.path, f.size) for f in
             api.list_repo_tree(REPO, repo_type="dataset", recursive=True)
             if getattr(f, "size", None) is not None]
    if args.include:
        prefixes = tuple(p for p in args.include.split(",") if p)
        files = [(p, s) for p, s in files if p.startswith(prefixes)]
    total = sum(s for _, s in files)
    log(f"{len(files)} files, {total / 1e9:.1f} GB -> {DEST}")
    if args.dry_run:
        return 0

    done = [0]

    def one(item):
        name, _ = item
        out = DEST / name
        if out.exists():
            done[0] += 1
            return None
        for attempt in range(6):
            try:
                local = hf_hub_download(REPO, name, repo_type="dataset")
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(local, out)
                done[0] += 1
                if done[0] % 50 == 0:
                    log(f"  {done[0]}/{len(files)}")
                return None
            except Exception as exc:  # noqa: BLE001
                if attempt == 5:
                    return f"{name}: {exc}"
                # the hub allows 1000 API requests per 5 minutes; a 429 needs a long wait
                time.sleep(60 if "429" in str(exc) else 3 * (attempt + 1))
        return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        errors = [e for e in pool.map(one, files) if e]
    log(f"downloaded {len(files) - len(errors)}/{len(files)}")
    for e in errors[:5]:
        log(f"  FAILED {e}")

    log("--- LeRobot trees found:")
    for info_path in sorted(DEST.rglob("meta/info.json")):
        tree = info_path.parent.parent
        info = json.loads(info_path.read_text())
        videos = len(list((tree / "videos").rglob("*.mp4"))) if (tree / "videos").exists() else 0
        streams = [k for k, v in info["features"].items() if v["dtype"] == "video"]
        episodes = int(info.get("total_episodes", 0))
        want = episodes * max(1, len(streams))
        state = "ok" if videos >= want else f"INCOMPLETE (want {want})"
        log(f"  {str(tree.relative_to(DEST)):<62} {info.get('codebase_version')} "
            f"ep={episodes:<5} frames={info.get('total_frames'):<8} mp4={videos} {state}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
