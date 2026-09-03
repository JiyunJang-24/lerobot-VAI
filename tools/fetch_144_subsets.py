#!/usr/bin/env python
"""Pull every 56combo_144_* subset that exists on the hub, and convert each to v3.0.

Exits non-zero until at least one NEW subset (beyond the closed one already local) has landed with
its videos/, so the caller can keep polling. A subset without videos/ is skipped rather than half
downloaded -- info.json declares the frames whether or not the mp4s were uploaded, so absence of
videos/ is the only reliable readiness signal.
"""

import argparse
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "ChiefJang/visual_robust_barx"
DEST = Path("/dataset/jiyun/dataset_git/eef_pairs")


def log(msg: str) -> None:
    print(f"[fetch144] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    subs = [f.path.split("/")[-1] for f in
            api.list_repo_tree(REPO, "eef_pairs", repo_type="dataset", recursive=False)]
    targets = sorted(s for s in subs if s.startswith("56combo_144"))
    log(f"144 subsets on the hub: {targets}")

    landed = []
    for sub in targets:
        top = [f.path.split("/")[-1] for f in
               api.list_repo_tree(REPO, f"eef_pairs/{sub}", repo_type="dataset", recursive=False)]
        if "videos" not in top:
            log(f"  {sub}: no videos/ yet -- skipping")
            continue
        local = DEST / sub
        import json

        if (local / "meta/info.json").exists():
            version = json.loads((local / "meta/info.json").read_text()).get("codebase_version")
            if version == "v3.0":
                log(f"  {sub}: already local at v3.0")
                landed.append(sub)
                continue
        files = [f.path for f in
                 api.list_repo_tree(REPO, f"eef_pairs/{sub}", repo_type="dataset", recursive=True)
                 if getattr(f, "size", None) is not None and "/raw_images/" not in f.path]
        log(f"  {sub}: {len(files)} files")
        if not args.write:
            landed.append(sub)
            continue

        def one(name, sub=sub):
            for attempt in range(6):
                try:
                    path = hf_hub_download(REPO, name, repo_type="dataset")
                    out = DEST / sub / Path(name).relative_to(f"eef_pairs/{sub}")
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, out)
                    return None
                except Exception as exc:  # noqa: BLE001
                    if attempt == 5:
                        return f"{name}: {exc}"
                    time.sleep(60 if "429" in str(exc) else 3 * (attempt + 1))
            return None

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            errors = [e for e in pool.map(one, files) if e]
        if errors:
            log(f"  {sub}: {len(errors)} files failed, leaving it for the next attempt")
            continue

        import subprocess

        subprocess.run([sys.executable, "tools/add_episode_stats_count.py", str(local)], check=False)
        from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset

        # root is the PARENT; repo_id is appended to it.
        convert_dataset(repo_id=sub, root=str(DEST), push_to_hub=False)
        log(f"  {sub}: converted to v3.0")
        landed.append(sub)

    new = [s for s in landed if s != "56combo_144_bg12_closed"]
    log(f"ready: {landed}   new beyond closed: {new}")
    return 0 if new else 1


if __name__ == "__main__":
    raise SystemExit(main())
