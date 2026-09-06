#!/usr/bin/env python
"""Watch the account for a dataset or subset that did not exist when the snapshot was taken.

Scans repositories and their eef_pairs/ subsets, so a new subset inside an existing repo is caught
as well as a whole new repo -- the last two arrivals came in the second form.

Also reports whether each new subset's upload has FINISHED, by counting video files per declared
stream. A subset once shipped with data/ and meta/ present and videos/ missing: info.json declared
231,678 frames and none could be decoded, and the listing looked complete.

    python tools/watch_new_datasets.py --once        # print the current state and exit
    python tools/watch_new_datasets.py --interval 600
"""

import argparse
import json
import time
from pathlib import Path

AUTHOR = "ChiefJang"
STATE = Path("/home/gpuuser/jiyun/lerobot-VAI/outputs/logs/known_datasets.json")


def log(msg: str) -> None:
    print(f"[watch] {time.strftime('%m-%d %H:%M:%S')} {msg}", flush=True)


def snapshot(api):
    out = {}
    for ds in api.list_datasets(author=AUTHOR):
        subsets = []
        try:
            subsets = sorted(x.path.split("/")[-1] for x in
                             api.list_repo_tree(ds.id, "eef_pairs", repo_type="dataset",
                                                recursive=False))
        except Exception:  # noqa: BLE001 - repo may simply have no eef_pairs/
            pass
        out[ds.id] = {"modified": str(ds.lastModified), "subsets": subsets}
    return out


def upload_complete(api, repo: str, subset: str) -> str:
    from huggingface_hub import hf_hub_download

    prefix = f"eef_pairs/{subset}"
    try:
        info = json.loads(Path(hf_hub_download(repo, f"{prefix}/meta/info.json",
                                               repo_type="dataset")).read_text())
    except Exception:  # noqa: BLE001
        return "no info.json yet"
    episodes = int(info.get("total_episodes", 0))
    streams = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    files = [f.path for f in api.list_repo_tree(repo, prefix, repo_type="dataset", recursive=True)
             if getattr(f, "size", None) is not None]
    parts = []
    for stream in streams:
        n = sum(1 for f in files if f"/{stream}/" in f and "/videos/" in f)
        parts.append(f"{stream.split('.')[-1]} {n}/{episodes}")
    parquet = sum(1 for f in files if "/data/" in f and f.endswith(".parquet"))
    parts.append(f"parquet {parquet}/{episodes}")
    done = all(int(p.split()[-1].split("/")[0]) >= episodes for p in parts)
    return ("COMPLETE  " if done else "uploading ") + ", ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--hours", type=float, default=48)
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    known = json.loads(STATE.read_text()) if STATE.exists() else {}
    if not known:
        known = snapshot(api)
        STATE.write_text(json.dumps(known, indent=2))
        log(f"baseline: {len(known)} datasets, "
            f"{sum(len(v['subsets']) for v in known.values())} eef_pairs subsets")
        if args.once:
            return 0

    deadline = time.time() + args.hours * 3600
    while True:
        current = snapshot(api)
        for repo, info in current.items():
            was = known.get(repo)
            if was is None:
                log(f"NEW DATASET {repo}")
            else:
                for sub in set(info["subsets"]) - set(was["subsets"]):
                    log(f"NEW SUBSET  {repo}  eef_pairs/{sub}")
                    log(f"            {upload_complete(api, repo, sub)}")
        known = current
        STATE.write_text(json.dumps(known, indent=2))
        if args.once or time.time() > deadline:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
