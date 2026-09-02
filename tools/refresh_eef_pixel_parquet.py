#!/usr/bin/env python
"""Re-download the parquet of the trees that were re-exported with an EEF-pixel column.

Only the parquet: the videos are unchanged and re-pulling them would cost hours. This overwrites
data/ in place, which is safe here precisely because it is only data/ -- the v3.0 conversion that
these trees already went through lives in meta/, and section 6's "never re-download over a converted
tree" warning is about restoring v2.1 meta, not about refreshing columns.

Verifies the new column is actually present before touching anything on disk, so a re-export that
has not landed yet fails loudly instead of silently overwriting good data with the old schema.

    python tools/refresh_eef_pixel_parquet.py            # check only
    python tools/refresh_eef_pixel_parquet.py --write    # download
"""

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEST = Path("/dataset/jiyun/dataset_git")
TARGETS = [
    ("ChiefJang/visual_robust_robocasa_x",
     "new_barx/UR5eOmron_PnPSinkToCounter/lerobot",
     DEST / "visual_robust_new_barx_ur5e/new_barx/UR5eOmron_PnPSinkToCounter/lerobot"),
    ("ChiefJang/barx_panda_ur5e_iiwa", "IIWAOmron/pretrain/PnPSinkToCounter/lerobot",
     DEST / "barx_panda_ur5e_iiwa/IIWAOmron/pretrain/PnPSinkToCounter/lerobot"),
    ("ChiefJang/barx_panda_ur5e_iiwa", "PandaOmron/pretrain/PnPSinkToCounter/lerobot",
     DEST / "barx_panda_ur5e_iiwa/PandaOmron/pretrain/PnPSinkToCounter/lerobot"),
    ("ChiefJang/barx_panda_ur5e_iiwa", "UR5eOmron/pretrain/PnPSinkToCounter/lerobot",
     DEST / "barx_panda_ur5e_iiwa/UR5eOmron/pretrain/PnPSinkToCounter/lerobot"),
]
PIXEL_HINTS = ("pixel", "eef_pixel", "uv")


def log(msg: str) -> None:
    print(f"[refresh] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="actually download; default is a dry check")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    ok = True
    for repo, prefix, dest in TARGETS:
        try:
            files = [f for f in list_repo_files(repo, repo_type="dataset")
                     if f.startswith(prefix + "/data/") and f.endswith(".parquet")]
        except Exception as exc:  # noqa: BLE001
            log(f"{repo}:{prefix} -- cannot list ({type(exc).__name__}: {exc})")
            ok = False
            continue
        if not files:
            log(f"{repo}:{prefix} -- no parquet under data/, NOT ready")
            ok = False
            continue

        probe = hf_hub_download(repo, files[0], repo_type="dataset")
        import pyarrow.parquet as pq

        columns = pq.ParquetFile(probe).schema.names
        hit = [c for c in columns if any(h in c.lower() for h in PIXEL_HINTS)]
        log(f"{repo}:{prefix}  {len(files)} parquet, pixel-ish columns: {hit or 'NONE'}")
        if not hit:
            log("   -> the re-export has not landed yet; skipping")
            ok = False
            continue
        if not args.write:
            continue

        def fetch(name, repo=repo, prefix=prefix, dest=dest):
            for attempt in range(4):
                try:
                    local = hf_hub_download(repo, name, repo_type="dataset")
                    out = dest / Path(name).relative_to(prefix)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(local, out)
                    return None
                except Exception as exc:  # noqa: BLE001
                    if attempt == 3:
                        return f"{name}: {type(exc).__name__}: {exc}"
            return None

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            errors = [e for e in pool.map(fetch, files) if e]
        log(f"   wrote {len(files) - len(errors)}/{len(files)} parquet into {dest}")
        for e in errors[:5]:
            log(f"   FAILED {e}")
        ok = ok and not errors
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
