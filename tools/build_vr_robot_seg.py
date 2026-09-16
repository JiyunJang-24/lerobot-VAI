#!/usr/bin/env python
"""Build the auxiliary contrastive corpus: ROBOT-SEGMENTED front renders, one per embodiment.

visual_robust_new_segmentation renders each frame from three embodiments and ships a segmentation
video for every (embodiment, camera). This turns that into the corpus the visual-robust contrastive
loss consumes, with one change from the previous version: each view is the robot CUT OUT of its
render rather than the whole frame.

That is the point of the experiment. Previously the loss pulled together whole frames of the same
moment rendered with different arms, so "same scene" was available as a shortcut and a large part
of the similarity came from the kitchen (measured earlier on real frames: stock SigLIP scored +0.369
on same-scene/different-pose pairs). On robot-only crops there is no background left to match on,
so the objective is purely "same configuration, different arm".

Output keys are observation.image.<Embodiment>, matching VISUAL_ROBUST_FRONT_PREFIXES's
"observation.image." so the existing loader and view-selection code need no change.

FRONT CAMERA ONLY -- the wrist keys are not carried over. They are dropped by not writing them,
not by a runtime flag, because --dataset.use_wrist_cam does not match a key named
`robot0_eye_in_hand` and would leave them in.

    python tools/build_vr_robot_seg.py
"""

import argparse
import glob
import json
import shutil
from pathlib import Path

import av
import numpy as np
import pandas as pd

NEW = Path("dataset_git/visual_robust_seg_regen")
# NO external skeleton. The regenerated corpus keeps the same TOTAL frame count as the older
# visual_robust_new_barx export but its episode boundaries differ -- 48 of 108 episodes have a
# different length -- so borrowing that tree's v3.0 metadata would silently misalign every frame
# after the first divergence. The v3.0 tree is written from the regenerated dataset's own meta.
TREES = ["IIWAOmron_PnPCounterToSink", "PandaOmron_TurnOnSinkFaucet", "UR5eOmron_PnPSinkToCounter"]
EMBS = ["IIWAOmron", "PandaOmron", "UR5eOmron"]
CAM = "robot0_agentview_right"


def log(m: str) -> None:
    print(f"[vrseg] {m}", flush=True)


def decode(p: Path) -> np.ndarray:
    out = []
    with av.open(str(p)) as c:
        for f in c.decode(video=0):
            out.append(f.to_ndarray(format="rgb24"))
    return np.stack(out)


class Writer:
    def __init__(self, path: Path, w: int, h: int, fps: int, crf: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.c = av.open(str(path), "w")
        self.s = self.c.add_stream("libx264", rate=fps)
        self.s.width, self.s.height, self.s.pix_fmt = w, h, "yuv420p"
        self.s.options = {"crf": str(crf)}
        self.n = 0

    def add(self, f):
        self.c.mux(self.s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")))
        self.n += 1

    def close(self):
        self.c.mux(self.s.encode())
        self.c.close()


def build(tree: str, out_root: Path, crf: int) -> None:
    src = NEW / tree / "lerobot"
    dst = out_root / tree / "lerobot"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    info = json.loads((src / "meta" / "info.json").read_text())
    fps = info["fps"]
    n_eps = int(info["total_episodes"])
    template = f"observation.images.{EMBS[0]}.{CAM}"
    h, w = info["features"][template]["shape"][:2]
    lengths = {}
    with open(src / "meta" / "episodes.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            lengths[int(d["episode_index"])] = int(d["length"])

    out_keys = [f"observation.image.{e}" for e in EMBS]
    writers = {k: Writer(dst / "videos" / k / "chunk-000" / "file-000.mp4", w, h, fps, crf)
               for k in out_keys}
    fracs = {e: [] for e in EMBS}
    for ep in range(n_eps):
        want = int(eps[eps.episode_index == ep]["length"].iloc[0])
        for e in EMBS:
            # the video directories drop the "observation.images." prefix the feature keys carry
            vdir = src / "videos" / "chunk-000"
            rgb = decode(vdir / f"{e}.{CAM}" / f"episode_{ep:06d}.mp4")
            seg = decode(vdir / f"{e}.{CAM}_segmentation" / f"episode_{ep:06d}.mp4")
            if len(rgb) < want or len(seg) < want:
                raise RuntimeError(f"{tree} ep{ep} {e}: {len(rgb)}/{len(seg)} frames, need {want}")
            m = seg[:want, ..., 0] > 127
            fracs[e].append(float(m.mean()))
            wr = writers[f"observation.image.{e}"]
            for t in range(want):
                wr.add((rgb[t] * m[t][..., None]).astype(np.uint8))
        if ep % 25 == 0:
            log(f"  {tree} ep{ep:3d}/{n_eps}")
    for wr in writers.values():
        wr.close()

    total = int(eps["length"].sum())
    for k, wr in writers.items():
        if wr.n != total:
            raise RuntimeError(f"{tree}/{k}: wrote {wr.n}, expected {total}")

    # Same per-episode frame counts, order and fps as the skeleton's own front streams, so its
    # chunk/file/timestamp bookkeeping carries over unchanged. Everything else is removed: the
    # corpus must contain FRONT ROBOT-SEGMENTED views only.
    keep_video = set(out_keys)
    for f in glob.glob(str(dst / "meta/episodes/*/*.parquet")):
        t = pd.read_parquet(f)
        for k in out_keys:
            for col in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
                t[f"videos/{k}/{col}"] = t[f"videos/{template}/{col}"]
        drop = [c for c in t.columns if c.startswith("videos/")
                and c.split("/")[1] not in keep_video]
        t = t[[c for c in t.columns if c not in drop]]
        t.to_parquet(f, index=False)

    feat = info["features"]
    for k in out_keys:
        feat[k] = json.loads(json.dumps(feat[template]))
    for k in [k for k in list(feat) if k.startswith("observation.") and "image" in k
              and k not in keep_video]:
        feat.pop(k)
    info["features"] = feat
    info["total_videos"] = len(out_keys) * n_eps
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    sp = dst / "meta" / "stats.json"
    if sp.exists():
        st = json.loads(sp.read_text())
        for k in out_keys:
            if template in st:
                st[k] = json.loads(json.dumps(st[template]))
        for k in [k for k in list(st) if "image" in k and k not in keep_video]:
            st.pop(k)
        sp.write_text(json.dumps(st, indent=4))

    log(f"{tree}: {n_eps} eps, {total} frames/view, robot fraction "
        + ", ".join(f"{e}={np.mean(fracs[e]):.3f}" for e in EMBS))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("dataset_git/vr_new_seg_robot"))
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    for tree in TREES:
        if args.only and tree not in args.only.split(","):
            continue
        log(f"building {tree}")
        build(tree, args.out, args.crf)
    log(f"done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
