#!/usr/bin/env python
"""Auxiliary contrastive corpus: ROBOT-SEGMENTED front renders, built from the regenerated dataset.

Each view is the robot CUT OUT of its own render. Previously the contrastive loss compared whole
frames of the same moment rendered with different arms, which left "same kitchen" available as a
shortcut -- measured on real frames, stock SigLIP scored +0.369 on same-scene/different-pose pairs,
so most of that similarity was the background. On robot-only crops there is no background to match,
so the objective is purely "same configuration, different arm".

Output keys are observation.image.<Embodiment>, matching VISUAL_ROBUST_FRONT_PREFIXES's
"observation.image." so the loader needs no change.

Written in the SOURCE's own v2.1 layout and converted to v3.0 afterwards. It does NOT borrow the
v3.0 metadata of visual_robust_new_barx: that export has the same total frame count but different
episode boundaries -- 48 of 108 episodes differ in length -- so reusing it would misalign every
frame past the first divergence while looking perfectly healthy.

FRONT CAMERA ONLY. The wrist keys are dropped by not writing them, not by a runtime flag, because
--dataset.use_wrist_cam matches neither `robot0_eye_in_hand` nor anything else here.

    python tools/build_vr_robot_seg_v21.py
"""

import argparse
import json
import shutil
from pathlib import Path

import av
import numpy as np

SRC = Path("dataset_git/visual_robust_seg_regen")
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


def write(path: Path, frames: np.ndarray, fps: int, crf: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.height, s.width, s.pix_fmt = frames.shape[1], frames.shape[2], "yuv420p"
        s.options = {"crf": str(crf)}
        for f in frames:
            c.mux(s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")))
            n += 1
        c.mux(s.encode())          # drain the encoder -- skipping this is what truncated the source
    return n


def build(tree: str, out_root: Path, crf: int) -> None:
    src = SRC / tree / "lerobot"
    dst = out_root / tree / "lerobot"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    shutil.copytree(src / "data", dst / "data")
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    for f in ("episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl", "stats.json"):
        if (src / "meta" / f).exists():
            shutil.copy(src / "meta" / f, dst / "meta" / f)

    info = json.loads((src / "meta" / "info.json").read_text())
    fps = info["fps"]
    lengths = {}
    with open(src / "meta" / "episodes.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            lengths[int(d["episode_index"])] = int(d["length"])

    out_keys = [f"observation.image.{e}" for e in EMBS]
    fracs = {e: [] for e in EMBS}
    for ep in sorted(lengths):
        want = lengths[ep]
        for e in EMBS:
            vd = src / "videos" / "chunk-000"
            rgb = decode(vd / f"{e}.{CAM}" / f"episode_{ep:06d}.mp4")
            seg = decode(vd / f"{e}.{CAM}_segmentation" / f"episode_{ep:06d}.mp4")
            if len(rgb) != want or len(seg) != want:
                raise RuntimeError(f"{tree} ep{ep} {e}: rgb={len(rgb)} seg={len(seg)} meta={want}")
            m = seg[..., 0] > 127
            fracs[e].append(float(m.mean()))
            got = write(dst / "videos" / "chunk-000" / f"observation.image.{e}"
                        / f"episode_{ep:06d}.mp4", (rgb * m[..., None]).astype(np.uint8), fps, crf)
            if got != want:                 # the check the source pipeline was missing
                raise RuntimeError(f"{tree} ep{ep} {e}: wrote {got} frames, expected {want}")
        if ep % 25 == 0:
            log(f"  {tree} ep{ep:3d}/{len(lengths)}")

    feats = {k: v for k, v in info["features"].items() if "images" not in k}
    tmpl = info["features"][f"observation.images.{EMBS[0]}.{CAM}"]
    for k in out_keys:
        feats[k] = json.loads(json.dumps(tmpl))
    info["features"] = feats
    info["total_videos"] = len(out_keys) * len(lengths)
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    log(f"{tree}: {len(lengths)} eps, robot fraction "
        + ", ".join(f"{e}={np.mean(fracs[e]):.3f}" for e in EMBS))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("dataset_git/vr_seg_robot_v21"))
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    for tree in TREES:
        if args.only and tree not in args.only.split(","):
            continue
        log(f"building {tree}")
        build(tree, args.out, args.crf)
    log(f"done -> {args.out}   (now convert each tree to v3.0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
