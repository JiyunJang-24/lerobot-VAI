#!/usr/bin/env python
"""Split the barx3 front camera into a ROBOT-only and a SCENE-only stream using the shipped masks.

barx3_segmentation is the same corpus as barx_panda_ur5e_iiwa with a segmentation video added per
camera (verified: episode 0 of UR5eOmron_PnPSinkToCounter matches the prepared subset to 0.0 in
both state and action, same 526 frames). The segmentation video is effectively BINARY -- 0 for
scene, 255 for robot -- stored through a lossy codec, so the intermediate ids it contains are
ringing at the boundaries rather than classes. Threshold at 128.

Output is a drop-in replacement for dataset_git/barx3_i100_p100_u100: the same 100 episodes, the
same parquet, the same language-normalised task strings, the same meta -- only the video streams
differ. That keeps this experiment comparable with the baselines already trained on that subset,
and it means the existing training scripts need nothing but a --cameras change.

    observation.images.robot   RGB where the mask is robot, black elsewhere
    observation.images.scene   RGB where the mask is scene, black elsewhere

Masked-out pixels are black, which is -1 after SmolVLA's [-1,1] normalisation. That is a strong,
unambiguous "nothing here" value; a grey fill would sit at 0 and be confusable with real content.

    python tools/build_barx3_segsplit.py --episodes 100
"""

import argparse
import glob
import json
import shutil
from pathlib import Path

import av
import numpy as np
import pandas as pd

SRC = Path("dataset_git/barx3_segmentation")
PREPARED = Path("dataset_git/barx3_i100_p100_u100/raw")
SUBSETS = {                      # prepared name -> segmentation tree
    "iiwa": "IIWAOmron_PnPCounterToSink",
    "panda_mg": "PandaOmron_TurnOnSinkFaucet",
    "ur5e": "UR5eOmron_PnPSinkToCounter",
}
CAM = "robot0_agentview_right"
AGENT_KEY = "observation.images.robot0_agentview_right"
OUT_KEYS = ("observation.images.robot", "observation.images.scene")


def log(msg: str) -> None:
    print(f"[segsplit] {msg}", flush=True)


def decode(path: Path) -> np.ndarray:
    out = []
    with av.open(str(path)) as c:
        for frame in c.decode(video=0):
            out.append(frame.to_ndarray(format="rgb24"))
    return np.stack(out)


class Writer:
    def __init__(self, path: Path, w: int, h: int, fps: int, crf: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.c = av.open(str(path), "w")
        self.s = self.c.add_stream("libx264", rate=fps)
        self.s.width, self.s.height, self.s.pix_fmt = w, h, "yuv420p"
        self.s.options = {"crf": str(crf)}
        self.n = 0

    def add(self, frame: np.ndarray) -> None:
        self.c.mux(self.s.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        self.n += 1

    def close(self) -> None:
        self.c.mux(self.s.encode())
        self.c.close()


def build_subset(name: str, tree: str, n_eps: int, out_root: Path, crf: int) -> None:
    src = SRC / tree
    prep = PREPARED / name
    dst = out_root / name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # everything except the videos is copied verbatim -- same episodes, same tasks, same stats
    shutil.copytree(prep / "data", dst / "data")
    shutil.copytree(prep / "meta", dst / "meta")

    eps = pd.concat([pd.read_parquet(p) for p in glob.glob(str(prep / "meta/episodes/*/*.parquet"))])
    eps = eps.sort_values("episode_index")
    info = json.loads((prep / "meta" / "info.json").read_text())
    fps = info["fps"]
    h, w = info["features"][AGENT_KEY]["shape"][:2]

    writers = {k: Writer(dst / "videos" / k / "chunk-000" / "file-000.mp4", w, h, fps, crf)
               for k in OUT_KEYS}
    robot_frac = []
    for ep in range(n_eps):
        rgb = decode(src / "videos" / "chunk-000" / CAM / f"episode_{ep:06d}.mp4")
        seg = decode(src / "videos" / "chunk-000" / f"{CAM}_segmentation" / f"episode_{ep:06d}.mp4")
        want = int(eps[eps.episode_index == ep]["length"].iloc[0])
        if len(rgb) < want or len(seg) < want:
            raise RuntimeError(f"{name} ep{ep}: source has {len(rgb)}/{len(seg)} frames, need {want}")
        rgb, seg = rgb[:want], seg[:want]
        mask = seg[..., 0] > 127
        robot_frac.append(float(mask.mean()))
        for t in range(want):
            m = mask[t][..., None]
            writers["observation.images.robot"].add((rgb[t] * m).astype(np.uint8))
            writers["observation.images.scene"].add((rgb[t] * ~m).astype(np.uint8))
        if ep % 25 == 0:
            log(f"  {name} ep{ep:3d}/{n_eps}  robot={np.mean(robot_frac):.3f}")
    for wr in writers.values():
        wr.close()

    total = int(eps[eps.episode_index < n_eps]["length"].sum())
    for k, wr in writers.items():
        if wr.n != total:
            raise RuntimeError(f"{name}/{k}: wrote {wr.n} frames, expected {total}")

    # The new streams have identical per-episode frame counts, order and fps as the agentview
    # stream, so its chunk/file/timestamp bookkeeping transfers unchanged. The SOURCE image keys are
    # then removed outright rather than left in place to be filtered at run time -- the wrist flag
    # in this repo drops `observation.wrist_image` and keys containing "wrist", so a key named
    # `robot0_eye_in_hand` survives it silently. A dataset that physically contains only the two
    # streams cannot be misconfigured that way.
    drop = [AGENT_KEY, "observation.images.robot0_eye_in_hand"]
    for f in glob.glob(str(dst / "meta/episodes/*/*.parquet")):
        t = pd.read_parquet(f)
        for k in OUT_KEYS:
            for col in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
                t[f"videos/{k}/{col}"] = t[f"videos/{AGENT_KEY}/{col}"]
            for col in [c for c in t.columns if c.startswith(f"stats/{AGENT_KEY}/")]:
                t[col.replace(AGENT_KEY, k)] = t[col]
        t = t[[c for c in t.columns if not any(c.startswith(f"{pre}/{d}/") for d in drop
                                               for pre in ("videos", "stats"))]]
        t.to_parquet(f, index=False)

    feat = info["features"]
    for k in OUT_KEYS:
        feat[k] = json.loads(json.dumps(feat[AGENT_KEY]))
    for d in drop:
        feat.pop(d, None)
    info["features"] = feat
    info["total_videos"] = len(OUT_KEYS) * int(info["total_episodes"])
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    stats_path = dst / "meta" / "stats.json"
    if stats_path.exists():
        st = json.loads(stats_path.read_text())
        if AGENT_KEY in st:
            for k in OUT_KEYS:
                st[k] = json.loads(json.dumps(st[AGENT_KEY]))
        for d in drop:
            st.pop(d, None)
        stats_path.write_text(json.dumps(st, indent=4))
    log(f"{name}: {n_eps} eps, {total} frames/stream, mean robot fraction "
        f"{np.mean(robot_frac):.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--out", type=Path, default=Path("dataset_git/barx3_segsplit/raw"))
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    todo = {k: v for k, v in SUBSETS.items() if not args.only or k in args.only.split(",")}
    for name, tree in todo.items():
        log(f"building {name} from {tree}")
        build_subset(name, tree, args.episodes, args.out, args.crf)
    log(f"done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
