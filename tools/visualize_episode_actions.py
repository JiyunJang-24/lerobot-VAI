#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def _resolve_dataset_dir(dataset_root: Path, repo_id: str | None) -> Path:
    if (dataset_root / "meta" / "info.json").exists():
        return dataset_root
    if repo_id is None:
        candidates = sorted(p for p in dataset_root.iterdir() if (p / "meta" / "info.json").exists())
        if not candidates:
            raise FileNotFoundError(f"No LeRobot dataset found under {dataset_root}")
        return candidates[0]
    return dataset_root / repo_id


def _load_data_table(dataset_dir: Path):
    files = sorted((dataset_dir / "data").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir / 'data'}")
    table = pa.concat_tables([pq.read_table(path) for path in files], promote_options="default")
    return table.sort_by([("index", "ascending")])


def _image_from_cell(cell: dict) -> np.ndarray:
    image = Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
    return np.asarray(image)


def _draw_action_panel(action: np.ndarray, width: int, height: int = 150) -> np.ndarray:
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(panel, "action", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

    left = 110
    right = width - 20
    bar_w = max(12, right - left)
    row_h = max(16, (height - 34) // len(action))
    center_x = left + bar_w // 2

    cv2.line(panel, (center_x, 35), (center_x, height - 10), (160, 160, 160), 1)
    for i, value in enumerate(action):
        y = 42 + i * row_h
        v = float(np.clip(value, -1.0, 1.0))
        end_x = int(center_x + v * (bar_w // 2))
        color = (40, 120, 230) if v >= 0 else (230, 90, 40)
        cv2.putText(panel, f"a{i}: {value:+.3f}", (12, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (35, 35, 35), 1)
        cv2.rectangle(panel, (min(center_x, end_x), y - 8), (max(center_x, end_x), y + 7), color, -1)
    return panel


def _make_frame(row: dict, action_scale: float) -> np.ndarray:
    front = _image_from_cell(row["observation.image"])
    wrist = _image_from_cell(row["observation.wrist_image"])
    if front.shape != wrist.shape:
        wrist = cv2.resize(wrist, (front.shape[1], front.shape[0]), interpolation=cv2.INTER_AREA)

    top = np.concatenate([front, wrist], axis=1)
    top = cv2.cvtColor(top, cv2.COLOR_RGB2BGR)
    cv2.putText(top, "front", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(top, "wrist", (front.shape[1] + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(
        top,
        f"episode {row['episode_index']}  frame {row['frame_index']}  t={row['timestamp']:.2f}s",
        (12, top.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    action = np.asarray(row["action"], dtype=np.float32) / action_scale
    panel = _draw_action_panel(action, width=top.shape[1])
    return np.concatenate([top, panel], axis=0)


def visualize_episode(
    dataset_root: Path,
    repo_id: str | None,
    episode_index: int,
    output: Path | None,
    action_scale: float,
) -> Path:
    dataset_dir = _resolve_dataset_dir(dataset_root, repo_id)
    table = _load_data_table(dataset_dir)
    episodes = pq.read_table(sorted((dataset_dir / "meta" / "episodes").glob("*/*.parquet"))).to_pandas()

    if episode_index not in set(episodes["episode_index"].tolist()):
        raise ValueError(f"Episode {episode_index} not found in {dataset_dir}")

    ep = episodes.loc[episodes["episode_index"] == episode_index].iloc[0]
    start = int(ep["dataset_from_index"])
    end = int(ep["dataset_to_index"])
    rows = table.slice(start, end - start).to_pylist()

    if output is None:
        out_dir = dataset_root / "action_visualizations"
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"{dataset_dir.name}_episode_{episode_index:06d}_actions.mp4"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    first_frame = _make_frame(rows[0], action_scale=action_scale)
    fps = 10.0
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (first_frame.shape[1], first_frame.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output}")

    writer.write(first_frame)
    for row in rows[1:]:
        writer.write(_make_frame(row, action_scale=action_scale))
    writer.release()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one LeRobot episode with front/wrist images and actions.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--action-scale", type=float, default=1.0)
    args = parser.parse_args()

    output = visualize_episode(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        episode_index=args.episode_index,
        output=args.output,
        action_scale=args.action_scale,
    )
    print(output)


if __name__ == "__main__":
    main()
