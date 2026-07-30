#!/usr/bin/env python3
"""Visualize visual-robust wrist images with gripper opening labels."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_REPO = (
    "visual_robust_ex3_multi_embodiment_goal_30_01_0.0_0.0/"
    "v-1.000-1.000_num5"
)
WRIST_PREFIX = "observation.wrist_image."


def image_from_cell(cell: dict, dataset_root: Path) -> Image.Image:
    if cell.get("bytes") is not None:
        return Image.open(BytesIO(cell["bytes"])).convert("RGB")
    if cell.get("path") is not None:
        path = Path(cell["path"])
        if not path.is_absolute():
            path = dataset_root / path
        return Image.open(path).convert("RGB")
    raise ValueError(f"Unsupported image cell: {cell}")


def gripper_width(state: np.ndarray) -> float:
    return abs(float(state[0]) - float(state[1]))


def normalized_opening(width: float, width_min: float, width_max: float) -> float:
    if width_max <= width_min:
        raise ValueError("--width-max must be greater than --width-min")
    return float(np.clip((width - width_min) / (width_max - width_min), 0.0, 1.0))


def draw_label(image: Image.Image, lines: list[str]) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    line_height = 14
    pad = 5
    box_height = pad * 2 + line_height * len(lines)
    draw.rectangle((0, 0, image.width, box_height), fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((pad, pad + i * line_height), line, fill=(255, 255, 255), font=font)
    return image


def make_grid(images: list[Image.Image], cols: int) -> Image.Image:
    if not images:
        raise ValueError("No images to render")
    width, height = images[0].size
    rows = int(np.ceil(len(images) / cols))
    canvas = Image.new("RGB", (cols * width, rows * height), (255, 255, 255))
    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        canvas.paste(image, (col * width, row * height))
    return canvas


def strip_prefix(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.startswith(prefix) else text


def quantile_indices(widths: np.ndarray, count: int) -> list[int]:
    if count >= len(widths):
        return list(range(len(widths)))
    quantiles = np.linspace(0.0, 1.0, count)
    targets = np.quantile(widths, quantiles)
    indices = []
    for target in targets:
        index = int(np.argmin(np.abs(widths - target)))
        if index not in indices:
            indices.append(index)
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset_git/visual_robust_ex03"),
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--width-min", type=float, default=0.0)
    parser.add_argument("--width-max", type=float, default=0.08)
    parser.add_argument("--num-rows", type=int, default=8)
    parser.add_argument("--max-views", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/visual_robust_gripper_opening/wrist_opening_grid.png"),
    )
    args = parser.parse_args()

    repo_root = args.dataset_root / args.repo_id
    parquet_files = sorted((repo_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {repo_root / 'data'}")

    frames = []
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        df["_parquet_file"] = str(parquet_file)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    wrist_keys = sorted(key for key in df.columns if key.startswith(WRIST_PREFIX))[: args.max_views]
    if not wrist_keys:
        raise ValueError("No auxiliary wrist image keys found")

    widths = np.asarray([gripper_width(state) for state in df["observation.state"]], dtype=np.float32)
    selected = quantile_indices(widths, args.num_rows)

    tiles = []
    for row_rank, row_index in enumerate(selected):
        row = df.iloc[row_index]
        width = float(widths[row_index])
        opening = normalized_opening(width, args.width_min, args.width_max)
        frame_index = int(row["frame_index"])
        episode_index = int(row["episode_index"])
        for wrist_key in wrist_keys:
            image = image_from_cell(row[wrist_key], repo_root)
            image = draw_label(
                image,
                [
                    strip_prefix(wrist_key, WRIST_PREFIX),
                    f"row={row_rank} ep={episode_index} frame={frame_index}",
                    f"width={width:.4f} opening={opening:.3f}",
                ],
            )
            tiles.append(image)

    grid = make_grid(tiles, cols=len(wrist_keys))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)

    summary_path = args.output.with_suffix(".csv")
    summary = pd.DataFrame(
        [
            {
                "rank": rank,
                "dataset_row": index,
                "episode_index": int(df.iloc[index]["episode_index"]),
                "frame_index": int(df.iloc[index]["frame_index"]),
                "width": float(widths[index]),
                "normalized_opening": normalized_opening(float(widths[index]), args.width_min, args.width_max),
            }
            for rank, index in enumerate(selected)
        ]
    )
    summary.to_csv(summary_path, index=False)
    print(f"Saved grid: {args.output}")
    print(f"Saved summary: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
