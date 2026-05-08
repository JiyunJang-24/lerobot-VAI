#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ACTION_STAT_PAIRS = {
    "min": "max",
    "max": "min",
    "q01": "q99",
    "q10": "q90",
    "q50": "q50",
    "q90": "q10",
    "q99": "q01",
}
ACTION_STAT_QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}


def _dataset_dirs(root: Path) -> list[Path]:
    if (root / "meta" / "info.json").exists():
        return [root]
    dirs = sorted(p for p in root.iterdir() if (p / "meta" / "info.json").exists())
    if not dirs:
        raise FileNotFoundError(f"No LeRobot dataset directories found under {root}")
    return dirs


def _read_data_table(dataset_dir: Path) -> pa.Table:
    files = sorted((dataset_dir / "data").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir / 'data'}")
    tables = [pq.read_table(path) for path in files]
    return pa.concat_tables(tables, promote_options="default").sort_by([("index", "ascending")])


def _read_episode_table(dataset_dir: Path) -> pa.Table:
    files = sorted((dataset_dir / "meta" / "episodes").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files found in {dataset_dir / 'meta' / 'episodes'}")
    return pa.concat_tables([pq.read_table(path) for path in files], promote_options="default")


def _replace_column(table: pa.Table, name: str, values, type_: pa.DataType | None = None) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx < 0:
        return table
    array = pa.array(values, type=type_ or table.schema.field(name).type)
    return table.set_column(idx, table.schema.field(name), array)


def _action_stats(values: np.ndarray) -> dict:
    stats = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }
    for key, q in ACTION_STAT_QUANTILES.items():
        stats[key] = np.quantile(values, q, axis=0).tolist()
    return stats


def _gripper_moving_mask(actions: np.ndarray) -> np.ndarray:
    gripper = actions[:, -1]
    moving = np.zeros(len(gripper), dtype=bool)
    moving[1:] = ~np.isclose(gripper[1:], gripper[:-1])
    return moving


def _rewind_data_table(table: pa.Table, episodes_df, fps: int, conditional_gripper: bool) -> pa.Table:
    order: list[int] = []
    gripper_moving: list[bool] = []
    frame_indices: list[int] = []
    timestamps: list[float] = []
    episode_indices: list[int] = []
    global_indices: list[int] = []
    original_actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)

    next_index = 0
    for _, ep in episodes_df.sort_values("episode_index").iterrows():
        start = int(ep["dataset_from_index"])
        end = int(ep["dataset_to_index"])
        length = end - start
        ep_order = list(range(end - 1, start - 1, -1))
        order.extend(ep_order)
        ep_moving = _gripper_moving_mask(original_actions[start:end])
        gripper_moving.extend(ep_moving[::-1].tolist())
        frame_indices.extend(range(length))
        timestamps.extend([i / fps for i in range(length)])
        episode_indices.extend([int(ep["episode_index"])] * length)
        global_indices.extend(range(next_index, next_index + length))
        next_index += length

    rewound = table.take(pa.array(order, type=pa.int64()))
    action_type = rewound.schema.field("action").type
    actions = -np.asarray(rewound.column("action").to_pylist(), dtype=np.float32)
    if conditional_gripper:
        original_rewound_actions = np.asarray(rewound.column("action").to_pylist(), dtype=np.float32)
        moving = np.asarray(gripper_moving, dtype=bool)
        actions[~moving, -1] = original_rewound_actions[~moving, -1]
    neg_actions = actions.tolist()

    rewound = _replace_column(rewound, "action", neg_actions, action_type)
    rewound = _replace_column(rewound, "frame_index", frame_indices, pa.int64())
    rewound = _replace_column(rewound, "timestamp", timestamps, pa.float32())
    rewound = _replace_column(rewound, "episode_index", episode_indices, pa.int64())
    rewound = _replace_column(rewound, "index", global_indices, pa.int64())
    return rewound


def _transform_action_stats(stats: dict, rewound_data: pa.Table | None = None) -> dict:
    if "action" not in stats:
        return stats
    if rewound_data is not None:
        stats = dict(stats)
        stats["action"] = _action_stats(np.asarray(rewound_data.column("action").to_pylist(), dtype=np.float32))
        return stats
    old = stats["action"]
    new = dict(old)
    for dst, src in ACTION_STAT_PAIRS.items():
        if src in old:
            new[dst] = (-np.asarray(old[src], dtype=np.float64)).tolist()
    if "mean" in old:
        new["mean"] = (-np.asarray(old["mean"], dtype=np.float64)).tolist()
    stats = dict(stats)
    stats["action"] = new
    return stats


def _transform_episode_action_stats(episodes_df, rewound_data: pa.Table | None = None):
    out = episodes_df.copy()
    if rewound_data is not None:
        actions = np.asarray(rewound_data.column("action").to_pylist(), dtype=np.float32)
        for i, row in out.iterrows():
            start = int(row["dataset_from_index"])
            end = int(row["dataset_to_index"])
            stats = _action_stats(actions[start:end])
            for stat_key, stat_value in stats.items():
                col = f"stats/action/{stat_key}"
                if col in out.columns:
                    out.at[i, col] = stat_value
        return out

    for dst, src in ACTION_STAT_PAIRS.items():
        dst_col = f"stats/action/{dst}"
        src_col = f"stats/action/{src}"
        if dst_col in out.columns and src_col in episodes_df.columns:
            out[dst_col] = episodes_df[src_col].apply(lambda x: (-np.asarray(x, dtype=np.float64)).tolist())
    mean_col = "stats/action/mean"
    if mean_col in out.columns:
        out[mean_col] = episodes_df[mean_col].apply(lambda x: (-np.asarray(x, dtype=np.float64)).tolist())
    return out


def _write_table_chunks(table: pa.Table, output_data_dir: Path, rows_per_file: int) -> None:
    output_data_dir.mkdir(parents=True, exist_ok=True)
    for file_index, start in enumerate(range(0, table.num_rows, rows_per_file)):
        out = output_data_dir / "chunk-000" / f"file-{file_index:03d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.slice(start, rows_per_file), out)


def _write_episode_table(episodes_df, schema: pa.Schema, output_dir: Path) -> None:
    out = output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(episodes_df, schema=schema, preserve_index=False)
    pq.write_table(table, out)


def _rewind_one_dataset(
    input_dir: Path,
    output_dir: Path,
    rows_per_file: int,
    conditional_gripper: bool,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)

    for name in ["info.json", "tasks.parquet"]:
        shutil.copy2(input_dir / "meta" / name, output_dir / "meta" / name)

    with open(input_dir / "meta" / "info.json") as f:
        info = json.load(f)
    fps = int(info["fps"])

    data_table = _read_data_table(input_dir)
    episode_table = _read_episode_table(input_dir)
    episodes_df = episode_table.to_pandas().sort_values("episode_index").reset_index(drop=True)

    rewound_data = _rewind_data_table(
        data_table,
        episodes_df,
        fps=fps,
        conditional_gripper=conditional_gripper,
    )
    _write_table_chunks(rewound_data, output_dir / "data", rows_per_file=rows_per_file)

    stats_path = input_dir / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        with open(output_dir / "meta" / "stats.json", "w") as f:
            json.dump(_transform_action_stats(stats, rewound_data=rewound_data), f)

    rewound_episodes = _transform_episode_action_stats(episodes_df)
    cursor = 0
    for i, row in rewound_episodes.iterrows():
        length = int(row["length"])
        rewound_episodes.at[i, "dataset_from_index"] = cursor
        rewound_episodes.at[i, "dataset_to_index"] = cursor + length
        rewound_episodes.at[i, "data/chunk_index"] = 0
        rewound_episodes.at[i, "data/file_index"] = cursor // rows_per_file
        if "meta/episodes/chunk_index" in rewound_episodes.columns:
            rewound_episodes.at[i, "meta/episodes/chunk_index"] = 0
        if "meta/episodes/file_index" in rewound_episodes.columns:
            rewound_episodes.at[i, "meta/episodes/file_index"] = 0
        cursor += length

    rewound_episodes = _transform_episode_action_stats(rewound_episodes, rewound_data=rewound_data)
    _write_episode_table(rewound_episodes, episode_table.schema, output_dir)


def create_rewind_dataset(input_root: Path, output_root: Path, rows_per_file: int, conditional_gripper: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for input_dir in _dataset_dirs(input_root):
        output_dir = output_root / input_dir.name if input_dir != input_root else output_root
        print(f"rewind {input_dir} -> {output_dir}")
        _rewind_one_dataset(
            input_dir,
            output_dir,
            rows_per_file=rows_per_file,
            conditional_gripper=conditional_gripper,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a rewind copy of a LeRobot dataset root.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rows-per-file", type=int, default=1000)
    parser.add_argument(
        "--conditional-gripper",
        action="store_true",
        help="Only negate the final gripper action dimension when the gripper command changes.",
    )
    args = parser.parse_args()
    create_rewind_dataset(
        args.input_root,
        args.output_root,
        rows_per_file=args.rows_per_file,
        conditional_gripper=args.conditional_gripper,
    )
    print(args.output_root)


if __name__ == "__main__":
    main()
