#!/usr/bin/env python
"""Cache access and batch sampling for Experiment 1 (selective visual alignment).

One underlying state is (scene, frame). One sample is (state, embodiment). The cache built by
tools/exp1_build_cache.py stores a whole scene per npz, which is convenient rather than awkward:
the hard negatives this experiment needs are OTHER STATES IN THE SAME SCENE, so a batch is drawn
from one scene and nothing is gained by holding many scenes resident at once.

Sampling contract, identical for every method so the comparison is clean:

    states_per_batch states from one scene, embodiments_per_state renders of each.
    Rows sharing a state are positives. Rows of different states are negatives, and because they
    come from the same scene they differ only in where the arm is -- background cannot win.

Rows are filtered on `reachable` and `pos_err_m`: an unreachable frame is a render of the arm
parked somewhere else entirely, and treating it as the same underlying state would be a labelling
error, not a hard positive.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Exp1Config:
    """Everything the dataset-scaling axes need, none of it hard-coded at the call site."""

    split: Path = Path("outputs/exp1/split.json")
    states_per_batch: int = 8
    embodiments_per_state: int = 6
    max_train_scenes: int = 0        # 0 = all scenes in the split
    max_train_embodiments: int = 0   # 0 = all embodiments in the split
    states_per_scene: int = 0        # 0 = every cached state
    total_state_budget: int = 0      # 0 = unbounded; caps scenes x states for fixed-budget runs
    max_pos_err_m: float = 0.01
    resident_scenes: int = 6         # LRU size, ~570 MB per scene
    seed: int = 0
    extra: dict = field(default_factory=dict)


class SceneCache:
    """LRU over per-scene npz files. Each holds images/masks/eef for every embodiment."""

    def __init__(self, root: Path, capacity: int = 6):
        self.root = Path(root)
        self.capacity = capacity
        self._store: OrderedDict[int, dict] = OrderedDict()
        self._means: dict[int, np.ndarray] = {}
        self.embodiments: list[str] = json.loads((self.root / "embodiments.json").read_text())
        self.index = {e: i for i, e in enumerate(self.embodiments)}

    def __call__(self, scene: int) -> dict:
        if scene in self._store:
            self._store.move_to_end(scene)
            return self._store[scene]
        with np.load(self.root / f"ep{scene:03d}.npz") as z:
            blob = {k: z[k] for k in z.files}
        self._store[scene] = blob
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)
        return blob

    def image(self, scene: int, t: int, emb: str) -> np.ndarray:
        return self(scene)["images"][t, self.index[emb]]

    def mask(self, scene: int, t: int, emb: str) -> np.ndarray:
        blob = self(scene)
        packed = blob["masks"][t, self.index[emb]]
        width = blob["images"].shape[3]
        return np.unpackbits(packed, axis=-1)[..., :width].astype(bool)

    def eef(self, scene: int, t: int, emb: str) -> np.ndarray:
        return self(scene)["eef"][t, self.index[emb]]

    def scene_mean_xyz(self, scene: int) -> np.ndarray:
        """Mean EEF position over the usable renders of this scene.

        EEF is stored in world coordinates and each kitchen sits somewhere different in the
        world, so raw xyz carries roughly a metre of pure scene identity -- a probe asked to
        regress it would score well by recognising the kitchen. Subtracting the scene mean turns
        the target into "where in this workspace is the arm", which is the quantity the
        experiment is actually about.
        """
        if scene not in self._means:
            blob = self(scene)
            ok = (blob["reachable"] > 0.5) & (blob["pos_err"] < 0.01)
            xyz = blob["eef"][..., :3][ok]
            self._means[scene] = xyz.mean(0) if len(xyz) else np.zeros(3, dtype=np.float32)
        return self._means[scene]

    def eef_rel(self, scene: int, t: int, emb: str) -> np.ndarray:
        e = self.eef(scene, t, emb).copy()
        e[:3] -= self.scene_mean_xyz(scene)
        return e


class Exp1Dataset:
    """The split, the usable (scene, t, embodiment) index, and a batch sampler over it."""

    def __init__(self, cfg: Exp1Config, group: str = "train"):
        self.cfg = cfg
        self.split = json.loads(Path(cfg.split).read_text())
        self.cache = SceneCache(Path(self.split["cache"]), cfg.resident_scenes)

        if group == "train":
            scenes = list(self.split["train_scenes"])
            embs = list(self.split["train_embodiments"])
        else:
            spec = self.split["groups"][group]
            scenes, embs = list(spec["scenes"]), list(spec["embodiments"])

        rng = np.random.default_rng(cfg.seed)
        if cfg.max_train_scenes and group == "train":
            scenes = sorted(rng.choice(scenes, min(cfg.max_train_scenes, len(scenes)), replace=False).tolist())
        if cfg.max_train_embodiments and group == "train":
            embs = sorted(rng.choice(embs, min(cfg.max_train_embodiments, len(embs)), replace=False).tolist())

        self.scenes, self.embodiments = scenes, embs
        self.group = group
        self._usable: dict[int, dict[str, np.ndarray]] = {}
        self._states: dict[int, np.ndarray] = {}
        self._build_index(rng)

    def _build_index(self, rng: np.random.Generator) -> None:
        """Per scene: which timesteps are usable for which embodiments."""
        budget = self.cfg.total_state_budget
        per_scene = self.cfg.states_per_scene
        if budget and not per_scene:
            per_scene = max(1, budget // max(1, len(self.scenes)))

        for scene in self.scenes:
            with np.load(Path(self.split["cache"]) / f"ep{scene:03d}.npz") as z:
                reach, perr = z["reachable"], z["pos_err"]
            ok = (reach > 0.5) & (perr < self.cfg.max_pos_err_m)
            idx = {e: np.flatnonzero(ok[:, self.cache.index[e]]) for e in self.embodiments}
            # a state is only usable if at least two embodiments render it -- one render has no pair
            counts = sum(ok[:, self.cache.index[e]].astype(int) for e in self.embodiments)
            states = np.flatnonzero(counts >= 2)
            if per_scene and len(states) > per_scene:
                states = np.sort(rng.choice(states, per_scene, replace=False))
            self._usable[scene] = idx
            self._states[scene] = states

    @property
    def n_states(self) -> int:
        return int(sum(len(v) for v in self._states.values()))

    def embodiments_at(self, scene: int, t: int) -> list[str]:
        return [e for e in self.embodiments if t in set(self._usable[scene][e].tolist())]

    def sample_batch(self, rng: np.random.Generator) -> dict:
        """One scene, S states, V embodiments each. Returns arrays, not tensors."""
        cfg = self.cfg
        for _ in range(32):
            scene = int(rng.choice(self.scenes))
            states = self._states[scene]
            if len(states) >= cfg.states_per_batch:
                break
        else:
            raise RuntimeError("no scene has enough usable states")
        chosen = rng.choice(states, cfg.states_per_batch, replace=False)

        blob = self.cache(scene)
        images, masks, eefs, labels, emb_ids = [], [], [], [], []
        width = blob["images"].shape[3]
        for label, t in enumerate(chosen):
            avail = [e for e in self.embodiments
                     if blob["reachable"][t, self.cache.index[e]] > 0.5
                     and blob["pos_err"][t, self.cache.index[e]] < cfg.max_pos_err_m]
            if len(avail) < 2:
                continue
            take = rng.choice(avail, min(cfg.embodiments_per_state, len(avail)), replace=False)
            for e in take:
                j = self.cache.index[e]
                images.append(blob["images"][t, j])
                masks.append(np.unpackbits(blob["masks"][t, j], axis=-1)[..., :width])
                rel = blob["eef"][t, j].copy()
                rel[:3] -= self.cache.scene_mean_xyz(scene)
                eefs.append(rel)
                labels.append(label)
                emb_ids.append(j)
        return {
            "images": np.stack(images),
            "masks": np.stack(masks).astype(bool),
            "eef": np.stack(eefs).astype(np.float32),
            "labels": np.asarray(labels, dtype=np.int64),
            "embodiment": np.asarray(emb_ids, dtype=np.int64),
            "scene": scene,
        }

    def enumerate_group(self, states_per_scene: int, rng: np.random.Generator) -> list[tuple]:
        """Flat (scene, t, embodiment) list for evaluation -- deterministic given the seed."""
        out = []
        for scene in self.scenes:
            states = self._states[scene]
            if states_per_scene and len(states) > states_per_scene:
                states = np.sort(rng.choice(states, states_per_scene, replace=False))
            blob = self.cache(scene)
            for t in states:
                for e in self.embodiments:
                    j = self.cache.index[e]
                    if (blob["reachable"][t, j] > 0.5
                            and blob["pos_err"][t, j] < self.cfg.max_pos_err_m):
                        out.append((scene, int(t), e))
        return out
