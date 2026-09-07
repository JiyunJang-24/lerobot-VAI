#!/usr/bin/env python
"""Write the missing normalizer/unnormalizer files into checkpoints that lack them.

`policy.save_pretrained()` writes config.json and model.safetensors and nothing else. The
evaluation path also needs the processor pipelines:

    policy_preprocessor.json
    policy_preprocessor_step_5_normalizer_processor.safetensors
    policy_postprocessor.json
    policy_postprocessor_step_0_unnormalizer_processor.safetensors

Without them a checkpoint whose config says STATE: MEAN_STD and ACTION: MEAN_STD loads fine and
then emits actions on the wrong scale -- it fails silently, which is why this went unnoticed.

The statistics must be the ones TRAINING used, not a fresh aggregate, or the policy is
unnormalised with different numbers than it was normalised with. Both trainers passed
`dataset._datasets[0].meta.stats`, i.e. the FIRST tree only, so that is what is rebuilt here.
(That is a flaw in its own right for a multi-robot corpus -- the normalisation is fit to one
robot -- but it is consistent across every run, so reproducing it is what keeps the checkpoints
correct and the comparisons intact.)

    python tools/repair_checkpoint_processors.py --runs outputs/dp_embodiment/*/
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

NEEDED = ("policy_preprocessor.json", "policy_postprocessor.json")

# which corpus each trainer's first tree is, so the stats can be rebuilt without a full load
FIRST_TREE = {
    "dp": (Path("/dataset/jiyun/dataset_git/barx_panda_ur5e_iiwa"),
           "IIWAOmron/pretrain/PnPSinkToCounter/lerobot"),
    "smolvla": (Path("/dataset/jiyun/dataset_git/visual_robust_libero/action"),
                "IIWA_Robotiq85_spatial_t0"),
}


def _config_from_json(cls, ckpt: Path):
    """Rebuild a policy config from its saved config.json.

    Not `from_pretrained`: that goes through draccus, which refuses the "type" discriminator that
    save_pretrained itself writes. The fields are read straight into the dataclass instead, and
    the enum-valued ones are converted back from their serialised names.
    """
    from dataclasses import fields

    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

    raw = json.loads((ckpt / "config.json").read_text())
    raw.pop("type", None)
    known = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in raw.items() if k in known}
    for key in ("input_features", "output_features"):
        if isinstance(kwargs.get(key), dict):
            kwargs[key] = {
                name: PolicyFeature(type=FeatureType[spec["type"]], shape=tuple(spec["shape"]))
                for name, spec in kwargs[key].items()
            }
    if isinstance(kwargs.get("normalization_mapping"), dict):
        kwargs["normalization_mapping"] = {
            k: NormalizationMode[v] for k, v in kwargs["normalization_mapping"].items()
        }
    return cls(**kwargs)


def log(msg: str) -> None:
    print(f"[repair] {msg}", flush=True)


def stats_for(kind: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    root, repo = FIRST_TREE[kind]
    meta = LeRobotDatasetMetadata(repo, root=root / repo)
    return meta.stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, help="run directories holding checkpoint_*/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cache: dict = {}
    for run in args.runs:
        checkpoints = sorted(run.glob("checkpoint_*"))
        if not checkpoints:
            log(f"{run}: no checkpoints")
            continue
        config_path = checkpoints[0] / "config.json"
        kind = "dp" if json.loads(config_path.read_text()).get("type") == "diffusion" else "smolvla"
        if kind not in cache:
            log(f"loading {kind} statistics from {FIRST_TREE[kind][1]}")
            cache[kind] = stats_for(kind)
        stats = cache[kind]

        for ckpt in checkpoints:
            if all((ckpt / n).exists() for n in NEEDED):
                log(f"  {ckpt.relative_to(run.parent)}: already complete")
                continue
            if args.dry_run:
                log(f"  {ckpt.relative_to(run.parent)}: WOULD write processors")
                continue
            if kind == "dp":
                from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
                from lerobot.policies.diffusion.processor_diffusion import (
                    make_diffusion_pre_post_processors as make,
                )

                # from_pretrained runs the config through draccus, which rejects the "type"
                # discriminator save_pretrained writes. Build the dataclass from the JSON directly.
                config = _config_from_json(DiffusionConfig, ckpt)
            else:
                from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
                from lerobot.policies.smolvla.processor_smolvla import (
                    make_smolvla_pre_post_processors as make,
                )

                config = _config_from_json(SmolVLAConfig, ckpt)
            pre, post = make(config, dataset_stats=stats)
            pre.save_pretrained(ckpt)
            post.save_pretrained(ckpt)
            written = sorted(p.name for p in ckpt.iterdir())
            log(f"  {ckpt.relative_to(run.parent)}: wrote processors -> {len(written)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
