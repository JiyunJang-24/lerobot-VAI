# Experiment index

`outputs/` and `dataset_git/` in this repo are symlinks to `/dataset/jiyun/...`, so nothing under
them is in git. These are copies of the analysis documents that live there. Checkpoints and
datasets stay on `/dataset`.

## Read first
- **`../CLAUDE.md`** — the working notes. Section 11 is the most recent work (barx3 robot/scene
  segmentation split); sections 9–10 cover embodiment transfer and motion prediction.
- **`ALL_EXPERIMENTS.txt`** — index of every experiment with checkpoint locations.

## By topic

| file | what it covers |
|---|---|
| `BARX3_SETUP.txt` | barx3 setup, written to be shared as-is (datasets, where contrastive attaches, caveats) |
| `EXP1_PLAN.txt` / `EXP1_RESULTS.txt` | Experiment 1, selective visual alignment: plan, results, go/no-go, and the four measurement artifacts that were fixed |
| `EMBODIMENT_REPRESENTATION.txt` | what the contrastive backbone represents on trained vs held-out robots, plus the real-Jaco check |
| `DATASET_NEEDS.txt` | which datasets exist, and what to render next — with measured scaling curves rather than guesses |
| `VR_NEW_SEG_DEFECT.txt` | defect report for the truncated segmentation corpus (hand to whoever regenerates it) |
| `DP_LANGUAGE_CONDITIONING.txt` | how language attaches to the diffusion policy |
| `pose_invariance_report.txt` | same pose / unseen robot probe |
| `motion_experiments_report.txt`, `vlm_motion_v2_report.txt` | cross-embodiment motion prediction |
| `dataset_redesign_spec.txt`, `dataset_regeneration_prompt.txt` | render specs handed to the data-generation side |

## Standing caveat

Everything reported so far is **training loss**. No experiment in this repo has a held-out task
success number yet, and the baselines flatten early on these dataset sizes. Treat rankings as
provisional until a rollout evaluation exists — the point is made repeatedly in the documents
above because it keeps being the limiting factor.
