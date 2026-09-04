# lerobot-VAI — smolVLA cross-embodiment training

Working notes for this checkout. Read this before running anything; several of the rules below
were learned by losing hours to them.

---

## 0. Environment

```bash
source /home/gpuuser/miniforge3/etc/profile.d/conda.sh && conda activate smolvla
```

**The ffmpeg shim is mandatory for `torchcodec`.** This box lacks the ffmpeg shared libraries
torchcodec links against; `.ffmpeg_shim/` exposes PyAV's bundled ffmpeg 7 under the plain
sonames. `train_smolVLA_robocasa_x.sh` builds and exports it automatically, but any script you
run by hand that touches video must do it itself:

```bash
export LD_LIBRARY_PATH="/home/gpuuser/jiyun/lerobot-VAI/.ffmpeg_shim:$LD_LIBRARY_PATH"
```

Without it you get `Could not load libtorchcodec ... libavutil.so.59: cannot open shared object
file`. Fallback is `VIDEO_BACKEND=pyav`, but pyav is ~11x slower on these packed videos and
starves every DDP rank.

Hardware: 8x H100 80GB, 96 CPU cores, 1.9 TB RAM.

`conda activate` in a non-interactive shell (e.g. `tmux send-keys`) needs the explicit
`source .../conda.sh` first — otherwise the activate silently fails, `tmux has-session`
reports success, and nothing is running.

---

## 1. Datasets

`dataset_git/` is a **symlink to `/dataset/jiyun/dataset_git`** (NFS). Do not commit it.

| path under `dataset_git/` | what |
|---|---|
| `barx_panda_ur5e_iiwa/` | source barx export, per-embodiment task trees |
| `barx_frontonly_p900_i1000_u1000/raw/{panda_mg,iiwa,ur5e}/` | **prepared policy corpus** (2900 eps) |
| `visual_robust_new_barx/new_barx/{IIWAOmron_PnPCounterToSink,PandaOmron_TurnOnSinkFaucet,UR5eOmron_PnPSinkToCounter}/lerobot/` | **auxiliary export**, 108 eps each |
| `robocasa_sweep/`, `cross_embodiment/`, `robocasa_x_atmoic/` | older corpora (TurnOnSinkFaucet, PnP mixes) |

### The auxiliary ("visual robust") export

Each tree stores one episode per row and renders it from **all three embodiments**, so one
frame yields three images differing only in which robot is present — same kitchen, same
instant, same camera pose. Keys: `observation.image.{IIWAOmron,PandaOmron,UR5eOmron}` plus
matching `observation.wrist_image.*`.

`observation.state` is `[8]` = `fingertip_xyz(3) + quat_xyzw(4) + gripper_close(1)`, and there
is **one state per frame shared by all three renders**. That is what makes state regression a
route to embodiment invariance.

### State layouts differ between the two corpora

| corpus | `observation.state` |
|---|---|
| auxiliary | `[8]`: fingertip xyz(0:3) **world frame**, quat_xyzw(3:7), gripper(7) |
| policy | `[16]`: base pose(0:7), **eef pose(7:14)**, gripper(14:16) — unnamed in metadata |

The `[16]` layout was recovered from variance structure, not documentation. Verified
between-episode / within-episode std ratios:

- `d0:3` base xyz → **1910x / 41.9x / 0.6** — the robot's parking spot
- `d3:7` base quat — yaw only (`d3=d4=0`), ratio 127x
- `d7:10` **eef xyz → 0.5 / 0.5 / 0.1 — already base-relative**
- `d10:14` eef quat → 0.5–0.8
- `d14:16` gripper → 0.1 / 0.2

**Never regress world-frame position from these images.** `agentview_right` is robot-mounted,
so the image is identical wherever the base is parked; world position is not inferable. A head
asked for it plateaus at the between-episode std (measured: 1.57 m). The auxiliary export's
fingertip xyz needs per-episode centering; the policy corpus's eef pose does not.

---

## 2. Training entry points

Layered, each adding one thing. Call the highest layer that fits.

```
run_barx_frontonly_{l2sp,distill,eefstate,frozenvis}.sh   experiment presets
run_barx_visualrobust_frontonly.sh                        VR contrastive/alignment preset
  └─ run_visual_robust_with_oom_backoff.sh                OOM ladder (see caveat)
      └─ train_smolVLA_robocasa_x_visual_robust.sh        builds --dataset.visual_robust_* flags
          └─ train_smolVLA_robocasa_x.sh                  DATA PREP + accelerate launch
```

`train_smolVLA_robocasa_x.sh` is the only place that prepares data and launches. Everything
above it just sets env vars. All knobs are `${VAR:-default}` — **keep it that way**; hardcoded
values in a wrapper have silently overridden caller intent three separate times here (batch 64
over a requested 48, vr_batch 8 over 32, `contrastive` over `alignment`).

### Baseline (no auxiliary loss)

```bash
PANDA_TOTAL_EPISODES=900 IIWA_EPISODES=1000 UR5E_EPISODES=1000 \
USE_PANDA_HUMAN=false NORMALIZE_TASK_LANGUAGE=true \
SOURCE_PANDA_MG=$PWD/dataset_git/barx_panda_ur5e_iiwa/PandaOmron/pretrain/TurnOnSinkFaucet/lerobot \
SOURCE_IIWA=$PWD/dataset_git/barx_panda_ur5e_iiwa/IIWAOmron/pretrain/PnPCounterToSink/lerobot \
SOURCE_UR5E=$PWD/dataset_git/barx_panda_ur5e_iiwa/UR5eOmron/pretrain/PnPSinkToCounter/lerobot \
DATASET_ROOT=$PWD/dataset_git/barx_frontonly_p900_i1000_u1000 \
CAMERAS=observation.images.robot0_agentview_right USE_WRIST_CAM=false \
BATCH_SIZE=64 JOB_TAG=barx_frontonly_baseline \
TRAIN_SCRIPT=src/lerobot/scripts/lerobot_train_with_visual_robust.py \
  ./train_smolVLA_robocasa_x.sh
```

Data prep is **idempotent** — it skips a `DATASET_ROOT` that already has the requested episode
counts. Pass `FORCE=true` only when changing episode counts.

### The experiment presets

```bash
# vision tower frozen
./run_barx_frontonly_frozenvis.sh

# weight-space L2-SP toward pretrained
VISION_L2SP_WEIGHT=1e-3 ./run_barx_frontonly_l2sp.sh

# feature-space distillation from a frozen pretrained tower
VISION_DISTILL_WEIGHT=0.1 ./run_barx_frontonly_distill.sh

# EEF-state regression (auxiliary views only)
VR_STATE_WEIGHT=0.1 VR_STATE_POOL=attn ./run_barx_frontonly_eefstate.sh

# EEF-state on auxiliary views AND the policy batch  <-- best gap so far
VR_STATE_WEIGHT=0.1 VR_STATE_POLICY_WEIGHT=0.1 VR_STATE_POOL=attn \
  ./run_barx_frontonly_eefstate.sh

# VR contrastive / alignment
VISUAL_ROBUST_FRONT_OBJECTIVE=contrastive VISUAL_ROBUST_CONTRASTIVE_WEIGHT=0.5 \
  ./run_barx_visualrobust_frontonly.sh
VISUAL_ROBUST_FRONT_OBJECTIVE=alignment  VISUAL_ROBUST_CONTRASTIVE_WEIGHT=1.0 \
  ./run_barx_visualrobust_frontonly.sh
```

Defaults common to all: 8 GPUs, batch 64/GPU (**512 effective — every comparison in §4 depends
on this**), 50k steps, save every 10k, front camera only, wandb project
`robocasa_x_smolvla` / entity `DynamicVLA`.

Useful overrides: `STEPS`, `SAVE_FREQ`, `WANDB_MODE=offline`, `NUM_WORKERS`, `GPU_IDS`,
`MAIN_PROCESS_PORT`, `JOB_TAG`, `LOG`.

### Running two jobs at once

Needs **both**: `MAIN_PROCESS_PORT` (accelerate defaults to 29500; the second job dies with
"address already in use") and lowered `NUM_WORKERS` (12/rank x 8 x 2 jobs oversubscribes 96
cores).

Do **not** route a parallel job through `run_visual_robust_with_oom_backoff.sh` — its
`wait_for_free_gpus` blocks until every `lerobot_train` process exits, so it waits forever
behind the job it is meant to share with. Call `train_smolVLA_robocasa_x.sh` directly.

Measured: sharing 8 GPUs costs ~16% aggregate throughput (2.10 it/s solo → 0.96 + 0.89).
Splitting 4+4 at batch 128 is ~12% *faster* in aggregate, but per-GPU throughput already
saturates by batch 32 (32→64 buys only +8%), so at equal effective batch the two arrangements
are near-equivalent — and splitting is a loss whenever it forces a restart. **Balance by
remaining work, not by job count**: an even split leaves half the machine idle once the shorter
job finishes.

---

## 3. Auxiliary losses (all in `lerobot_train_with_visual_robust.py`)

Config fields live in `src/lerobot/configs/default.py::DatasetConfig`.

| loss | config | acts on |
|---|---|---|
| VR contrastive / alignment | `visual_robust_contrastive_weight`, `visual_robust_front_objective` | mean-pooled tower output (or connector+MLP with `visual_robust_head_mode=adapter_mlp`) |
| L2-SP | `vision_l2sp_weight`, `vision_l2sp_scope` | `sum (w - w0)^2` on tower **weights** |
| feature distillation | `vision_distill_weight` | per-token MSE vs a frozen pretrained tower |
| EEF-state (auxiliary) | `visual_robust_state_weight`, `visual_robust_state_pool` | MLP on tower tokens → 8-d state |
| EEF-state (policy batch) | `visual_robust_state_policy_weight` | same, on the policy batch |

### Non-obvious implementation constraints

**Auxiliary heads must be standalone modules**, built before the optimizer, given their own
`optimizer.add_param_group`, and `accelerator.prepare()`d separately. Registering one on the
policy breaks DDP: the trainer uses `find_unused_parameters=True`, the head is never called
inside `policy.forward`, so DDP marks its params ready at backward start and the real gradient
marks them a second time →
`Expected to mark a variable ready only once ... has been marked as ready twice`.

**Everything that compares two nearly-equal quantities must leave autocast.**
`with torch.autocast(device_type=..., enabled=False)` and `.float()`. bf16 rounding destroyed
the alignment objective outright — its cosine pinned at exactly 1.0 and the gradient vanished.
Same reasoning applies to L2-SP `(w - w0)` and distillation `(f - f0)`.

**One `accelerator.backward(loss)`.** Add every auxiliary term to `loss` first. Separate
backward calls make DDP see the same parameters twice.

**Student features come from a forward hook**, not a second encode. `_StudentFeatureCapture`
grabs the tower's output from inside `policy.forward`, so distillation and the policy-batch
state term cost no extra vision compute. The hook is armed only around `policy.forward` so the
auxiliary encodes cannot leak into it.

**Anchors must be the pretrained tower, and `--resume` needs `load_pretrained_vision_anchor()`.**
Snapshotting off the live policy is correct only at step 0; on a resumed run it would anchor to
already-drifted weights and the run would look healthy while optimising something else. The
pretrained SigLIP is deterministic, so it is reloaded from `cfg.policy.vlm_model_name` and
checked parameter-for-parameter by `_assert_anchor_matches` (197/197 verified).

**Pool per-token, not mean-pooled, when the target has spatial content.** The alignment
objective's failure was that it constrained a pooled statistic (cosine → 0.9999) while the
1024 tokens the policy actually reads went untouched — final action loss identical to baseline.
Distillation is per-token for this reason. For EEF-state, `visual_robust_state_pool=attn`
(learned attention pooling concatenated with the mean) beats `mean` decisively: at 600 steps
mean's position loss *rises* (0.748 → 0.773) while attn's halves (0.898 → 0.574).

**Two state branches need two heads.** The auxiliary target is per-episode-centered world
fingertip; the policy target is base-relative eef. One shared head cannot satisfy both
normalizations — measured `policy_state_loss` 6.6 vs 0.87 with separate heads. Separate heads
still shape the same backbone, which is where the benefit lives.

**Quaternions: use a sign-invariant geodesic loss**, never MSE, and never canonicalize by the
sign of `w`. 64% of this export's frames sit at `|w| < 0.05` — right on the `w=0` boundary — so
forcing `w >= 0` flips half of them and splits near-identical rotations onto opposite targets
(13,216 of 27,066 near-boundary frames). Use `1 - |<q_pred, q_target>|`.

**Guard constant dimensions in any state normalizer.** `PandaOmron_TurnOnSinkFaucet` never
opens its gripper (std exactly 0 in that tree). The usual `std + 1e-8` maps its constant value
to `|z| ~ 1e8` and the loss becomes nothing but that dead dimension. Same trap put 36% of the
action MSE into `panda_human`'s frozen base-motion dims on an earlier PnP mix. Floor the std
and zero-weight the dimension.

---

## 4. Results — barx front-only, 900/1000/1000, batch 512, 50k steps

Final action loss, and the SigLIP positive/negative gap (see
`outputs/siglip_feature_analysis/info.txt` for exact definitions — read it before quoting these):

| run | action loss | rel. weight drift | centered gap | raw gap |
|---|---|---|---|---|
| pretrained init | — | 0 | −0.050 | +0.001 |
| **baseline (no aux)** | **0.069** | 0.1029 | **−0.138** | −0.029 |
| frozen encoder | 0.092 | 0 | −0.050 | +0.001 |
| L2-SP w=1e-3 | 0.098 | 0.0027 | −0.067 | −0.002 |
| distill w=0.1 | — | — | −0.056 | +0.001 |
| alignment w=0.1/0.2/0.5 | 0.069 (= baseline) | ~0.103 | +0.11 … +0.13 | +0.000 |
| contrastive w=0.1/0.5 | — | 0.133 / 0.140 | **+1.00 / +1.01** | +0.82 / +0.83 |
| eef-state w=0.1 | — | — | +0.748 | +0.081 |
| **eef-state w=0.1 + policy w=0.1** | — | — | +0.485 | **+0.165** |
| KI fast (§8) | 0.0947 | 0.0864 | **−0.250** | −0.009 |
| KI lap (§8) | ~0.098 | 0.1009 | **+0.056** | +0.037 |

What this says:

- **The two knowledge-insulation targets move the encoder in opposite directions.** Same
  insulation, same data, only the token target differs: FAST ids give −0.250 (worse than plain
  fine-tuning), the English sentence gives +0.056 (better than pretrained init). A FAST code is
  embodiment-specific, an English phrase is shared across all three arms — which is the LAP claim,
  and it shows up here as a sign flip. The effect is an order of magnitude below eef-state, and
  neither variant helps the action loss. Details in `outputs/siglip_feature_analysis/info.txt` §5b.
- **Fine-tuning the tower makes cross-embodiment similarity worse**, not better: −0.050 at init
  → −0.138 after baseline training. It buys 25% lower action loss (0.092 → 0.069).
- **Alignment is inert.** Positives-only drives every raw cosine to 0.9999 — pos *and* neg — and
  leaves the action loss at exactly baseline. It collapses rather than separates.
- **The regularisers cap out at the init gap.** L2-SP and distillation reach −0.067 / −0.056,
  i.e. they prevent damage but cannot improve on pretrained. L2-SP at 1e-3 crushed drift 40x
  (0.1029 → 0.0027) and landed at frozen's action loss, so it is effectively "frozen with extra
  steps" at that weight.
- **Only contrastive and EEF-state flip the sign meaningfully.** EEF-state does it while pinning
  the invariant to a physical quantity, and it lowers negatives for real (0.979 → 0.689) rather
  than collapsing everything.
- Adding the policy batch doubled raw gap (+0.081 → +0.165) but lowered centered gap
  (+0.748 → +0.485) because it also spread the representation out (raw pos 0.979 → 0.853, i.e.
  less anisotropy). Neither number alone decides which is better for the policy.

**Weight calibration is not optional** — run the probes before a 14-hour job:
`tools/calibrate_l2sp_weight.sh`, `tools/calibrate_distill_weight.sh`. Loss *values* are
useless for sizing these terms (L2-SP is a sum over 86.4M params); read the drift/error
diagnostics instead. Feature drift is ~90% relative error after 210 unregularised steps while
weight drift is ~1% — the two spaces are not substitutes.

---

## 5. Analysis tools

```bash
# gap across every finished barx_frontonly checkpoint (CPU, safe next to training)
CUDA_VISIBLE_DEVICES="" python tools/compare_siglip_all_checkpoints.py
python tools/measure_vision_weight_drift.py          # ||w - w0|| per checkpoint
python tools/explain_siglip_pairs.py --checkpoint <ckpt>   # which images are pos/neg
python tools/dump_visual_robust_pairs.py             # visualise an auxiliary batch
python tools/probe_frozen_encoder_memory.py --checkpoint <ckpt> --batch-sizes 16,32,64,128
python tools/check_visual_robust_ready.py            # is the auxiliary export converted?
```

`compare_siglip_all_checkpoints.py` discovers checkpoints from
`outputs/train/*/*barx_frontonly*/checkpoints/050000/pretrained_model` and labels them from
each `train_config.json`. **Labels must be unique** — `found` is a dict keyed by label, so a
collision silently drops a checkpoint and looks exactly like one that was never trained. Six
runs have `visual_robust_contrastive_weight=0`, so the label is built from every auxiliary
field. This bit twice: once dropping the frozen run, once about to drop four more.

Measured peak memory per rank, front camera only, bf16 (`probe_frozen_encoder_memory.py`):

| batch | freeze=true | freeze=false |
|---|---|---|
| 32 | 5.67 GiB | 17.49 GiB |
| 64 | **9.15 GiB** | **32.44 GiB** |
| 128 | — | ~62 GiB (extrapolated) |

Freezing is 3.5x cheaper at batch 64 despite cutting only 21% of parameters — with no leaf
under the tower requiring grad, autograd stops storing its activations entirely.

---

## 6. Operational gotchas

**`pgrep -f "<pattern>"` matches the shell that is running the check.** A wait loop whose own
command line contains the pattern waits on itself forever; a kill loop kills its own shell
(observed: exit 144, and once a "kill the frozen run" that left the run happily training to
completion while reporting success). Either build the pattern at runtime from concatenated
pieces so it never appears in a cmdline, or use `pgrep -x`.

**Bash arrays cannot cross a process boundary.** `export EXTRA_TRAIN_ARGS=(...)` delivers
*nothing* to the child, which then trains with no auxiliary loss while looking perfectly
healthy. Pass a newline-delimited string and `mapfile -t` it back.

**Env-var continuation chains are fragile.** Inserting a block in the middle of a
`VAR=x \`-continued chain silently broke `USE_PANDA_HUMAN=false`, which pulled a stray
107-episode 1-camera dataset into a run. Keep computed values above the chain.

**Waiting on a job: wait on the process, not a log marker.** A crashed run never writes "End of
training", so a marker wait blocks forever. Conversely, judge success by exit status — one
sweep reported "FAILED exit 0" because it grepped a log the trainer never wrote to.

**Checkpoints save every 10k steps and `last` is a symlink to the newest.** Killing at step
31,995 rolls back to 30,000. There is nothing to resume from before step 10,000.

**Verify a tmux launch actually started**: `pgrep -f lerobot_train`. `tmux has-session`
succeeding means nothing.

**Never re-download over a converted (v3.0) tree.** It restores the original `episode_*.parquet`
and reverts `info.json` to v2.1, destroying hours of conversion. Readiness checks must treat
v3.0 as done.

`tolerance_s=1e-3` (not the 1e-4 default) is required: the PyAV/SVT-AV1 re-encode during episode
subsetting introduces ~1e-4 s of timestamp drift by the end of a file, tripping the dataloader's
assertion on the last frame of a re-encoded episode.

---

## 7. Where things are

```
train_smolVLA_robocasa_x.sh              data prep + launch (the one that matters)
run_barx_frontonly_*.sh                  experiment presets
tools/prepare_robocasa_x_dataset.py      builds DATASET_ROOT/raw/* subsets (idempotent)
tools/prepare_visual_robust_x_dataset.py reshapes the auxiliary export
src/lerobot/scripts/lerobot_train_with_visual_robust.py   all auxiliary losses
src/lerobot/configs/default.py           DatasetConfig — every auxiliary knob
src/lerobot/policies/smolvla/fast_action_tokenizer.py     FAST postfix (knowledge insulation)
tools/smoke_test_knowledge_insulation.py knowledge-insulation checks — run before launching it
outputs/train/<date>/<time>_<job_name>/  checkpoints + train_config.json
outputs/logs/                            run logs
outputs/siglip_feature_analysis/         gap figures + info.txt (read this)

--- the embodiment-transfer line of work (section 9) ---------------------------------------
src/lerobot/scripts/pretrain_siglip_eefpairs.py   contrastive / EEF-state / EEF-pixel pre-training
src/lerobot/scripts/pretrain_siglip_visual_robust.py  the older multi-view-column variant
src/lerobot/scripts/train_dp_embodiment.py        language-conditioned DP, 4 integration modes
tools/eval_heldout_embodiment.py         THE evaluation — scores a tower on unseen embodiments
tools/plot_heldout_results.py            the two result figures
tools/predict_eef_pixel.py               runs the EEF-pixel head on the POLICY corpus (9.5)
tools/diagnose_eef_pixel_sensitivity.py  which axis breaks it — appearance vs geometry (9.5)
tools/measure_feature_displacement.py    how far the FEATURE moves per change (9.5)
tools/inspect_selfws_pairs.py            what a positive/negative pair actually looks like
tools/add_episode_stats_count.py         fixes eef_pairs before v3.0 conversion
tools/declare_selfws_extra_features.py   fixes selfws_v2 before v3.0 conversion
outputs/siglip_pretrain/<tag>/           pre-trained towers + pretrain_info.json (holdout list!)
outputs/heldout_*.json                   evaluation results
outputs/dp_embodiment/<mode>/            DP checkpoints + history.json
```

tmux sessions used: `smolvla`, `smolvla_distill`, `smolvla_eef`.

---

## 8. Knowledge insulation (pi_0.5) — implemented, not yet run to completion

Goal: the flow-matching loss must not reach the VLM (stop-gradient), and the VLM is instead
trained to predict FAST-tokenized actions with its own LM head. Both halves, always — a
stop-gradient with no token objective just freezes the VLM and reproduces the existing
`frozen encoder` result (action loss 0.092) under a misleading name. `SmolVLAConfig.__post_init__`
rejects `ki_fast_loss_weight <= 0` and `train_expert_only=True` for that reason.

```bash
./run_barx_frontonly_ki.sh                       # BATCH_SIZE=32 recommended, see memory below
python tools/smoke_test_knowledge_insulation.py  # run this first, ~3 min on one GPU
```

### What was built

| where | what |
|---|---|
| `configuration_smolvla.py` | `knowledge_insulation`, `ki_fast_loss_weight`, `ki_fast_max_tokens`, `ki_fast_skip_tokens`, `ki_fast_tokenizer_path` |
| `fast_action_tokenizer.py` (new) | action chunk → `"Action: " + FAST ids + eos`, ids mapped into the LM vocab tail |
| `smolvlm_with_expert.py` | the stop-gradient (both layer types), `lm_head` unfrozen under the flag |
| `modeling_smolvla.py` | postfix in the prefix stream, mask/position surgery, CE loss |
| `tools/smoke_test_knowledge_insulation.py` (new) | seven checks, all passing |
| `run_barx_frontonly_ki.sh` (new) | preset |

The sequence the VLM sees is `images | language | state | Action: <FAST ids> eos`, and the action
expert's noisy-action tokens hang off it as before. This deviates from openpi in one place: state
stays a continuous `state_proj` embedding instead of openpi's 256-bin discretization into the text
prompt. Changing it would rewrite the prefix for every other run in this repo, and the token
objective does not need it.

### The two blockers, and how they were resolved

1. **`lm_head` was frozen** by `set_requires_grad()`'s `frozen_layers` list. Under the flag it is
   removed from that list. The original reason (DDP unused-parameter errors) does not apply once a
   CE loss exists — verified: DDP now reports *no* unused parameters at all.
   Worth knowing: the other two entries in that list, `text_model.model.norm.weight` and
   `text_model.model.layers.N.`, **match nothing**. The real names are `model.text_model.norm.weight`
   and `model.text_model.layers.N.`. `lm_head.weight` was the only parameter that list ever froze,
   so "the last text layers are frozen" is not true of this checkout and never was.

2. **The self-attention layers needed the attention split, not a detach.** Resolved as planned:
   * odd layers (`forward_cross_attn_layer`) — one `.detach()` on the prefix K/V the expert reads.
   * even layers (`forward_attn_layer`) — prefix queries attend over undetached prefix K/V, suffix
     queries over detached prefix K/V plus their own, and the two outputs are concatenated.
     Restricting the prefix block drops only entries the mask already killed (prefix rows can
     never reach suffix columns: their `att_masks` cumsum is strictly smaller), so this is the
     same function, just with a gradient cut through the middle of it.

### Two things that are easy to get wrong and are handled

**The action expert must not read the postfix.** The postfix holds the ground-truth actions, and
the cumulative `att_masks` rule would happily let the suffix attend to it — free labels, and the
flow-matching loss would collapse into a lie. `forward()` clears that block of the 2-D mask
explicitly, which also covers the cross-attention layers because they slice their expert mask out
of the same tensor.

**The suffix keeps the position ids it would have had without a postfix.** Otherwise ~250 extra
tokens would shift every action token's RoPE phase and the expert would no longer be comparable to
any baseline. The postfix and the suffix therefore share a position range, which is safe precisely
because they never attend to each other.

### Verified (`tools/smoke_test_knowledge_insulation.py`, 7/7 pass)

* FAST tokens round-trip through the LM vocabulary mapping: `max |a - decode(encode(a))| = 0.074`.
* The split attention reproduces the stock action path to bf16 precision (2.6e-2 on a loss of 13).
* **No label leakage**: `d(flow_loss)/d(postfix embeddings) = 0` exactly, while `d(CE)/d(postfix)`
  is 7.0 — the probe is live, the path is not. Autograd, not a numerical comparison, because
  bf16 noise across two forward passes is larger than a small leak would be.
* **Insulation holds**: the flow-matching term alone leaves `|g| = 0` across all 237 VLM
  parameters while the expert gets `|g| = 13.2`. Without the flag the same loss puts `|g| = 14.6`
  on the VLM — that control is in the test, since "zero gradient" is also what a broken forward
  pass produces.
* **The CE is genuinely next-token.** `d(CE)/d(last postfix token) = 0` exactly — nothing is
  predicted from it. An off-by-one in the shift would let every position see its own target, and
  the loss would fall convincingly while teaching the VLM nothing; this is the check that rules
  that out, and it is worth keeping.
* **The masks say what the design claims**, read off the tensor the model actually passes to the
  transformer: expert → postfix blind, core prefix → postfix blind (so the features the expert
  reads are unchanged by the postfix), postfix causal and able to read the core, and the suffix's
  position ids and suffix → core mask identical to a forward without a postfix.
* **The two losses own disjoint parameters**: CE-only 239, flow-only 45, both 0, neither 0.
* The CE trains the VLM including `lm_head` and the vision tower.
* Overfitting one synthetic batch: CE 45.1 → 0.016, token accuracy 0 → 1.00 in 150 steps.

Real data, barx front-only, batch 64 x 2 GPUs, 300 steps: CE **9.50 → 5.62**, token accuracy
0.068, `fast_tokens_mean` **172** ids per 50x12 chunk (max 243, nothing truncated at the 256 cap),
flow-matching loss 0.22-0.25. Both terms move; nothing NaNs; checkpointing works.

### Three consequences of the disjoint parameter split

**`ki_fast_loss_weight` is close to a no-op.** Because the two losses reach disjoint parameters,
it is only a scale on the VLM branch's gradient — and AdamW's update `m̂/(√v̂ + eps)` is invariant
to a constant gradient scale. To actually tune how fast the VLM learns relative to the expert, give
it its own `add_param_group` learning rate; changing this weight will do almost nothing.

**`state_proj` is now trained by the CE only.** It feeds the VLM prefix, so the stop-gradient cuts
it off from the flow-matching term. That is the correct reading of pi_0.5 — state is part of what
the VLM sees — but it does mean the action expert's only view of the state is through detached
prefix K/V.

**Global gradient clipping briefly couples them anyway.** `optimizer_grad_clip_norm=10` is applied
across the whole model, and the CE pushes the total norm to 22-37 for the first ~1k steps
(baseline: 1-3, never clipped), so everything including the expert gets scaled down together.
Uniform scaling plus Adam's scale invariance makes this second-order, and by step ~2k the norm is
back to 3-5 and clipping stops firing. Worth knowing, not worth fixing.

**Compare `flow_matching_loss`, not `loss`.** The logged `loss` is now flow + CE and is dominated
by the CE (~4.5 vs ~0.17), so it is not comparable to the section-4 numbers.

### Cost — this is the part that changes how you launch it

The postfix adds ~172 tokens to a ~163-token sequence, and attention here is eager (it materializes
`B x heads x L x L` in float32), so memory grows with the square. Measured peak on one H100,
2-GPU DDP, front camera only:

| run | per-rank batch | peak GPU mem | s/step |
|---|---|---|---|
| baseline | 64 | 36.6 GiB | 0.387 |
| baseline | 48 | 28.2 GiB | — |
| KI fast | 32 | 56.4 GiB | 0.317 |
| KI fast | 48 | 70.3 GiB (2 GPU) / **79.5 GiB (8 GPU)** | 0.450 / 0.483 |
| KI fast | 64 | **79.7 GiB of 79.7 available** | 0.576 |
| KI lap | 48 | 31.5 GiB (8 GPU) | 0.372 |

Compare like with like: `lap` costs 3.3 GiB **more** than the baseline at the same batch, not less.
It only looks cheap next to a batch-64 baseline. The whole difference is postfix length — 22 tokens
against fast's 230-250, so the VLM sequence is ~167 against ~385 and the quadratic attention term
is 5x smaller, plus the LM head runs over ~19 supervised positions per sample instead of ~176.

**Per-rank memory is not uniform under this flag, and the table above understates 8-way DDP.**
The postfix length is data-dependent — it is the longest FAST sequence in that rank's own batch,
plus 4 — so every rank runs a different `L` every step (measured 232-250 at batch 64 over 300
steps, cap 256), and eager attention squares that difference. On the live 8-GPU batch-48 run the
ranks spread over **52-79 GiB** and the same rank moves tens of GiB between samples, because
`nvidia-smi` reports the caching allocator's reserved pool and varying shapes make it release and
regrow segments. No rank is structurally heavier: over a long run every rank meets a near-cap
batch, so treat the worst observed number as everyone's ceiling. The baseline has a fixed sequence
length and shows none of this (36.6 GiB, flat).

If a rank does OOM, resume rather than restart — the env var is not part of the config, so this
stays a legal continuation of the same run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
BATCH_SIZE=48 OUTPUT_DIR=<the run's dir> EXTRA_TRAIN_ARGS_STR=$'--resume=true' \
  ./run_barx_frontonly_ki.sh   # loses at most SAVE_FREQ steps
```

Batch 64 completed 100 steps but sits at 98% of the card — one longer-than-usual postfix would
OOM it, and it cannot share GPUs with anything. **Use `BATCH_SIZE=32`** (effective 256 on 8 GPUs,
half the 512 the section-4 baselines used — say so when comparing) or 48 if the box is otherwise
idle. If the effective batch has to stay at 512, the cheap win available is `masked_fill_` instead
of `torch.where` in `eager_attention_forward`, which drops one full `L x L` float32 tensor per
layer; it is numerically identical but touches the path every existing run uses, so it was left
alone here.

`ki_fast_max_tokens` (default 256) caps the postfix; `fast_truncated_frac` in the metrics tells
you if it is biting. Truncation costs supervision on the tail of the chunk, nothing else.

### Results — batch 48 x 8 GPUs (effective 384, not the 512 of section 4)

`./run_ki_then_lap.sh` runs both objectives back to back, everything else held fixed.

| run | flow-matching loss | token CE | token acc | vision drift | lm_head drift |
|---|---|---|---|---|---|
| baseline (fine-tuned tower, §4) | **0.069** | — | — | 0.1029 | 0.00000 |
| frozen encoder (§4) | 0.092 | — | — | 0 | — |
| **KI fast**, 50k done | **0.0947** | 4.06 | 0.136 | **0.0864** | **0.14925** |
| **KI lap**, in progress | ~0.098 @37k | 0.0024 | **1.000** | (pending) | (pending) |

**Both KI variants land on the frozen-encoder action loss, not the baseline's.** The insulation
works exactly as designed and the VLM does train — `fast` moved the vision tower 84% as far as the
baseline did (0.0864 vs 0.1029) on the token objective alone, and unfroze `lm_head` (0 → 0.149).
It just does not buy the action expert anything here.

**The `lap` objective saturates and then stops teaching.** Content-word accuracy: 0.69 at step 200,
0.91 by 2.4k, 0.985 by 10k, **1.000 from ~15k onward**, CE 0.002. The label is not degenerate —
525 distinct sentences over the corpus, 7.72 bits of entropy, top-1 only 2.7% — but 7.7 bits per
chunk over ~12 epochs is memorisable, and once it is memorised the CE gradient vanishes. From 15k
on, the VLM has neither loss reaching it, which is the frozen-encoder condition under another name.
`fast` never saturates (CE 4.06, accuracy 0.136 at 50k), which is the argument for it.

### bf16 parameters silently freeze the LayerNorm scales

Not a knowledge-insulation issue — it is true of every run in this checkout, and it turned up while
auditing whether the backbone really trains. The VLM is loaded with `torch_dtype="bfloat16"`, so
the *parameters* are bf16, not just the compute. bf16's ULP near 0.2 is ~0.0016 while the median
`max|dw|` over 50k steps is 0.029, so any tensor whose accumulated update stays under half a ULP
never moves at all. Measured on the 50k checkpoints:

| run | tensors bit-identical to pretrained | of which LayerNorm/RMSNorm |
|---|---|---|
| baseline (no KI) | 45 of 345 | 39 |
| KI fast | 38 of 345 | 38 |

0.02% of parameters, so it does not change any conclusion — but "everything is trainable" is not
the same as "everything moved". Fixing it means fp32 master weights, which would break
comparability with every result in section 4, so it is left alone deliberately.

### Not done
* RA-BC per-sample weighting with knowledge insulation raises `NotImplementedError` — the CE is a
  scalar over tokens, not a per-sample loss. Nothing needs it today.
* Chunks that run past the end of an episode are dropped from the token loss (they are padded with
  repeated frames, and one token stream cannot mask per timestep the way the flow loss does). The
  key that flags them, `actions_id_pad`, is not currently produced by this dataset path, so that
  guard is untested in practice.

---

## 9. Embodiment transfer — can a policy act on robots it has no demonstrations for?

**The research question.** Collecting action demonstrations for every embodiment does not scale.
Rendering a robot from its URDF at many EEF poses does. So: can *static* embodiment data, never
used for policy learning, let a policy act on embodiments it has no demonstrations for? If yes the
recipe becomes `new embodiment -> URDF -> render -> adapt the representation` instead of
`new embodiment -> collect demonstrations`.

The advisor (Jensen) raised two concerns, and they organise everything below:

1. **Is the embodiment set large enough to generalise to held-out embodiments?**
2. **Is a pre-trained representation enough for control?** (he expects frozen features not to be)

### 9.1 The three-layer evaluation, and why layer 3 does not exist yet

| layer | what it measures | status |
|---|---|---|
| 1 — representation, held-out embodiments | does the *encoder* transfer? | **done, section 9.4** |
| 2 — policy action loss, held-out embodiments | does the *policy* transfer? | blocked, see below |
| 3 — task success rollout | the actual claim | **impossible here** |

**Layer 3 cannot run on this machine**: robosuite / robocasa / mujoco are not installed and the
only registered envs are aloha / pusht / libero / metaworld. `cfg.env` has never been set in any
run in this repo, so `pc_success` has never been computed — every number anywhere in this file is
action loss or a feature statistic, never task success. Standing up a RoboCasa env is a separate
piece of work and is the single biggest gap in the project.

**Layer 2 is data-blocked, not code-blocked**: the demonstration corpora (barx, PnPSinkToCounter)
have 3-6 embodiments, so holding one out leaves almost nothing to hold out. The 56-embodiment data
is static poses, not trajectories. Doing layer 2 properly needs a demonstration corpus with many
embodiments.

### 9.2 The datasets, and what "positive pair" means in each

All under `/dataset/jiyun/dataset_git/`. **All arrived as v2.1 and needed conversion** — see 9.6.

| dataset | shape | positive pair |
|---|---|---|
| `visual_robust_new_barx/new_barx` | 3 trees, one row per frame, **one column per embodiment** (3) | several columns of one row |
| `visual_robust_new_barx_ur5e/new_barx` | 1 tree, same layout, **6 embodiments** (+Jaco, +PandaGripper variants) | same |
| `eef_pairs/48_bg12_layout_{closed,opened}` | one image per **row**, `observation.embodiment_index` | rows sharing an episode |
| `eef_pairs/56combo_48_bg12_{closed,open}{,_furniture}` | **56 embodiments** x 48 poses x 12 backgrounds x 4 cameras | rows sharing an episode |

The column-per-embodiment layout and the row-per-embodiment layout need *different code*; that is
why `pretrain_siglip_visual_robust.py` and `pretrain_siglip_eefpairs.py` both exist. Same
objective, different assembly.

**The 56combo family is the one to use.** Its four subsets are re-renders of ONE pose set — episode
0 carries EEF xyz `[0.5222, 0.1529, 0.4192]` in all four — which forces a decision the loss is
sensitive to:

* colour/texture (`_furniture`) is **nuisance appearance**: the arm is in the identical
  configuration, so those pairs are POSITIVE.
* gripper open vs closed is **real state**, not appearance. Pulling those together would train the
  encoder to discard whether the gripper is open, which a policy needs. So they are NEGATIVE.

`build_row_table(share_poses=True)` implements exactly that: 96 poses = 48 closed + 48 open, with
colour variants merged into each. Getting this backwards is silent — the loss still falls.

**A caveat that limits the analysis**: `embodiment_index` is an integer with no name mapping
anywhere in the export (not in `meta/`, not in the dataset card). So held-out splits are random and
"unseen *arm*" cannot be separated from "unseen *gripper*". The older
`visual_robust_new_barx_ur5e` DOES name them (`IIWAOmron_R85` etc). **Ask for the
index -> (arm, gripper) mapping** — it is the single cheapest thing that would sharpen concern #1.

### 9.3 Pre-training the tower (`pretrain_siglip_eefpairs.py`)

```bash
python src/lerobot/scripts/pretrain_siglip_eefpairs.py \
  --subsets 56combo_48_bg12_closed,56combo_48_bg12_open,56combo_48_bg12_closed_furniture,56combo_48_bg12_open_furniture \
  --objective all --holdout-embodiments 0,1,3,8,12,14,23,27,28,33,34,36,42,49 \
  --steps 4000 --output-dir outputs/siglip_pretrain/<tag>
```

`--objective {contrastive,eef_state,eef_pixel,all}`. The three targets are **not** interchangeable,
and the data says why:

| target | std WITHIN a pose | what it can teach |
|---|---|---|
| xyz + quat | **0.0000** | one value per pose — the same information the contrastive grouping already carries |
| pixel (u,v) | **40.7** | varies across the 4 camera views — the only target that asks *where in THIS image* |

So the pixel head uses attention pooling (a mean over 1024 position-tagged patches is close to
location-blind — measured, section 3) and the state head uses mean pooling. Quaternions use
`1 - |<q,q̂>|`, never MSE: q and -q are one rotation.

**The image cache is not optional.** Decoding frames inside the training loop drove load average
past 500 on 96 cores while the GPUs sat at **0%**. The script now decodes once into
`/dev/shm/eefpairs_cache_<subsets>.pt` (53 GB for all four) and indexes RAM after. Two rules:

* build the cache with ONE run first, then launch the rest — eight processes opening a half-written
  53 GB file all die with `PytorchStreamReader failed reading zip archive`.
* stagger launches ~25 s apart so the eight 53 GB reads do not collide.

### 9.4 Layer 1 results — the representation DOES transfer

`tools/eval_heldout_embodiment.py` scores a tower **only on embodiments it never trained on**. This
is the measurement that was missing: `compare_siglip_all_checkpoints.py`,
`probe_embodiment_invariance.py` and the section-4 gap column are all computed on the *training*
embodiments and therefore cannot answer the generalisation question at all.

Three metrics, because the gap alone is ambiguous:

* **A/B/C pair split** — A = same pose / different embodiment (want HIGH), B = different pose /
  same embodiment (want LOW). A alone cannot separate invariance from collapse.
* **pose retrieval top-1** — query an unseen embodiment against a gallery of seen ones, correct if
  the neighbour shares the canonical pose. **The honest one**: its form is nothing like the
  contrastive objective, so it cannot be satisfied by having memorised that loss.
* **embodiment probe** — a linear probe recovering *which* robot. Invariance should push it to chance.

**Scaling ladder** (each run scored on its own complement; figure: `outputs/heldout_scaling.png`):

| embodiments trained on | held-out gap | pose retr@1 | emb-probe | (held-out size) |
|---|---|---|---|---|
| stock SigLIP | −0.099 | 0.118–0.210 | 0.15–0.52 | — |
| 4 | +0.329 | 0.587 | 0.024 | 52 |
| 8 | +0.646 | 0.785 | 0.000 | 48 |
| 16 | +0.668 | 0.848 | 0.005 | 40 |
| 32 | +0.973 | 0.969 | 0.009 | 24 |
| **42** | **+0.994** | **0.987** | 0.015 | 14 |

**Answer to concern #1: yes, and it has not saturated.** 4 embodiments already transfer (retr 0.587
vs 0.210 stock); 42 reaches 98.7% — a pose on a robot the encoder has never seen is retrieved
almost perfectly. Nothing here suggests a ceiling, so more embodiments is still the right lever.

*Read the trend with its confound*: the held-out set SHRINKS as N grows (52 → 14), so the x axis
moves the test set too. Scoring every rung on one common set would be worse — 2 of the common 14
are inside n=4's training set and 8 are inside n=32's. The figure states this rather than hiding it.

**Objectives** at n=42, all on the SAME 14 held out (`outputs/heldout_objectives.png`):

| objective | held-out gap | pose retr@1 | emb-probe |
|---|---|---|---|
| contrastive | **+0.994** | **0.987** | 0.015 |
| EEF state (xyz+quat) | +0.798 | 0.969 | 0.029 |
| EEF pixel (u,v) | +0.154 | 0.799 | **0.324** |
| all three | +0.988 | 0.982 | **0.000** |

**EEF-pixel alone barely creates invariance** (gap +0.154, probe 0.324) and the reason is
structural, not a bug: "where in this image" can be answered without ever deciding that two robots
are the same, so nothing in it suppresses embodiment identity. It still learns the task (2.1 px
error) and retrieves at 0.799 — it carries pose information, just not invariance.

**`all` is the tower to use**: gap and retrieval match contrastive while the embodiment probe drops
to exactly 0.000. That is the combination the project wants — pose information kept (via the
regression heads), embodiment identity gone.

### 9.5 Does the pose knowledge survive the domain change? No — and this is the sharp result

`tools/predict_eef_pixel.py` takes the EEF-pixel head off a pre-trained tower and asks it "where is
the gripper" on the POLICY corpus, with no fine-tuning. Two figures per tower, named after `--out`:
`outputs/eef_pixel_<tag>.png` (overlays) and `outputs/eef_pixel_<tag>_track.png` (one episode per
robot). Nothing here needed new training — the heads were saved alongside the towers.

| | `all4_n42_all` | `all4_n42_eefpixel` |
|---|---|---|
| training pixel error | 3.55 px | 2.08 px |
| **eef_pairs, held-out embodiments** | **5.37 px** | **3.02 px** |
| policy corpus, per-episode camera fit | 18.69 px | 20.25 px |
| the same fit on a **shuffled** pairing | 22.86 px | 25.64 px |
| predicted spread (u, v) | 26, 21 | 28, 25 |
| the spread it was trained on | 46, 43 | 46, 43 |

**In its own domain the head is excellent and embodiment-invariant**: 3-5 px on a 320x180 image, on
robots the tower never trained on. That is the strongest confirmation of 9.4 available — the
invariance is not an artefact of the contrastive metric, because a head reading these features can
locate the gripper on an unseen robot.

**On the policy corpus it collapses to a prior.** The check is not the picture: within one episode
the camera is fixed and the base does not move, so if the head were finding the gripper, one 11-DOF
pinhole projection would map `observation.state[7:10]` to its predictions within a few pixels. It
does not — the residual is ~0.8x what the *same fit on randomly re-paired frames* scores, over 18
episodes and three robots. Meanwhile the predicted spread has fallen to about half the spread of
the training labels, and the mean sits in the middle of the frame. Both signatures say the same
thing: the output barely depends on the input. The track figure shows it directly — a blob near the
sink with the time colouring scrambled inside it, while the gripper is at the top of the frame.

**It is geometry, not background** — `tools/diagnose_eef_pixel_sensitivity.py`, scored against real
labels on held-out embodiments (`outputs/eef_pixel_sensitivity.png`). "It is sensitive to the
background" is the natural reading of the failure and it is **wrong**, which matters because the two
readings point at completely different fixes:

| axis varied | seen in pre-training? | pixel error |
|---|---|---|
| 12 backgrounds | yes | 3.38 – 5.50 (flat) |
| 4 camera views | yes | 3.83 – 4.70 (flat) |
| furniture recolour, same pose | yes | 4.63 → 4.81 |
| **zoom out to 0.75 / 0.55 / 0.40 / 0.30** | no | **19.1 / 36.6 / 51.8 / 65.5** |
| **zoom in to 0.75 / 0.55** (no padded border) | no | **20.1 / 29.3** |
| **shift 40 px across, size unchanged** | no | **29.9** |

Appearance is free — 12 backgrounds and a furniture recolour move the error by under 1 px. Geometry
is not: a 40-pixel horizontal shift, 12% of the image width, takes it from 4.6 px to 29.9 px with
the apparent size untouched. Zoom-in is the control that rules out the padded border zoom-out
introduces, and it fails just as hard.

So the head never learned "find the gripper". It learned an absolute pose → pixel mapping for the
**4 fixed canonical camera geometries** eef_pairs contains, and it handles those four beautifully on
robots it has never seen. Any fifth geometry — which is what the policy corpus is — is outside it.

**And the features move with it, not just the head's readout.**
`tools/measure_feature_displacement.py` perturbs one image and measures how far its *feature* goes,
in the only unit that means anything here — how far the feature goes for a genuine change of pose
(`outputs/feature_displacement.png`). That reference is itself 98% of the distance between two
unrelated images, so read these as "fraction of the way to nothing in common":

| what changed | feature displacement |
|---|---|
| background (same pose, same robot) | **1%** |
| furniture recolour (same pose, same robot) | **2%** |
| different robot (same pose) | **4%** |
| shift 40 px (same image, moved) | **31%** |
| zoom out to 0.75 (same image, smaller) | **41%** |
| genuine pose change | 100% (the reference) |
| a policy-corpus frame | **102%** — fully decorrelated |

So the answer to "wrong pixel means the feature moved too" is **yes**, and this is the sharpest
statement of the problem available: the tower treats a 40-pixel camera shift as a third of the way
to a completely unrelated image, while treating a different *robot* as 4%. The invariance it learned
is to embodiment and appearance; it is emphatically **not** invariance to camera geometry, and pose
and camera pose are entangled in the same directions.

This is what makes it a policy problem and not a head problem: the policy reads these features and
never touches the pixel head. A policy-corpus frame sits at the unrelated-image floor, so a frozen
tower hands the policy features that carry none of the pre-trained structure — which is the
mechanism behind "frozen contrastive tower ≈ frozen stock tower" in section 4 (0.136 vs 0.129).

**This makes the fix cheap and specific.** Random resized crop / shift augmentation during
pre-training attacks exactly the axis that is broken, and costs nothing but a re-run; adding more
backgrounds or more embodiments attacks axes that are already flat. Worth measuring before anything
more elaborate.

**Why this matters more than it looks.** It is a concrete mechanism for the section-4 puzzle, where
an auxiliary objective moved the feature metric from -0.05 to +1.00 and left action loss exactly at
baseline. The pre-training does teach a real, embodiment-invariant, physically-grounded skill; that
skill just is not available on the images the policy actually sees. So the shortfall is a
scene-domain gap, and the levers it points at are different from "more embodiments": render
embodiments into the *policy's* scenes, or mix policy frames into pre-training.

**Do not read this as "eef_pixel is a bad objective."** 9.4 measures how it shapes the
representation; this measures whether one head transfers across a visual domain shift. They are
different questions, and this test has no bearing on the `all` tower's use in section 9.6.

### 9.6 Layer 2 attempt — language-conditioned Diffusion Policy (RUNNING)

The point of a DP here is to remove the VLM as a confound: if the representation helps a plain
policy but not the VLA, the problem is VLM integration; if it helps neither, the representation
itself is what needs revisiting.

Stock `DiffusionPolicy` has **neither language conditioning nor a SigLIP backbone**, so both were
added to `configuration_diffusion.py` / `modeling_diffusion.py`, defaulted OFF so every existing
diffusion run is untouched:

* `language_conditioned` appends a **frozen** sentence embedding of `batch["task"]` to the UNet's
  global conditioning. Frozen on purpose — this experiment isolates the VISUAL representation, and
  a trainable text tower would add a second moving part.
* `use_siglip_encoder` / `siglip_encoder_path` / `freeze_vision_encoder` swap the ResNet encoder for
  the pre-trained tower. Without this, "offline pretrain → frozen / finetune" is not expressible,
  because the pre-training artefact IS a SigLIP tower.

```bash
python src/lerobot/scripts/train_dp_embodiment.py --mode {online,frozen,finetune,scratch} \
  --tower outputs/siglip_pretrain/all4_n42_all/vision_tower.safetensors \
  --steps 30000 --output-dir outputs/dp_embodiment/<mode>
```

Corpus: **PnPSinkToCounter for IIWA / Panda / UR5e** (`dataset_git/barx_panda_ur5e_iiwa`), one task
across three embodiments, 1,445,640 frames. The four modes are the three integration strategies
from the plan plus a control:

| mode | tower | trainable params | note |
|---|---|---|---|
| `online` | pre-trained | 364.3M | 56combo contrastive applied jointly, on the policy's OWN encoder |
| `frozen` | pre-trained, fixed | 277.8M | fastest — no backward graph through the tower |
| `finetune` | pre-trained, adapts | 364.3M | |
| `scratch` | stock SigLIP | 364.3M | **the control** — without it, "does embodiment data help" has no baseline |

Verify the mode took effect from the trainable-parameter count in the log, not from the flag.

**The online aux term silently ran at chance for its first launch — check for this.**
`load_images` returns [-1, 1] (the tower's range) while `SiglipRgbEncoder.forward` re-scales
[0, 1] -> [-1, 1] itself, because that is what the DP dataloader hands it. Feeding the aux batch
straight through gave the tower [-3, 1]. Measured on one real batch with the same tower: **4.84
through the DP path vs 1.63 through the pre-training path**, and in the live run it settled at
**3.849-3.851 = ln(47)**, exactly uniform over a 48-sample batch. `next_batch` now converts back
to [0, 1].

`aux_loss ≈ ln(batch − 1)` is the signature of a dead contrastive term, and it is worth checking on
sight: nothing about the run looks wrong otherwise — the action loss falls normally, no NaN, no
warning. The second tell is the gradient norm, 0.91 broken vs 15.33 fixed at the same step.

**Cost: the step is ~95% vision tower, so watch the tower and ignore everything else.** Measured
at batch 64 on one H100:

| | time |
|---|---|
| SigLIP fwd+bwd, 512x512, **fp32** | **1021 ms** |
| SigLIP fwd only (what `frozen` pays) | 352 ms |
| SigLIP fwd+bwd, 512x512, **bf16 autocast** | **247 ms** |
| SigLIP fwd+bwd at the image's native 192x320 | 213 ms |
| ResNet18, the stock DP encoder, fwd+bwd | 14.8 ms |

`finetune`/`scratch` ran at 0.93 it/s = 1075 ms/step against 1021 ms of tower: the diffusion UNet
and the dataloader are noise. Two separate causes, and only one is free:

* **fp32.** `SiglipRgbEncoder` loads with `dtype=torch.float32` and nothing wrapped it in autocast,
  so the tensor cores went unused. `--amp` (default on) is **4.1x** on the tower and changes
  nothing numerically — fp32-vs-bf16 feature cosine is **1.00000 at the minimum**, not the mean.
  This is autocast over fp32 master weights, so it is NOT the bf16-parameter trap in section 8.
  End to end it is ~2x rather than 4x, because once the tower stops dominating the UNet and the
  loader become the next terms.
* **512x512.** The frames are natively 180x320 but the tower is fed 512x512, i.e. 1024 patch
  tokens where 240 would do. Worth 4.8x and NOT free: every tower in `outputs/siglip_pretrain/`
  was pre-trained at 512 (`load_images` pads to 512), so taking this would mean redoing
  pre-training at the new resolution as well. Do both together or neither.

The first round of four runs predates `--amp` and is fp32 throughout; `--no-amp` reproduces it.

### 9.7 Every preprocessing trap these exports contained

All four were silent-until-fatal, and all are fixed by tools that are idempotent:

1. **selfws_v2: 120–304 undeclared parquet columns.** `eef_state.<tag>`, `reachable.<tag>` etc are
   in the data and documented in `embodiment_coverage.json`, but absent from `info.json`'s
   `features`. `Dataset.from_parquet` fails with "column names don't match" before conversion
   starts. → `tools/declare_selfws_extra_features.py` (infers dtype/shape from the parquet).
2. **selfws_v2 no_kitchen: declared videos that do not exist.** `YAMOmron_AG` is in `info.json` and
   not on disk — *and not in the hub listing either* (76 present, 78 declared), so it was an export
   bug, not a partial download. Checked before re-downloading.
3. **eef_pairs: `episodes_stats.jsonl` has no `count`.** `aggregate_stats` needs it to weight each
   episode; conversion dies with `KeyError: 'count'`. → `tools/add_episode_stats_count.py` reads the
   episode length back from the parquet rather than guessing.
4. **`hf download --include` matches nothing on these repos** (`min() arg is an empty sequence`)
   even though `list_repo_files` returns the files. Work around it by listing files via the API and
   calling `hf_hub_download` per file with a thread pool. Rate limit is **1000 API requests / 5 min**,
   so `raw_images` (140k+ files) turns a 13-hour download into a few minutes when excluded — and it
   is not needed for training.

### 9.8 What a new session should do next

**Do not** re-run pre-training to "check it works" — the towers are in `outputs/siglip_pretrain/`
and each carries its held-out list in `pretrain_info.json`. Read that before evaluating anything.

In priority order:

1. **Collect the DP results** (section 9.6, running now). Compare `scratch` against the three
   integration modes. That is the direct answer to concern #2.
2. **Stand up a RoboCasa env for layer 3.** Everything so far is a proxy, and section 4 already
   shows the proxy and the policy disagreeing (contrastive reached gap +1.00 with no action-loss
   benefit). Until task success exists, no result here settles the research question.
3. **Get the `embodiment_index -> (arm, gripper)` mapping** and redo the ladder split by axis. "Is
   an unseen *arm* harder than an unseen *gripper*" is a sharper form of concern #1 than the random
   split can answer.
4. **Push the ladder past 42** if more embodiments become available — nothing in 9.4 has saturated.

**Standing caution for this line of work**: the section-4 table shows an auxiliary objective can
move the feature metric a long way (contrastive: gap −0.05 → +1.00) while leaving action loss
exactly at baseline. A representation number improving is not evidence the policy improved. Report
both, or say plainly which one is missing.


---

## 10. Cross-embodiment motion prediction (experiments 1–3)

**The hypothesis.** If a model learns to read the *same* Cartesian EEF motion across many
synthetically rendered robots, that visual-motion knowledge may transfer to held-out embodiments and
carry task knowledge to robots with no demonstrations. Three questions, deliberately separable:

* **Q1** can visual motion knowledge generalise to unseen morphologies? — experiment 1
* **Q2** does more synthetic embodiment diversity improve that? — experiment 2
* **Q3** does it improve task transfer to robots without demonstrations? — experiment 3

### 10.1 What the repo already had, and what had to be built

| need | reused |
|---|---|
| paired renders of one motion across robots | `eef_pairs/56combo_*` — no new rendering needed, see below |
| image cache | `/dev/shm/eefpairs_cache_*.pt` from `pretrain_siglip_eefpairs.py` |
| visual backbone | the SmolVLM SigLIP tower, same loader as `SiglipRgbEncoder` |
| pooling that is not location-blind | `RegressionHead`'s attention pooling (section 3) |
| policy for experiment 3 | `train_dp_embodiment.py` — language-conditioned DP, already built |
| held-out scoring | `tools/eval_heldout_embodiment.py` pattern |

New: `src/lerobot/scripts/motion_data.py` (conventions + guards),
`src/lerobot/scripts/train_motion_prediction.py` (experiments 1–2),
`tools/report_motion_experiments.py`, and `--motion-aux` / `--heldout-robots` on the DP trainer.

**No new rendering was required.** All 56 embodiments in `56combo_*` are posed at the SAME 48
canonical EEF states, so "pose i → pose j" is literally the same motion on every robot. Drawing ONE
pool of pose pairs and reusing it for every embodiment makes embodiment-identity-as-shortcut
impossible *by construction* rather than by sampling luck — `assert_no_motion_shortcut()`.

### 10.2 The action convention — read this before trusting any motion label

`observation.state` is `[16]`: base pose `0:7`, **eef xyz `7:10`**, **eef quat `10:14` (wxyz)**,
gripper `14:16`. `eef_pairs` is `[9]`: xyz `0:3`, **quat `3:7` (xyzw)**, eef pixel `7:9`.
**The two corpora use opposite quaternion order.** `motion_data.py` is xyzw throughout and is the
only place the other convention appears.

`action` is `[12]` = `[base(3), torso, mode, dpos(3), drot(3), gripper]` — and **it is not a metric
displacement**. `a[0:4]` are identically zero, `a[4]` is −1 (mode), and `dpos`/`drot` are normalised
OSC commands. Regressing achieved base-frame EEF delta on commanded `dpos`:

| robot | R² | per-unit scale |
|---|---|---|
| IIWA | 0.47 | ~0.007 m |
| Panda | 0.31 | ~0.006 m |
| UR5e | 0.71 | ~0.010 m |

A different near-diagonal matrix per robot, and less than half the variance explained. Using the
action as the cross-embodiment motion target would bake per-robot controller gains into the label —
precisely the embodiment shortcut the hypothesis is trying to avoid. **Every target here is the
measured state delta instead**, in the robot BASE frame, which is also the frame the base-mounted
`agentview_right` camera is fixed to.

Required sanity test, run at every startup (`verify_reconstruction`): `p_i + d_pos` reproduces
`p_j` to **0.00e+00 m** and `q_rel ⊗ q_i` reproduces `q_j` to **7e-08**, sign-invariantly.

### 10.3 The one mismatch that is NOT fixed

Translation matches: real motion over 25 steps @20 fps is 0.09 m; synthetic pose pairs within
0.15 m average 0.096 m. **Rotation does not**: real is ~9°, synthetic is ~60°, because those 48
poses were sampled to *cover the workspace*, not to look like trajectory segments. Capping helps and
costs vocabulary:

| cap | pairs | mean rotation |
|---|---|---|
| none | 1262 | 60.2° |
| 45° | 458 | 27.7° |
| **30° (experiment 3 default)** | **270** | **21.2°** |
| 20° | 96 | 14.0° |

Both numbers are logged, never smoothed over. **The proper fix is at render time**: sample `s_t`
then a *local* `delta_eef`, which is what the original brief described. The existing `eef_pairs`
export approximates it by pairing workspace-covering poses.

### 10.4 Commands

```bash
# experiment 1
python src/lerobot/scripts/train_motion_prediction.py --tag exp1_n42 --steps 6000
# experiment 2 — protocol A (data grows) and B (data fixed)
... --tag exp2A_n8 --num-embodiments 8 --samples-per-embodiment 4000
... --tag exp2B_n8 --num-embodiments 8 --fixed-total-samples 60000
python tools/report_motion_experiments.py          # tables + outputs/motion_experiments.png

# experiment 3 — A vanilla, B real motion aux, C synthetic + real
python src/lerobot/scripts/train_dp_embodiment.py --mode scratch --motion-aux none \
    --heldout-robots UR5eOmron --output-dir outputs/dp_motion/A_vanilla
... --motion-aux real  --output-dir outputs/dp_motion/B_real
... --motion-aux both  --output-dir outputs/dp_motion/C_synth_real
```

Results land in `outputs/motion_prediction/<tag>/results.json` (per-embodiment breakdown included)
and `outputs/dp_motion/<tag>/history.json` (with `per_robot`, the seen-vs-held-out table).
Task loss and motion loss are logged separately, always.

### 10.5 Results — Q1 and Q2 are both yes

**Experiment 1** (42 train / 14 held-out embodiments, 6000 steps). Baselines: predict-the-mean is
4.66 cm, predicting no rotation at all is 60.2 deg.

| split | translation MAE | direction cos | rotation err | gripper acc |
|---|---|---|---|---|
| seen | 0.41 cm | 0.993 | 8.2° | 0.992 |
| **held-out** | **0.67 cm** | **0.980** | **10.8°** | **0.970** |

**Q1: yes.** On robots it has never seen, the model reads the Cartesian displacement to 0.67 cm —
7× better than predict-the-mean — with the direction essentially right (cos 0.980). The seen →
held-out gap is small (0.41 → 0.67 cm), so this is generalisation, not recall. Per-embodiment
spread is 0.41–1.00 cm; no held-out embodiment fails.

**Experiment 2.** Held-out error against training-embodiment count, under both protocols:

| n | A: 4000/emb → total | A held MAE | A rot | B: 60k total | B held MAE | B rot |
|---|---|---|---|---|---|---|
| 2 | 8,000 | 3.61 cm | 52.9° | 60,000 | 3.56 cm | 55.1° |
| 4 | 16,000 | 2.54 cm | 32.2° | 60,000 | 2.34 cm | 33.0° |
| 8 | 32,000 | 1.73 cm | 27.2° | 60,000 | 1.79 cm | 27.9° |
| 16 | 64,000 | 1.20 cm | 17.7° | 60,000 | 1.16 cm | 16.5° |
| 42 | 168,000 | 0.74 cm | 11.1° | 60,000 | 0.84 cm | 12.0° |

**Q2: yes, and it is diversity rather than volume.** The two protocols are indistinguishable at
every rung even though A ends with 2.8× the data of B. Held-out error falls monotonically by 4.3×
from 2 to 42 embodiments and has not flattened.

The control that makes this readable: **seen-split error moves the other way**, 0.27 → 0.46 cm as
embodiments increase. Few embodiments is not "easier", it is memorisation — the model fits its two
robots better and transfers worse. Figure: `outputs/motion_experiments.png`.

### 10.6 Experiment 3 — the motion auxiliary does NOT help, and the metric may not be able to say

Three arms, identical stock tower / batch 32 / 15k steps, differing only in `--motion-aux`. UR5e is
excluded from task training entirely, then scored. Action loss, `outputs/dp_motion/*/history.json`:

| arm | IIWA (seen) | Panda (seen) | **UR5e (HELD OUT)** | seen mean | final motion loss |
|---|---|---|---|---|---|
| A vanilla | 0.0316 | 0.0368 | **0.0355** | 0.0342 | — |
| B real motion aux | 0.0332 | 0.0379 | **0.0376** | 0.0355 | 1.61 (from 7.45) |
| C synthetic + real | 0.0331 | 0.0377 | **0.0373** | 0.0354 | 1.79 (from 4.50) |

**The auxiliary was live** — its loss fell 4.6× and 2.5× — and it still bought nothing; A is
marginally best everywhere. The same null appears in the section-9.6 runs, step-matched so the
comparison is fair:

| mode | @10k | @15k | @21.75k | @30k |
|---|---|---|---|---|
| frozen | 0.0352 | 0.0363 | 0.0349 | 0.0384 |
| finetune | 0.0351 | 0.0359 | 0.0344 | 0.0380 |
| scratch (control) | 0.0347 | 0.0357 | 0.0343 | 0.0380 |
| online | 0.0348 | 0.0360 | — | — |

*(Read single log lines with suspicion — one `online` entry showed 0.0247 and looked like a real
gain until it was step-matched. It was one noisy batch.)*

**Before concluding "synthetic motion knowledge does not transfer", notice what A vanilla says.**
It never saw a UR5e demonstration and scores **0.0355** on UR5e, against **0.0368** on Panda, which
it trained on. Held-out is not worse than seen. Whatever this metric measures, a UR5e demonstration
is worth roughly nothing on it — so it has almost no headroom in which any method could show an
improvement, and the three arms landing within 6% of each other is what an insensitive metric looks
like, not necessarily what a failed hypothesis looks like. Two controls settle it — below.

**The controls came back and they settle it — the metric cannot answer Q3.**
(`outputs/exp3_headroom.png`)

| quantity | value |
|---|---|
| UR5e action loss **with** its own demonstrations (`D_headroom`) | 0.0348 |
| UR5e action loss **without** any (`A_vanilla`) | 0.0355 |
| **total headroom — what a UR5e demonstration is worth** | **0.0007** |
| seed-to-seed spread on the same arm (`C` vs `C_s1`) | **0.0021** |
| the A-vs-C effect the null was about | 0.0018 |

**The noise is 3× the entire headroom, and larger than the effect being claimed.** No method could
have demonstrated anything here: even a perfect one could only have moved 0.0007, which is a third
of the run-to-run variation. So experiment 3 is not evidence against the hypothesis — it is a
measurement with no resolution. Report it that way; calling it a negative result would be wrong.

Second seeds also flip the seen-robot ordering (`A_vanilla` IIWA 0.0316 → 0.0361 at seed 1, Panda
0.0368 → 0.0337), which is the same warning in another form: per-robot differences at this scale
are seed artefacts.

*Why the headroom is so small* is worth understanding, because it is not a bug. The action space is
a normalised delta convention shared across all three robots and the task motion is nearly the same
for each, so predicting UR5e actions barely requires having seen UR5e. A metric that a held-out
robot already scores well on cannot show that transfer improved.

**This is the same wall as section 9.1**: action loss keeps failing to discriminate, and there is
still no task-success evaluation anywhere in this repo. Q1 and Q2 got clean answers because motion
prediction has a real metric. Q3 does not yet, and standing up a RoboCasa rollout is now the single
highest-value thing left — without it Q3 cannot be answered either way.

### 10.7 The Jaco experiments — TWO different tasks, do not conflate them

The re-export added `eef_pixel` / `eef_in_frame`. Taking it needed care: it is v2.1 while the local
trees are v3.0, and the re-exported `info.json` for the visual-robust tree declares only 3 image
features where the local one declares 6 — **Jaco among the 3 it drops**. So only the new parquet was
pulled and joined on `(episode_index, frame_index)` into the local v3.0 files, all four trees at
100% (`tools/merge_eef_pixel_columns.py`; hub limit is 1000 requests / 5 min, back off on 429).

**A loading bug came with it.** Out-of-frame rows store the pixel as arrow null LIST ELEMENTS — the
column's own `null_count` is 0, so it looks clean, and pandas hides it further by showing
`array([nan, nan])`. The `datasets` library yields `[None, None]` and `hf_transform_to_torch` dies
on the first out-of-frame row read. Invisible until a specific row is touched. Fixing it needs
pyarrow: assigning the NaN array through pandas and calling `to_parquet` silently turns every NaN
back into a null. `tools/normalise_eef_pixel_nulls.py` — run it if these trees ever fail to load.

`visual_robust_new_barx_ur5e` renders the same 108 episodes from 6 NAMED embodiments, and
`observation.state` / `eef_pixel` are shared across all of them, so the label is identical and only
the robot's appearance changes. **It finally supplies the three held-out categories the brief asked
for**, which the integer `embodiment_index` on eef_pairs cannot.

| | task | seen | held-out Jaco | baseline |
|---|---|---|---|---|
| **10.7a pixel** | ONE image → where is the gripper | 11.4 px | **14.6 px** | 55 px zero-shot |
| **10.7b motion** | TWO images → Cartesian delta | 3.30 cm | **3.89 cm** | 5.23 cm |

**(a) One image, gripper location.** Zero-shot the synthetic head scores 50–63 px on these real
frames against 3–5 px in its own domain — confirming 9.5 *directly with ground truth*, where before
it could only be inferred by fitting a camera to the head's own predictions. The failure is uniform
across all six embodiments, so it is the scene domain and not morphology, exactly as 9.5 predicted.
Trained on the four non-Jaco embodiments it reaches 11.4 px, and 14.6 px on an arm never seen — 29%
worse, not broken.

**(b) Two images, the motion between them.** This is the experiment-1 task, on real trajectories,
and it had NOT been run on Jaco before (eef_pairs has no Jaco). Translation transfers: 3.30 → 3.89
cm against a 5.23 cm predict-the-mean baseline, direction cosine 0.699 → 0.587. **But this is far
short of experiment 1's 0.67 cm / 0.980, and the two are not comparable** — 738 training pairs here
against 168,000, a world-frame target with per-episode base yaw against a clean base-frame one, 2000
steps against 6000. The gap is the setup, not Jaco.

**Rotation is not learned at all here**: 9.0–9.4° against an 8.8° identity baseline, i.e. worse than
predicting no rotation. Real rotation over 25 frames averages ~7°, far subtler than the synthetic
pairs' 60°. And novel-arm+novel-gripper (3.57 cm) beats novel-arm+seen-gripper (4.22 cm), the
opposite of the intuitive ordering, on 140 samples with no repeat — noise, not a finding.

**Done — (b) redone properly, and the answer changed.** All 108 episodes at stride 12 instead of 40
episodes at stride 40, plus 3 seeds and a synthetic-pre-training arm
(`outputs/jaco_pretrain.png`, `tools/eval_jaco_motion.py --init-from`):

| | stock tower, 3 seeds | **synthetic-144 pre-trained, 3 seeds** | seed ranges |
|---|---|---|---|
| seen translation MAE | 1.84 cm [1.74–1.94] | **1.47 cm** [1.44–1.51] | **separated** |
| seen direction cos | 0.884 [0.881–0.890] | **0.921** [0.916–0.925] | **separated** |
| held-out Jaco MAE | 2.82 cm [2.69–2.96] | 2.90 cm [2.84–2.99] | overlap |
| **held-out Jaco direction cos** | 0.712 [0.691–0.732] | **0.756** [0.738–0.782] | **separated** |
| held-out Jaco rotation | 6.0° [5.8–6.2] | 6.2° [5.8–6.6] | overlap |
| baselines | 4.89 cm / 5.4° identity rotation | | |

First, **more data was worth a lot**: the earlier 738-pair run gave 3.30 cm seen / 3.89 cm Jaco, and
this gives 1.84 / 2.82 with the same code. The binding constraint really was pair count.

**Synthetic motion pre-training transfers, but only partly, and the split is the interesting part.**
On embodiments it trains on it is unambiguous — 20% lower error with non-overlapping seed ranges.
On the *novel arm* it improves **direction** (0.712 → 0.756, ranges separated) while leaving
**magnitude** unchanged (2.82 → 2.90, ranges overlapping). So what carries across morphology is
*which way the gripper went*, not *how far*. That is consistent with 9.5's finding that the tower
encodes camera geometry strongly: the scale of a displacement in pixels depends on the arm's
apparent size and how it is framed, and those are exactly what a novel arm changes.

**Rotation is never learned in the real domain, in any of the six runs**: 5.8–6.6° against a 5.4°
identity baseline — worse than predicting no rotation at all. Real motion rotates ~5° over 25
frames, and no amount of the data available here recovers it. Any claim about rotation transfer is
unsupported; that is now checked with 6 runs rather than 1.

The earlier oddity that novel-arm+novel-gripper beat novel-arm+seen-gripper survives at 3 seeds
(JacoOmron 2.65 ± 0.11 vs JacoOmronPandaGripper 3.00 ± 0.11 for the stock tower), so it is real
rather than noise — but the two Jaco variants differ in gripper only, so "novel gripper is easier"
is not a sensible reading. More likely the Panda gripper on a Jaco arm is the visually stranger
object of the two.

### 10.8 Would more data help? Which axis you add matters more than how much

Three different things get called "more data" here and the measured answers differ:

| axis | measured? | answer |
|---|---|---|
| more **samples** (same poses, same embodiments) | yes, 10.5 | **no** — protocol A carries 2.8× protocol B's data and the curves are identical at every rung |
| more **embodiments** | yes, 10.5 | **yes** — 4.3× from 2 to 42, not flattened |
| more **poses / motion vocabulary** | not yet | likely yes, and it removes a bottleneck flagged twice |

**Pose count is a diversity axis, not a volume axis** — that is why it is expected to behave like the
embodiment axis rather than the sample axis. `56combo_144_bg12_closed`
(`ChiefJang/visual_robust_barx`) triples the poses, and reading the parquet alone shows what that
buys before a single image is decoded:

| | 48 poses | 144 poses | ratio |
|---|---|---|---|
| pairs within 0.15 m | 1,262 | 7,772 | 6.2× |
| … and rotation ≤ 45° | 458 | 4,048 | 8.8× |
| … and rotation ≤ 30° (10.6's cap) | **270** | **2,100** | **7.8×** |
| … and rotation ≤ 20° | 96 | 946 | 9.9× |
| … and rotation ≤ 10° (real-trajectory scale) | **18** | **186** | 10.3× |
| workspace span x/y/z (m) | 0.26 / 0.096 / 0.258 | 0.296 / 0.160 / 0.445 | z 1.7×, y 1.7× |

**The gain is largest exactly where the constraint was tightest.** Section 10.3 had to cap synthetic
rotation at 30° to resemble real trajectories, and that left 270 pose pairs — the reason it was
recorded as "costs vocabulary". It becomes 2,100. The ≤10° regime that actually matches real motion
(~9°) goes from 18 pairs, which is unusable, to 186. And the poses are not merely denser: the
workspace is 1.7× taller and wider.

**BLOCKED: the 144 subset has no `videos/` on the hub.** `data/`, `meta/` and a partial
`raw_images/` (99 PNGs of episode 0 only) are there; the mp4s are not, while
`56combo_48_bg12_closed_furniture` in the same repo does have them. 231,678 frames are declared and
none can be decoded. `tools/`-side everything is ready; a watcher polls every 20 min
(`outputs/logs/watch144.log`) and will pull, convert and cache automatically.

Also note it is **closed-gripper only**, so gripper-change supervision needs the matching
`_open` / `_furniture` subsets at 144 before experiment 3's full objective can use it.

**RESULT: pose diversity helps, and it helps a lot — but only a common-eval comparison shows it.**
42 embodiments and 60,000 samples held fixed, pose count subsampled WITHIN `56combo_144_bg12_closed`
so no cross-export render difference can confound it. Figure: `outputs/pose_ladder.png`.

| poses trained on | scored on its OWN pose set | **scored on ONE common pool** | dir cos | rotation |
|---|---|---|---|---|
| 48 | 0.60 cm | **2.18 cm** | 0.851 | 23.6° |
| 96 | 0.74 cm | **1.13 cm** | 0.953 | 16.0° |
| 144 | 0.84 cm | **0.83 cm** | 0.978 | 14.3° |
| predict-the-mean | — | 4.90 cm | — | — |

**The raw column runs the wrong way and is a trap.** Each ladder run was scored on pairs drawn from
its own pose subset, so the 48-pose run was tested on a far easier motion distribution — 936
distinct pose pairs against 4000, over a narrower range of rotations. Read raw, it says "more poses
is worse". It is measuring task difficulty, not model quality. This is the same shape of confound as
the embodiment ladder's shrinking held-out set (9.4), and it is worth expecting: **any ladder that
varies the data also varies the test set unless you force it not to.**

Scored on one common pool — 4000 pairs from all 144 poses, on held-out embodiments,
`tools/eval_pose_ladder_common.py` — the ordering reverses and the effect is large: **2.6× better
translation from 48 to 144 poses**, direction cosine 0.851 → 0.978, rotation 23.6° → 14.3°. Pose
diversity behaves like the embodiment axis, not like the sample axis, which is what 10.8 predicted.

Note the 144-pose number is nearly identical in both columns (0.84 vs 0.83) — as it must be, since
for that run "its own pose set" and "the common pool" are the same thing. That agreement is the
check that the two evaluations are otherwise comparable.

### 10.9 LAP-style language ground truth for the motion task (`motion_language.py`)

**Why**: 10.7b found the regression head transfers *direction* to a novel arm but not *magnitude* —
direction cosine 0.712 → 0.756 with seed ranges separated, while MAE 2.82 → 2.90 overlaps. Pixel
displacement scale depends on how large the arm looks and how it is framed, which is exactly what a
new morphology changes. Language quantises magnitude into three buckets — a coarser claim that may
survive an appearance change when centimetres do not. That is the thing worth testing, and it is
also what would let the **VLM** (362M text model) learn this task rather than only the 86M tower.

```
"move forward moderately, move right slightly, turn left slightly, close gripper"
```

**The label set is healthy, which is not automatic.** Section 8 records the earlier LAP objective
saturating — content accuracy 1.000 by 15k steps on 7.72 bits, after which the VLM got no gradient
at all. These labels are far richer because they describe a free 25-frame displacement rather than
an action chunk:

| corpus | labels | distinct | entropy | top-1 share |
|---|---|---|---|---|
| earlier LAP (§8) | — | 525 | 7.72 bits | 2.7% |
| synthetic, closed only | 6,000 | 4,095 | **11.78 bits** | 0.2% |
| **synthetic, closed + open** | 8,000 | **6,152** | **12.44 bits** | **0.1%** |
| real trajectories | 4,334 | 2,718 | 10.93 bits | 1.3% |

Use **closed + open**: the two subsets carry byte-identical pose values (verified, max diff 0.0) and
differ only in gripper, so pairing across them is what produces the `open/close gripper` words at
all. Closed alone yields 13 content words and no gripper vocabulary; together, 15 words with the
gripper mentioned in 50.6% of labels.

**Frames — direction words are only meaningful in a frame the model can see.** eef_pairs already
uses one fixed frame across every embodiment, so `frame="fixed"`. The real trajectories store WORLD
xyz with the base parked differently per episode (1.52 m of cross-episode spread in x, and net
motion that flips sign), so world-frame words would be noise; `frame="eef"` rotates the displacement
into the gripper's own frame at time t, which is episode-independent and visible in the image.

**Translation transfers, rotation does not — and the thresholds say so before any training.**
Fitted as quantiles of each corpus independently:

| | synthetic | real | |
|---|---|---|---|
| translation idle / slight / moderate | 0.028 / 0.073 / 0.110 m | 0.026 / 0.072 / 0.107 m | **agree to 7%** |
| rotation idle / slight / moderate | 9.7° / 29.2° / 63.8° | 1.0° / 3.9° / 8.8° | **7× apart** |

The translation buckets landing on top of each other means "move forward moderately" denotes the
same physical displacement in both corpora — so a VLM pre-trained on synthetic translation language
is answering the same question on real data. The rotation buckets do not: "roll left moderately"
means 29° synthetically and 3.9° really. That is 10.3's rotation mismatch reappearing, and it lines
up with 10.7b measuring rotation never being learned in the real domain at all (6/6 runs worse than
predicting no rotation). **Treat the rotation words as untrusted for transfer**; the translation and
gripper words are the ones with evidence behind them.

```bash
python tools/preview_motion_language.py     # label stats, thresholds and vocabulary agreement
```

### 10.10 What is wrong with the eef_pairs dataset itself

Full spec with every supporting number: `outputs/dataset_redesign_spec.txt`. Ranked by what the
measurements say is actually blocking, not by what looks wrong:

| # | problem | the measurement | fix |
|---|---|---|---|
| **1** | **only 4 fixed camera poses** | gripper lands in 4 tight clusters ~200 px apart, each 12–22 px wide; a 40 px shift moves features **31%** toward an unrelated image while a different *robot* moves them 4% | randomise camera pose per episode; export extrinsics/intrinsics |
| 2 | poses cover the workspace, not trajectory segments | pairs within 0.15 m still average **48°** rotation; real is 6.5° | sample s_t then a LOCAL delta_eef, as the brief said |
| 3 | rotation is below visual resolution | 5° moves the gripper tip **0.87 px** = 1/11 of a patch; relative error 24% synthetic vs 95% real | higher resolution, or a wrist camera, or a longer horizon |
| 4 | gripper is binary and encoded as a *subset* | 2 values; forces cross-subset pairing to get any gripper signal | continuous `gripper` column in one subset |
| 5 | empty scene vs cluttered kitchen | a policy frame sits at **102%** of the unrelated-image distance | render into RoboCasa-like scenes with objects |
| 6 | `embodiment_index` has no name mapping | — | ship `meta/embodiments.json` → {arm, gripper} |

**Keep**: every embodiment posed at IDENTICAL EEF states. That is what makes embodiment identity
carry zero information about the motion label — the shortcut is impossible by construction rather
than by sampling luck. When moving to sampled pairs, draw the pose pairs ONCE and render every
embodiment at them.

**If only one thing changes, randomise the camera.** Every measurement points there, and the axes
one would intuitively add first — more embodiments, more backgrounds — are already flat at 1–4%.

### 10.11 Known limits of this design

* **The three held-out categories the brief asks for cannot be built yet.** `embodiment_index` has
  no name mapping in the export, so "novel arm + seen gripper" vs "seen arm + novel gripper" cannot
  be separated and the split is random. Section 9.2 — still the cheapest thing to ask for.
* **Experiment 3's held-out robot is one of three.** The barx corpus has IIWA / Panda / UR5e only,
  so holding one out leaves two. `visual_robust_new_barx_ur5e` has 6 *named* embodiments over the
  same episodes and is the better held-out set once its re-export lands.
* **Still no task success anywhere.** Everything is action loss (section 9.1).
