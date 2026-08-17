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

What this says:

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
* The CE trains the VLM including `lm_head` and the vision tower.
* Overfitting one synthetic batch: CE 45.1 → 0.016, token accuracy 0 → 1.00 in 150 steps.

Real data, barx front-only, batch 64 x 2 GPUs, 300 steps: CE **9.50 → 5.62**, token accuracy
0.068, `fast_tokens_mean` **172** ids per 50x12 chunk (max 243, nothing truncated at the 256 cap),
flow-matching loss 0.22-0.25. Both terms move; nothing NaNs; checkpointing works.

### Cost — this is the part that changes how you launch it

The postfix adds ~172 tokens to a ~163-token sequence, and attention here is eager (it materializes
`B x heads x L x L` in float32), so memory grows with the square. Measured peak on one H100,
2-GPU DDP, front camera only:

| run | per-rank batch | peak GPU mem | s/step |
|---|---|---|---|
| baseline | 64 | 36.6 GiB | 0.387 |
| KI | 32 | 56.4 GiB | 0.317 |
| KI | 48 | 70.3 GiB | 0.450 |
| KI | 64 | **79.7 GiB of 79.7 available** | 0.576 |

Batch 64 completed 100 steps but sits at 98% of the card — one longer-than-usual postfix would
OOM it, and it cannot share GPUs with anything. **Use `BATCH_SIZE=32`** (effective 256 on 8 GPUs,
half the 512 the section-4 baselines used — say so when comparing) or 48 if the box is otherwise
idle. If the effective batch has to stay at 512, the cheap win available is `masked_fill_` instead
of `torch.where` in `eager_attention_forward`, which drops one full `L x L` float32 tensor per
layer; it is numerically identical but touches the path every existing run uses, so it was left
alone here.

`ki_fast_max_tokens` (default 256) caps the postfix; `fast_truncated_frac` in the metrics tells
you if it is biting. Truncation costs supervision on the tail of the chunk, nothing else.

### Not done

* **Step 8, the full run.** Everything is in place and smoke-tested; no 50k-step job has been
  launched.
* RA-BC per-sample weighting with knowledge insulation raises `NotImplementedError` — the CE is a
  scalar over tokens, not a per-sample loss. Nothing needs it today.
* Chunks that run past the end of an episode are dropped from the token loss (they are padded with
  repeated frames, and one token stream cannot mask per timestep the way the flow loss does). The
  key that flags them, `actions_id_pad`, is not currently produced by this dataset path, so that
  guard is untested in practice.
