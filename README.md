# RoboFQE

Vision-based Fitted Q Evaluation (FQE) for off-policy evaluation of LIBERO BC
policies. Given a target BC policy and an offline dataset (the suite's
demonstrations plus augmented rollouts of other behavior policies, labeled by
an LRM success detector), FQE fits Q^π from frame-stacked camera observations
and reports the direct-method (DM) estimate V̂^π = E_{s₀, a∼π}[Q(s₀, a)].

Two variants share the same dataset, policy wrapper, trainer, temporal
aggregator and Q-head; they differ only in the per-camera image encoder:

| Variant | Image encoder | Module | Entry point |
|---|---|---|---|
| From scratch | ResNet-18, `weights=None` | `fqe.scratch` | `scripts/train_fqe_scratch.py` |
| DINOv3 | pretrained DINOv3 ViT (frozen / LoRA / partial FT) | `fqe.pretrained` | `scripts/train_fqe_dinov3.py` |

## Architecture

For each of the H = 10 frames in the stack:

1. Each camera (`agentview_rgb`, `eye_in_hand_rgb`) → image encoder → linear projection to 128-d
2. Concat with low-dim proprio (`gripper_states`, `joint_states`) → `frame_proj` → 256-d token, plus a learnable positional embedding

Then a 2-layer Transformer encoder (4 heads, FF 512, GELU, pre-norm) runs over the H
tokens. The last token is concatenated with the BERT task embedding and the action,
and a 2-layer MLP (256, 256) outputs a scalar Q. The final layer is initialised with
orthogonal(gain=0.01) so Q starts near 0 under the sparse 0/1 reward.

FQE update: MSE TD loss, soft target update (τ = 0.005), target clamped to
[0, 1], gradient clipping. Next actions are sampled from the target policy
(`LIBEROPolicyWrapper`, which feeds the explicit frame stack to the BC
policy instead of its stateful latent queue).

DINOv3 modes (`--mode`):

- `frozen`: backbone frozen end-to-end (forward under `no_grad`). Only the projection, temporal Transformer and head train (~1.5M params with ViT-S/16).
- `lora`: base weights frozen. peft adds rank-r LoRA adapters (default `q_proj,k_proj,v_proj`), trained by the same TD loss.
- `partial`: patch embedding and the bottom N−K blocks frozen. The top K blocks, final LayerNorm and pooler train at `--backbone-lr`.

Supported encoder tags: `dinov3_vits16` (default), `dinov3_vitb16`,
`dinov3_vitl16` (patch 16, 128² native), and `dinov2_vits14/vitb14/vitl14`
(patch 14, resized to 224² internally).

## Layout

```
fqe/
├── base.py             BaseOPEAlgorithm, PolicyWrapper, set_seed
├── data/libero.py      LIBERO HDF5 (demos + augmented rollouts) → frame-stacked transitions + BERT task_emb
├── rollouts.py         behavior policies (BC / noisy BC / random), LIBERO episodes → demo-format HDF5
├── lrm.py              LRM success detector: prompt, answer format, loading (labeling + fine-tuning)
├── policy.py           LIBEROPolicyWrapper (BC ckpt → π(a | frame stack, task_emb))
├── trainer.py          OPETrainer (train loop, periodic DM eval, checkpoints)
├── libero_patches.py   runtime patch so no-proprio BC ckpts load on upstream LIBERO
├── paths.py            experiments/ directory resolver
├── scratch/            from-scratch ResNet-18 Q-net + FQE
└── pretrained/         DINOv3 encoders + Q-net + FQE
scripts/
├── collect_rollouts.py     offline-dataset augmentation (rollouts of mixed behavior policies)
├── finetune_lrm.py         LoRA fine-tune of LRM's completion model on the official demos
├── label_rewards_lrm.py    LRM terminal-success labels → rollout `rewards`
├── train_fqe_scratch.py
└── train_fqe_dinov3.py
```

## Setup

```bash
git clone git@github.com:prediction-action-lab/RoboFQE.git && cd RoboFQE

# LIBERO, installed editable from ./LIBERO (gitignored).
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
touch LIBERO/libero/__init__.py   # needed for `import libero` with the editable install

conda env create -f environment.yml
conda activate robofqe
```

The DINOv3 weights (`facebook/dinov3-*`) are gated on Hugging Face. Request
access on the model page, then `export HF_TOKEN=<token>`.

## Data and target policies

The scripts resolve paths with `fqe.paths.get_paths`, rooted at `$ROBOFQE_ROOT`
(default: the repo root):

```
experiments/
├── _shared/bert/                          bert-base-cased cache (auto-downloaded)
├── _shared/lrm_libero_lora/               LRM LoRA adapter (scripts/finetune_lrm.py)
└── libero_{suite}/                        suite ∈ spatial, object, goal, 10, 90
    ├── data/libero_{suite}/*_demo.hdf5    official LIBERO demos (128×128)
    ├── data/libero_{suite}_rollouts/      augmented rollouts, *_rollouts.hdf5 (see below)
    ├── bc_ckpts/{policy_config}/          LIBERO multitask BC ckpts, e.g.
    │   └── multitask_model_ep50.pth         img128_noprop/multitask_model_ep50.pth
    └── fqe_results/                       written by the training scripts
```

BC checkpoints are those produced by LIBERO's `libero/lifelong/main.py`
(multitask `BCTransformerPolicy`). They must embed `cfg` with `shape_meta`.
Checkpoints trained without proprio (`img128_noprop`) are supported through
`fqe.libero_patches.apply_no_proprio_patch`, which both scripts apply.

## Offline dataset augmentation

The official demos are all successful, so FQE also trains on rollouts of other
behavior policies. Without their failure transitions, demo-only FQE
overestimates (paper App. A.4). The paper's LIBERO data adds 25 rollouts per
task from partially trained BC checkpoints. The world-model baselines are
post-trained on the same data, so `collect_rollouts.py` accepts any mix of
behavior policies to broaden state-action coverage:

```bash
export MUJOCO_GL=egl

# 1) Collect. One spec per behavior policy: NAME[@NOISE][=COUNT]
#      NAME   random (uniform actions in [-1, 1]^7), or a BC ckpt in
#             bc_ckpts/{policy_config}/ (or a path), e.g. LIBERO's multitask_model_ep{N}.pth
#      @NOISE Gaussian action-noise std (BC only)
#      =COUNT rollouts per task (all specs or none; otherwise --n-rollouts is split evenly)
#    paper setting: 25 rollouts per task from partially trained BC checkpoints
python scripts/collect_rollouts.py --suite spatial --policy-config img128_noprop \
    --policies multitask_model_ep5.pth multitask_model_ep15.pth multitask_model_ep20.pth
#    wider coverage (e.g. for the world models), kept in its own rollout dir
#    (train FQE on it with --rollout-dir)
python scripts/collect_rollouts.py --suite spatial --policy-config img128_noprop \
    --policies multitask_model_ep5.pth=10 multitask_model_ep20.pth=10 \
               multitask_model_ep20.pth@0.3=10 random=10 --init-states reset \
    --out-dir experiments/libero_spatial/data/libero_spatial_rollouts_coverage

# 2) Fine-tune LRM's completion model on the official demos (LoRA)
python scripts/finetune_lrm.py --suites spatial object goal 10

# 3) Label terminal success: writes `rewards`, reports agreement with the simulator
python scripts/label_rewards_lrm.py --suite spatial --adapter experiments/_shared/lrm_libero_lora
```

- Episodes follow LIBERO's own evaluator. They start from the benchmark init
  states (`--init-states reset` samples fresh ones), take 5 zero-action steps,
  then run the BC policy's stateful `get_action` with success = env `done`.
  The horizon is 220 / 280 / 300 / 520 steps for Spatial / Object / Goal / Long.
  Simulator workers use the `spawn` start method; forked workers fail to create
  EGL contexts.
- Each task gets a `{task}_rollouts.hdf5` in the official demo layout,
  including the per-episode `model_file` (MJCF) for re-rendering from
  `states`. It keeps `create_dataset.py`'s alignment, where `obs[t]` is the
  observation after `actions[t]`, so the last frame is the terminal
  observation the detector scores. Simulator success is kept in `sim_rewards`
  and the `success` attr. The FQE loader attaches each file to its demos' task
  by file name.
- The success detector is the Task Completion model of
  [LRM](https://huggingface.co/USC-PSI-Lab/LRM-models) (Qwen3-VL-8B, ~17.5 GB in
  bf16, public). It judges the terminal agentview frame with the model-card
  prompt, after rotating the frame 180°. That is the OpenVLA / LeRobot LIBERO
  orientation and the one in which LIBERO's "left" / "right" match the image.
  LIBERO's success predicates do not match the off-the-shelf model well; for
  example, a bowl on the plate counts as success while the gripper still holds
  it. The paper therefore labels with LRM fine-tuned on the official LIBERO
  demos (97.4% agreement with the simulator). `finetune_lrm.py` trains a LoRA
  on the completion model's language layers. Frames from the last 5 steps of
  each demo are labeled "yes" (demos end after a 10-step success hold), and
  frames from its first 80% are labeled "no". Pass the adapter with
  `--adapter`, or merged weights with `--model /path/to/dir`.
- FQE training reads demos + labeled rollouts by default. `--demos-only` is the
  demo-only ablation, and `--rollout-reward-key sim_rewards` trains on
  simulator ground truth instead of LRM labels.

## Running

```bash
export MUJOCO_GL=egl

# From-scratch ResNet-18
python scripts/train_fqe_scratch.py \
    --suite 10 --policy-config img128_noprop --ckpt-name multitask_model_ep50.pth

# DINOv3 frozen
python scripts/train_fqe_dinov3.py --mode frozen \
    --suite 10 --policy-config img128_noprop --encoder dinov3_vits16

# DINOv3 + LoRA
python scripts/train_fqe_dinov3.py --mode lora --lora-r 16 --lora-alpha 32 ...

# DINOv3 partial fine-tune (top 4 blocks)
python scripts/train_fqe_dinov3.py --mode partial \
    --n-finetune-blocks 4 --backbone-lr 1e-5 --lr 1e-4 ...
```

Shared defaults: 50k steps, batch 64, lr 1e-4, γ = 0.99, seed 42, demos +
LRM-labeled rollouts. Use `--gamma 1.0` for undiscounted return (V^π = success
rate), `--demos-only` / `--rollout-reward-key sim_rewards` for the data
ablations. Checkpoints go to
`fqe_results/{method_tag}/{policy_config}/{data}/{ckpt}_gamma{γ}_seed{seed}/`
(`{data}` ∈ `aug`, `aug-sim_rewards`, `demos`, with `@{rollout dir name}`
appended for a non-default `--rollout-dir`), and the final DM estimate is
printed at the end of training.
