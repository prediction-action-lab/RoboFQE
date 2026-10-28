# RoboFQE

Vision-based Fitted Q Evaluation (FQE) for off-policy evaluation of LIBERO BC
policies. Given a target BC policy and the suite's demonstration data, FQE
fits Q^π from frame-stacked camera observations and reports the direct-method
(DM) estimate V̂^π = E_{s₀, a∼π}[Q(s₀, a)].

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
├── data/libero.py      LIBERO HDF5 → frame-stacked transitions + BERT task_emb
├── policy.py           LIBEROPolicyWrapper (BC ckpt → π(a | frame stack, task_emb))
├── trainer.py          OPETrainer (train loop, periodic DM eval, checkpoints)
├── libero_patches.py   runtime patch so no-proprio BC ckpts load on upstream LIBERO
├── paths.py            experiments/ directory resolver
├── scratch/            from-scratch ResNet-18 Q-net + FQE
└── pretrained/         DINOv3 encoders + Q-net + FQE
scripts/
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
└── libero_{suite}/                        suite ∈ spatial, object, goal, 10, 90
    ├── data/libero_{suite}/*_demo.hdf5    official LIBERO demos (128×128)
    ├── bc_ckpts/{policy_config}/          LIBERO multitask BC ckpts, e.g.
    │   └── multitask_model_ep50.pth         img128_noprop/multitask_model_ep50.pth
    └── fqe_results/                       written by the training scripts
```

BC checkpoints are those produced by LIBERO's `libero/lifelong/main.py`
(multitask `BCTransformerPolicy`). They must embed `cfg` with `shape_meta`.
Checkpoints trained without proprio (`img128_noprop`) are supported through
`fqe.libero_patches.apply_no_proprio_patch`, which both scripts apply.

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

Shared defaults: 50k steps, batch 64, lr 1e-4, γ = 0.99, seed 42. Use
`--gamma 1.0` for undiscounted return (V^π = success rate). Checkpoints go to
`fqe_results/{method_tag}/{policy_config}/{ckpt}_gamma{γ}_seed{seed}/`, and
the final DM estimate is printed at the end of training.

## Attribution

The from-scratch Q-network and FQE algorithm, the LIBERO dataset loader,
policy wrapper and trainer were written by Sangwoo Shin
(<jsw7460@gmail.com>) in RoboOPE commit `6c2c61a` (*feat: vision-based FQE
and FQE-Cat for LIBERO benchmark*). They are ported here with only import
paths rewritten. File-level docstrings record the provenance.
