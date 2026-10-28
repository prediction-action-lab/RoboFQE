# RoboFQE: Rollout-Free Evaluation of Visual Robot Policies from Offline Trajectories

[**Project Page**](https://yfdeng.com/robofqe/) | [**arXiv**](https://arxiv.org/abs/2610.18699)

Yunfu Deng, Sangwoo Shin, Brahma Pavse, Josiah Hanna

University of Wisconsin–Madison

**Conference on Robot Learning (CoRL) 2026**

This is the official code of RoboFQE. RoboFQE estimates the discounted return
of a visual robot policy from offline trajectories, without rolling the policy
out. It combines fitted Q-evaluation (FQE) with a pretrained DINOv3 encoder and
labels terminal success with a learned success detector (LRM). This release
covers the LIBERO experiments.

## Installation

```bash
git clone https://github.com/prediction-action-lab/RoboFQE.git && cd RoboFQE

# LIBERO, installed in editable mode from ./LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
touch LIBERO/libero/__init__.py   # needed for `import libero` with the editable install

conda env create -f environment.yml
conda activate robofqe
export MUJOCO_GL=egl
```

The DINOv3 weights (`facebook/dinov3-*`) are gated on Hugging Face. Request
access on the model page, then `export HF_TOKEN=<token>`.

## Data and policies

Paths are resolved under `$ROBOFQE_ROOT` (default: the repo root):

```
experiments/
├── _shared/                               BERT cache, LRM adapter (step 2)
└── libero_{suite}/                        suite ∈ spatial, object, goal, 10 (LIBERO-Long)
    ├── data/libero_{suite}/*_demo.hdf5    official LIBERO demos
    ├── data/libero_{suite}_rollouts/      augmented rollouts (step 1)
    ├── bc_ckpts/{policy_config}/          BC checkpoints, e.g. img128_noprop/multitask_model_ep50.pth
    └── fqe_results/                       FQE runs (step 4)
```

- **Demos**: download them with LIBERO's
  `benchmark_scripts/download_libero_datasets.py` and put each suite folder
  (e.g. `libero_spatial/`) under `experiments/libero_{suite}/data/`.
  `libero_10` is part of `--datasets libero_100`.
- **Policies**: target and behavior policies are LIBERO multitask BC policies
  (`policy=bc_transformer_policy lifelong=multitask` in
  `libero/lifelong/main.py`), which saves `multitask_model_ep{N}.pth`
  checkpoints. Checkpoints must embed `cfg` with `shape_meta`. Checkpoints
  trained without proprioception (e.g. `img128_noprop`) are supported.

## Usage

### 1. Collect rollouts

The paper's offline dataset adds 25 rollouts per task from partially trained
BC checkpoints to the 50 demos per task:

```bash
python scripts/collect_rollouts.py --suite spatial --policy-config img128_noprop \
    --policies multitask_model_ep5.pth multitask_model_ep15.pth multitask_model_ep20.pth
```

Each `--policies` entry is `NAME[@NOISE][=COUNT]`: a checkpoint in
`bc_ckpts/{policy_config}/` (or a path) or `random`, an optional Gaussian
action-noise std, and an optional number of rollouts per task (otherwise
`--n-rollouts`, default 25, is split evenly). For example, a dataset with wider
state-action coverage, in its own directory:

```bash
python scripts/collect_rollouts.py --suite spatial --policy-config img128_noprop \
    --policies multitask_model_ep5.pth=10 multitask_model_ep20.pth@0.3=10 random=10 \
    --init-states reset --out-dir experiments/libero_spatial/data/libero_spatial_rollouts_coverage
```

Each task gets a `{task}_rollouts.hdf5` in the LIBERO demo format, with the
simulator success in `sim_rewards`.

### 2. Fine-tune the success detector

LoRA fine-tuning of the [LRM](https://huggingface.co/USC-PSI-Lab/LRM-models)
task-completion model (Qwen3-VL-8B) on the official demos. The adapter is saved
to `experiments/_shared/lrm_libero_lora`.

```bash
python scripts/finetune_lrm.py --suites spatial object goal 10
```

### 3. Label terminal rewards

```bash
python scripts/label_rewards_lrm.py --suite spatial --adapter experiments/_shared/lrm_libero_lora
```

LRM judges the last frame of every rollout. The label is written to `rewards`,
and the agreement with the simulator is reported in `lrm_labels.json` in the
rollout directory. `--dry-run` only reports.

### 4. Train FQE

`--ckpt-name` is the target policy in `bc_ckpts/{policy_config}/`.

```bash
# RoboFQE (DINOv3 encoder)
python scripts/train_fqe_dinov3.py --suite spatial --policy-config img128_noprop \
    --ckpt-name multitask_model_ep50.pth --mode partial

# RoboFQE-scratch ablation (ResNet-18 trained from scratch)
python scripts/train_fqe_scratch.py --suite spatial --policy-config img128_noprop \
    --ckpt-name multitask_model_ep50.pth
```

Training uses the demos and the labeled rollouts, with defaults of 50k steps,
batch 64, lr 1e-4, γ = 0.99 and seed 42. The DM estimate of the target
policy's discounted return is printed at the end of training. Checkpoints are
saved to
`experiments/libero_{suite}/fqe_results/{method}/{policy_config}/{data}/{ckpt}_gamma{γ}_seed{seed}/`.

| Option | |
|---|---|
| `--mode` | DINOv3 backbone: `partial` (default, the paper's setting: top `--n-finetune-blocks 4` blocks at `--backbone-lr 1e-5`), `lora` (`--lora-r 16 --lora-alpha 32`), or `frozen` |
| `--encoder` | `dinov3_vits16` (default), `dinov3_vitb16`, `dinov3_vitl16`, `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14` |
| `--demos-only` | train on the demos alone (`{data}` = `demos`; default `aug`) |
| `--rollout-reward-key sim_rewards` | simulator success instead of LRM labels (`aug-sim_rewards`) |
| `--rollout-dir` | another rollout directory (adds `@{dir name}` to `{data}`) |
| `--gamma 1.0` | undiscounted return (= success rate) |

## Acknowledgement

Our code builds on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO),
[LRM](https://huggingface.co/USC-PSI-Lab/LRM-models) and
[DINOv3](https://github.com/facebookresearch/dinov3).

## License

The code is released under the [MIT license](LICENSE). Model weights downloaded
from Hugging Face (DINOv3, LRM) are subject to their own licenses.

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{deng2026robofqe,
  title={RoboFQE: Rollout-Free Evaluation of Visual Robot Policies from Offline Trajectories},
  author={Deng, Yunfu and Shin, Sangwoo and Pavse, Brahma and Hanna, Josiah P},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2026},
  organization={PMLR}
}
```
