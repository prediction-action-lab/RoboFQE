"""Vision FQE for LIBERO BC policies.

Two Q-network variants share the same dataset / policy wrapper / trainer and
the same temporal aggregator + Q-head; they differ only in the per-camera
image encoder:

    fqe.scratch     — ResNet-18 trained from scratch (`FittedQEvaluationVision`)
    fqe.pretrained  — pretrained DINOv3 backbone, frozen / LoRA / partial FT
                      (`FittedQEvaluationVisionPretrained`)

Shared LIBERO plumbing:
    fqe.data        — LIBEROOPEDataset, libero_to_ope
    fqe.policy      — LIBEROPolicyWrapper
    fqe.trainer     — OPETrainer
    fqe.base        — BaseOPEAlgorithm, PolicyWrapper, set_seed

Offline dataset:
    fqe.rollouts    — behavior policies + LIBERO episodes → demo-format HDF5
                      (scripts/collect_rollouts.py)
    fqe.lrm         — LRM success detector for the terminal rewards
                      (scripts/finetune_lrm.py, scripts/label_rewards_lrm.py)

`fqe.scratch` / `fqe.pretrained` / `fqe.rollouts` / `fqe.lrm` are not imported
here so that their dependencies (torchvision vs. transformers / peft vs. the
LIBERO simulator) load only when used.
"""
from .base import BaseOPEAlgorithm, PolicyWrapper, set_seed
from .data import libero_to_ope, LIBEROOPEDataset
from .policy import LIBEROPolicyWrapper
from .trainer import OPETrainer

__all__ = [
    "BaseOPEAlgorithm",
    "PolicyWrapper",
    "set_seed",
    "libero_to_ope",
    "LIBEROOPEDataset",
    "LIBEROPolicyWrapper",
    "OPETrainer",
]
