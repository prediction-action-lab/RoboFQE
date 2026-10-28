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

`fqe.scratch` / `fqe.pretrained` are not imported here so that each variant's
dependencies (torchvision vs. transformers / peft) load only when used.
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
