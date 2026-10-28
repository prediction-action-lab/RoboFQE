"""From-scratch vision FQE: per-camera ResNet-18 (`weights=None`) + Transformer
temporal aggregator + task_emb-conditioned Q-head.

Authored by Sangwoo Shin (<jsw7460@gmail.com>) in RoboOPE main commit 6c2c61a
(*feat: vision-based FQE and FQE-Cat for LIBERO benchmark*); ported as-is,
only import paths rewritten.
"""
from .networks import VisionQNetwork, ResNet18ImageEncoder
from .algorithm import FittedQEvaluationVision

__all__ = [
    "VisionQNetwork",
    "ResNet18ImageEncoder",
    "FittedQEvaluationVision",
]
