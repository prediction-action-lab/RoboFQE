"""FQE with pretrained vision encoders (frozen backbone + trainable head).

Currently supported encoders:
  - DINOv3 (ViT-S/B/L; default: vits16, ~22M params, frozen)

Future plug-ins (drop in `encoders.py`, expose via `build_pretrained_encoder`):
  - VC-1, R3M, SigLIP, CLIP, ...

Same algorithm + temporal aggregator + Q-head as `fqe.scratch`; only the
per-camera image encoder is swapped.
"""
from .encoders import DINOv3ImageEncoder, build_pretrained_encoder
from .networks import VisionQNetworkPretrained
from .algorithm import FittedQEvaluationVisionPretrained

__all__ = [
    "DINOv3ImageEncoder",
    "build_pretrained_encoder",
    "VisionQNetworkPretrained",
    "FittedQEvaluationVisionPretrained",
]
