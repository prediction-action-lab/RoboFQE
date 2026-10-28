"""Shared path resolver for the training scripts.

Each LIBERO experiment lives under
    $ROBOFQE_ROOT/experiments/libero_{suite}/{data, bc_ckpts, fqe_results}/
and shared per-tool caches under
    $ROBOFQE_ROOT/experiments/_shared/{bert, ...}/

`ROBOFQE_ROOT` defaults to the repo root.

`get_paths(suite, policy_config)` returns a dict of absolute paths so each
training script doesn't have to duplicate the directory math.

`policy_config` is the BC-policy variant subdir name (e.g. ``img128_prop``,
``img128_noprop``). Empty string means a flat ``bc_ckpts/`` layout.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


ROOT = Path(os.environ.get("ROBOFQE_ROOT", Path(__file__).resolve().parent.parent))

VALID_SUITES = ("spatial", "object", "goal", "10", "90")


def get_paths(suite: str, policy_config: str = "") -> Dict[str, Path]:
    if suite not in VALID_SUITES:
        raise ValueError(f"suite must be one of {VALID_SUITES}, got {suite!r}")
    suite_root = ROOT / f"experiments/libero_{suite}"
    bc_dir = suite_root / "bc_ckpts"
    if policy_config:
        bc_dir = bc_dir / policy_config
    return {
        "root": ROOT,
        "suite_root": suite_root,
        "dataset_dir": suite_root / "data" / f"libero_{suite}",
        "bc_ckpt_dir": bc_dir,
        "fqe_results": suite_root / "fqe_results",
        "bert_cache": ROOT / "experiments" / "_shared" / "bert",
    }
