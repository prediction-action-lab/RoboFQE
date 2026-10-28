"""Label augmented LIBERO rollouts with LRM terminal-success rewards.

Paper Sec. 3.3 / 4.1: each trajectory gets one terminal label
r̂_i = R̂(o_T, l_i) ∈ {0, 1}, shared by all target policies. The augmented
LIBERO rollouts are labeled by LRM fine-tuned on the official demos
(scripts/finetune_lrm.py; detector details in fqe.lrm); the demos keep their
ground truth.

For every `{task}_rollouts.hdf5` of collect_rollouts.py, LRM judges the
terminal frame `obs/{image_key}[-1]` (o_T, see fqe.rollouts) against the task
instruction, and the label is written as `data/demo_k/rewards` (r̂ at the
last step, 0 elsewhere: the LIBERO reward layout LIBEROOPEDataset reads),
with the raw answer in the `lrm_response` attr. Simulator success is only
used to report the detector's agreement with it. Files already labeled with
the same detector setup are skipped, so collections can be labeled
incrementally.

Run:
    # LRM fine-tuned on the LIBERO demos (LoRA from scripts/finetune_lrm.py)
    python scripts/label_rewards_lrm.py --suite spatial \\
        --adapter experiments/_shared/lrm_libero_lora
    # ... or merged weights in a local dir
    python scripts/label_rewards_lrm.py --suite spatial --model /path/to/merged
    # off-the-shelf LRM (does not match LIBERO's success predicates well)
    python scripts/label_rewards_lrm.py --suite spatial
"""
import argparse
import json
import sys
from pathlib import Path

# Make `fqe` importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch

from fqe.lrm import (
    DEFAULT_MODEL, IMAGE_TRANSFORMS, default_subfolder, load_lrm, predict, to_pil,
)
from fqe.paths import get_paths, VALID_SUITES
from fqe.rollouts import ROLLOUT_TAG


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--suite", choices=list(VALID_SUITES), default="spatial")
    ap.add_argument("--rollout-dir", default=None,
                    help="Default: experiments/libero_{suite}/data/libero_{suite}_rollouts")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF repo id or local dir of the completion model.")
    ap.add_argument("--subfolder", default=None,
                    help="Subfolder of --model. Default: 'completion' for "
                         f"{DEFAULT_MODEL}, none for any other --model.")
    ap.add_argument("--adapter", default=None,
                    help="Optional LoRA adapter on --model (scripts/finetune_lrm.py), "
                         "merged before labeling.")
    ap.add_argument("--image-key", default="agentview_rgb")
    ap.add_argument("--image-transform", choices=list(IMAGE_TRANSFORMS), default="rot180")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true",
                    help="Relabel all files. By default, files already labeled "
                         "with the same detector setup are skipped.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report labels and agreement without writing them.")
    args = ap.parse_args()
    if args.subfolder is None:
        args.subfolder = default_subfolder(args.model)
    # The adapter must see frames the way it was fine-tuned on them.
    ft_cfg = Path(args.adapter or "") / "finetune_lrm.json"
    if args.adapter and ft_cfg.is_file():
        ft = json.loads(ft_cfg.read_text())
        for key in ("image_key", "image_transform"):
            if ft.get(key) not in (None, getattr(args, key)):
                print(f"[lrm] WARNING: adapter was fine-tuned with {key}={ft[key]!r}, "
                      f"labeling with {getattr(args, key)!r}")

    P = get_paths(args.suite)
    rollout_dir = Path(args.rollout_dir) if args.rollout_dir else P["rollout_dir"]
    files = sorted(rollout_dir.glob("*_rollouts.hdf5"))
    if not files:
        raise FileNotFoundError(f"No *_rollouts.hdf5 under {rollout_dir}")
    # Detector setup, stored on every labeled file.
    label_attrs = {
        "reward_source": "lrm",
        "lrm_model": args.model + (f"/{args.subfolder}" if args.subfolder else ""),
        "lrm_adapter": args.adapter or "",
        "lrm_image": f"{args.image_key} ({args.image_transform})",
    }
    # Check every file before loading the 8B model. Fully labeled files are
    # skipped if the setup matches, so collections can be labeled incrementally.
    todo = []
    for path in files:
        with h5py.File(path, "r") as f:
            data = f["data"]
            if data.attrs.get("tag") != ROLLOUT_TAG:
                raise ValueError(f"{path.name} was not written by collect_rollouts.py")
            if data.attrs.get("macros_image_convention") != "opengl":
                print(f"[lrm] WARNING: {path.name} frames are not in the OpenGL "
                      f"convention; check --image-transform")
            demo_keys = [k for k in data.keys() if k.startswith("demo_")]
            labeled = bool(demo_keys) and all("rewards" in data[k] for k in demo_keys)
            if labeled and not (args.overwrite or args.dry_run):
                if any(data.attrs.get(k) != v for k, v in label_attrs.items()):
                    raise ValueError(
                        f"{path.name} is labeled by another setup "
                        f"({data.attrs.get('lrm_model')}, adapter "
                        f"{data.attrs.get('lrm_adapter')!r}); pass --overwrite to relabel"
                    )
                continue
        todo.append(path)

    print(f"[lrm] model        = {label_attrs['lrm_model']}"
          + (f" + adapter {args.adapter}" if args.adapter else ""))
    print(f"[lrm] rollouts     = {len(todo)} of {len(files)} files in {rollout_dir} "
          f"({len(files) - len(todo)} already labeled with this setup)")
    if not todo:
        return
    model, processor = load_lrm(args.model, args.subfolder, args.adapter, args.device)

    records = []
    for path in todo:
        with h5py.File(path, "r" if args.dry_run else "r+") as f:
            data = f["data"]
            instruction = json.loads(data.attrs["problem_info"])["language_instruction"]
            demo_keys = sorted(
                [k for k in data.keys() if k.startswith("demo_")],
                key=lambda x: int(x.split("_")[1]),
            )
            for key in demo_keys:
                d = data[key]
                frames = d["obs"][args.image_key]
                T = frames.shape[0]
                image = to_pil(frames[T - 1], args.image_transform)
                pred, response = predict(
                    model, processor, image, instruction, args.max_new_tokens
                )
                label = int(pred is True)  # unparseable answers count as failure
                records.append({
                    "file": path.name,
                    "demo": key,
                    "behavior_policy": d.attrs.get("behavior_policy", ""),
                    "lrm": label,
                    "parsed": pred is not None,
                    "sim": int(d.attrs["success"]) if "success" in d.attrs else None,
                    "response": response,
                })
                if args.dry_run:
                    continue
                rewards = np.zeros(T, dtype=np.uint8)
                rewards[-1] = label
                if "rewards" in d:
                    del d["rewards"]
                d.create_dataset("rewards", data=rewards)
                d.attrs["lrm_success"] = bool(label)
                d.attrs["lrm_response"] = response
            if not args.dry_run:
                data.attrs.update(label_attrs)
        n_file = [r for r in records if r["file"] == path.name]
        print(f"[lrm] {path.name}: {sum(r['lrm'] for r in n_file)}/{len(n_file)} labeled success")

    # Agreement with the simulator, overall and per behavior policy.
    summary = {"episodes": len(records),
               "labeled_success": sum(r["lrm"] for r in records),
               "unparsed": sum(not r["parsed"] for r in records)}
    graded = [r for r in records if r["sim"] is not None]
    if graded:
        lrm = np.array([r["lrm"] for r in graded])
        sim = np.array([r["sim"] for r in graded])
        summary.update({
            "agreement": float((lrm == sim).mean()),
            "tp": int(((lrm == 1) & (sim == 1)).sum()),
            "fp": int(((lrm == 1) & (sim == 0)).sum()),
            "tn": int(((lrm == 0) & (sim == 0)).sum()),
            "fn": int(((lrm == 0) & (sim == 1)).sum()),
            "per_policy": {},
        })
        for policy in sorted({r["behavior_policy"] for r in graded}):
            rs = [r for r in graded if r["behavior_policy"] == policy]
            summary["per_policy"][policy] = {
                "n": len(rs),
                "agreement": float(np.mean([r["lrm"] == r["sim"] for r in rs])),
                "lrm_success_rate": float(np.mean([r["lrm"] for r in rs])),
                "sim_success_rate": float(np.mean([r["sim"] for r in rs])),
            }
    print("[lrm] summary: " + json.dumps(
        {k: v for k, v in summary.items() if k != "per_policy"}))
    for policy, s in summary.get("per_policy", {}).items():
        print(f"[lrm]   {policy}: agreement {s['agreement']:.3f} "
              f"(LRM success {s['lrm_success_rate']:.2f}, sim {s['sim_success_rate']:.2f}, n={s['n']})")
    report = rollout_dir / ("lrm_labels_dryrun.json" if args.dry_run else "lrm_labels.json")
    report.write_text(json.dumps(
        {"args": vars(args), "summary": summary, "records": records}, indent=1
    ))
    print(f"[lrm] report       = {report}")


if __name__ == "__main__":
    main()
