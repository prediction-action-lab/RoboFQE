"""LoRA fine-tuning of the LRM task-completion model on the official LIBERO demos.

Paper Sec. 4.1: the augmented LIBERO rollouts are labeled by LRM "fine-tuned
on the official LIBERO demonstration data" (97.4% agreement with simulator
success on the rollouts). The demos are all successful and carry no per-frame
labels, so supervision is temporal, as in LRM's own training on successful
episodes:

  "yes"  frames among the last `--pos-window` steps of a demo (LIBERO demos
         end after a 10-step success hold, so these show the goal reached);
  "no"   frames at normalized progress t / (T - 1) <= `--neg-max-progress`;
  "no"   optionally (`--num-mismatch`), a demo's final frame paired with the
         instruction of another task of the same suite, so the answer must
         depend on the instruction. Off by default: every LIBERO-Spatial task
         ends with a black bowl on the plate, so there the label is ambiguous.

LoRA adapters are trained on the language-model projections of the LRM
completion checkpoint (vision tower and base weights frozen, as LRM trains its
completion interface), with the loss on the answer tokens only; fqe.lrm holds
the prompt, image transform and answer format shared with the labeler. The
last `--val-demos-per-task` demos of every task are held out to report
accuracy before and after training; the agreement on the rollouts is reported
by label_rewards_lrm.py.

Run:
    python scripts/finetune_lrm.py --suites spatial object goal 10
    python scripts/label_rewards_lrm.py --suite spatial \\
        --adapter experiments/_shared/lrm_libero_lora
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

from fqe.data import find_libero_task_files, libero_task_descriptions_from_hdf5
from fqe.lrm import (
    DEFAULT_MODEL, IMAGE_TRANSFORMS, completion_answer, completion_inputs,
    default_subfolder, load_lrm, predict, to_pil,
)
from fqe.paths import ROOT, get_paths, VALID_SUITES

LORA_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def build_examples(args, rng):
    """(train, val) lists of {frame, instruction, label, kind} from the demos."""
    train, val = [], []
    for suite in args.suites:
        files = find_libero_task_files(str(get_paths(suite)["dataset_dir"]))
        instructions = libero_task_descriptions_from_hdf5(files)
        for ti, (path, instruction) in enumerate(zip(files, instructions)):
            others = [x for j, x in enumerate(instructions) if j != ti]
            with h5py.File(path, "r") as f:
                demo_keys = sorted(
                    [k for k in f["data"].keys() if k.startswith("demo_")],
                    key=lambda x: int(x.split("_")[1]),
                )
                for di, key in enumerate(demo_keys):
                    split = val if di >= len(demo_keys) - args.val_demos_per_task else train
                    frames = f["data"][key]["obs"][args.image_key]
                    T = frames.shape[0]
                    last = np.arange(max(0, T - args.pos_window), T)
                    early = np.arange(int(args.neg_max_progress * (T - 1)) + 1)
                    picks = (
                        [(t, instruction, True, "pos") for t in
                         rng.choice(last, min(args.num_pos, len(last)), replace=False)]
                        + [(t, instruction, False, "neg") for t in
                           rng.choice(early, min(args.num_neg, len(early)), replace=False)]
                        + [(T - 1, other, False, "mismatch") for other in
                           rng.choice(others, min(args.num_mismatch, len(others)), replace=False)]
                    )
                    for t, instr, label, kind in picks:
                        split.append({"frame": frames[int(t)], "instruction": str(instr),
                                      "label": label, "kind": kind})
    return train, val


def evaluate(model, processor, examples, image_transform):
    """Greedy-decoding accuracy per example kind (unparseable = wrong)."""
    hits = {}
    for ex in examples:
        pred, _ = predict(model, processor, to_pil(ex["frame"], image_transform),
                          ex["instruction"])
        hits.setdefault(ex["kind"], []).append(pred is not None and pred == ex["label"])
    acc = {kind: float(np.mean(h)) for kind, h in hits.items()}
    acc["all"] = float(np.mean([x for h in hits.values() for x in h]))
    return acc


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--suites", nargs="+", choices=list(VALID_SUITES),
                    default=["spatial", "object", "goal", "10"],
                    help="Suites whose official demos (experiments/libero_{suite}/"
                         "data/libero_{suite}) are used.")
    ap.add_argument("--output-dir",
                    default=str(ROOT / "experiments" / "_shared" / "lrm_libero_lora"))
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF repo id or local dir of the completion model.")
    ap.add_argument("--subfolder", default=None,
                    help="Subfolder of --model. Default: 'completion' for "
                         f"{DEFAULT_MODEL}, none for any other --model.")
    ap.add_argument("--image-key", default="agentview_rgb")
    ap.add_argument("--image-transform", choices=list(IMAGE_TRANSFORMS), default="rot180")
    ap.add_argument("--pos-window", type=int, default=5,
                    help="'yes' frames are drawn from the last N steps of a demo.")
    ap.add_argument("--num-pos", type=int, default=2, help="'yes' frames per demo.")
    ap.add_argument("--num-neg", type=int, default=2, help="'no' frames per demo.")
    ap.add_argument("--neg-max-progress", type=float, default=0.8,
                    help="'no' frames are drawn from normalized progress <= this.")
    ap.add_argument("--num-mismatch", type=int, default=0,
                    help="Final frames per demo paired with another task's "
                         "instruction (labeled 'no').")
    ap.add_argument("--val-demos-per-task", type=int, default=2)
    ap.add_argument("--max-train-examples", type=int, default=None,
                    help="Random subset of the training examples (debugging).")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="Examples per optimizer step (each forward is one example).")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", default=LORA_TARGETS,
                    help="Comma-separated module names; these only match the "
                         "language model of Qwen3-VL.")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--log-interval", type=int, default=50,
                    help="Optimizer steps between loss logs.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.subfolder is None:
        args.subfolder = default_subfolder(args.model)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    train, val = build_examples(args, rng)
    if args.max_train_examples is not None:
        train = [train[i] for i in rng.permutation(len(train))[:args.max_train_examples]]
    counts = {
        name: {k: sum(ex["kind"] == k for ex in split) for k in ("pos", "neg", "mismatch")}
        for name, split in (("train", train), ("val", val))
    }
    print(f"[finetune] suites       = {args.suites}")
    print(f"[finetune] examples     = {counts}")
    print(f"[finetune] model        = {args.model} [{args.subfolder}]")

    model, processor = load_lrm(args.model, args.subfolder, None, args.device)
    metrics = {}
    if val:
        metrics["val_before"] = evaluate(model, processor, val, args.image_transform)
        print(f"[finetune] val accuracy before: {metrics['val_before']}")

    from peft import LoraConfig, get_peft_model

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[t.strip() for t in args.lora_targets.split(",") if t.strip()],
        bias="none",
    ))
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    # Batch size 1 with gradient accumulation: no padding of image tokens.
    model.train()
    step, losses = 0, []
    for epoch in range(args.epochs):
        order = rng.permutation(len(train))
        for i, idx in enumerate(order):
            ex = train[idx]
            inputs = completion_inputs(
                processor, to_pil(ex["frame"], args.image_transform),
                ex["instruction"], completion_answer(ex["label"]),
            ).to(model.device)
            loss = model(**inputs).loss
            (loss / args.grad_accum).backward()
            losses.append(loss.item())
            if (i + 1) % args.grad_accum == 0 or i + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_interval == 0:
                    recent = losses[-args.log_interval * args.grad_accum:]
                    print(f"[finetune] epoch {epoch} step {step}: loss {np.mean(recent):.4f}")

    model.eval()
    if val:
        metrics["val_after"] = evaluate(model, processor, val, args.image_transform)
        print(f"[finetune] val accuracy after: {metrics['val_after']}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    (out_dir / "finetune_lrm.json").write_text(json.dumps(
        {**vars(args), "examples": counts, "optimizer_steps": step, "metrics": metrics},
        indent=2,
    ))
    print(f"[finetune] adapter      = {out_dir}")


if __name__ == "__main__":
    main()
