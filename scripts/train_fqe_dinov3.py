"""DINOv3 FQE training on a LIBERO suite, with a BC target policy.

Same algorithm + temporal aggregator + Q-head as `train_fqe_scratch.py`; the
per-camera image encoder is a pretrained DINOv3 backbone. `--mode` picks how
much of the backbone the FQE TD loss may adapt (no separate SSL step):

  frozen   Backbone frozen end-to-end; only the per-camera projection,
           temporal Transformer and Q-head train (~1.5M params for ViT-S/16).
  lora     Backbone base weights frozen; peft inserts trainable rank-r LoRA
           adapters (default attn q/k/v) trained jointly with the head.
  partial  Patch embedding + bottom (N - K) transformer blocks frozen; top
           `--n-finetune-blocks` K blocks + final LayerNorm + pooler train,
           with a separate (lower) `--backbone-lr`.

Run (MUJOCO_GL=egl, HF_TOKEN with access to facebook/dinov3-*):
    python scripts/train_fqe_dinov3.py --mode frozen \\
        --suite object --policy-config img128_noprop \\
        --ckpt-name multitask_model_ep50.pth --encoder dinov3_vits16

    python scripts/train_fqe_dinov3.py --mode lora \\
        --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \\
        --lora-targets q_proj,k_proj,v_proj ...

    python scripts/train_fqe_dinov3.py --mode partial \\
        --n-finetune-blocks 4 --backbone-lr 1e-5 --lr 1e-4 --grad-clip 0.5 ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `fqe` importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fqe.paths import get_paths, VALID_SUITES
from fqe.data import libero_to_ope
from fqe.policy import LIBEROPolicyWrapper
from fqe.trainer import OPETrainer
from fqe.pretrained import FittedQEvaluationVisionPretrained

MODES = ("frozen", "lora", "partial")
# Per-mode defaults for flags left unset on the CLI.
DEFAULT_GRAD_CLIP = {"frozen": 1.0, "lora": 1.0, "partial": 0.5}
DEFAULT_PARTIAL_BACKBONE_LR = 1e-5


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mode", choices=list(MODES), default="frozen")
    ap.add_argument("--suite", choices=list(VALID_SUITES), default="object")
    ap.add_argument("--policy-config", default="img128_noprop")
    ap.add_argument("--ckpt-name", default="multitask_model_ep50.pth")
    ap.add_argument("--encoder", default="dinov3_vits16",
                    help="See fqe.pretrained.encoders.build_pretrained_encoder.")
    ap.add_argument("--n-steps", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="LR for per-image proj / frame_proj / temporal "
                         "Transformer / Q-head MLP (and LoRA adapters in "
                         "--mode lora).")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="Default: 1.0 for frozen / lora, 0.5 for partial "
                         "(tighter to limit per-step backbone drift under "
                         "TD noise).")
    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    # --mode lora
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", default="q_proj,k_proj,v_proj",
                    help="Comma-separated peft target_modules. Add o_proj / "
                         "up_proj / down_proj for more capacity at the cost "
                         "of more trainable params.")
    # --mode partial
    ap.add_argument("--n-finetune-blocks", type=int, default=4,
                    help="Top K transformer blocks (+ final LN + pooler) "
                         "kept trainable. ViT-S/B/L all have 12 blocks; "
                         "4 = top 1/3.")
    ap.add_argument("--backbone-lr", type=float, default=None,
                    help="LR for the trainable backbone slice "
                         f"(default {DEFAULT_PARTIAL_BACKBONE_LR:g}); "
                         "--mode partial only.")
    args = ap.parse_args()

    if args.mode != "partial" and args.backbone_lr is not None:
        ap.error("--backbone-lr is only used with --mode partial")
    if args.mode == "partial" and args.n_finetune_blocks <= 0:
        ap.error("--mode partial needs --n-finetune-blocks > 0")
    grad_clip = (
        args.grad_clip if args.grad_clip is not None
        else DEFAULT_GRAD_CLIP[args.mode]
    )

    # Upstream LIBERO asserts at least one proprio modality is enabled (see
    # bc_transformer_policy.py:50). No-proprio BC ckpts (img128_noprop) need
    # this runtime monkey-patch before LIBEROPolicyWrapper invokes the
    # policy ctor. No-op for proprio configs.
    from fqe.libero_patches import apply_no_proprio_patch
    apply_no_proprio_patch()

    P = get_paths(args.suite, args.policy_config)
    policy_ckpt = P["bc_ckpt_dir"] / args.ckpt_name

    # Bake mode-specific hparams into the save_dir name so concurrent sweeps
    # don't collide and post-hoc analysis can grep the config out of the path.
    mode_kwargs = {}
    if args.mode == "frozen":
        method_tag = f"dinov3_frozen_{args.encoder}"
        mode_desc = "frozen, no FT"
    elif args.mode == "lora":
        lora_targets = [s.strip() for s in args.lora_targets.split(",") if s.strip()]
        targets_tag = "-".join(t.replace("_proj", "") for t in lora_targets)
        method_tag = (
            f"dinov3_lora_{args.encoder}_r{args.lora_r}_a{args.lora_alpha}_{targets_tag}"
        )
        mode_desc = (
            f"base frozen + LoRA r={args.lora_r} α={args.lora_alpha} "
            f"drop={args.lora_dropout} targets={lora_targets}"
        )
        mode_kwargs = dict(
            use_lora=True,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=lora_targets,
        )
    else:
        backbone_lr = (
            args.backbone_lr if args.backbone_lr is not None
            else DEFAULT_PARTIAL_BACKBONE_LR
        )
        method_tag = (
            f"dinov3_partft{args.n_finetune_blocks}_{args.encoder}"
            f"_bblr{backbone_lr:g}"
        )
        mode_desc = (
            f"top {args.n_finetune_blocks} blocks + final LN + pooler trainable "
            f"(rest frozen), backbone lr {backbone_lr:g}"
        )
        mode_kwargs = dict(
            n_finetune_blocks=args.n_finetune_blocks,
            backbone_lr=backbone_lr,
            head_lr=args.lr,
        )
    cfg_tag = args.policy_config or "legacy"
    ckpt_tag = args.ckpt_name.replace(".pth", "") + f"_gamma{args.gamma:g}_seed{args.seed}"
    save_dir = P["fqe_results"] / method_tag / cfg_tag / ckpt_tag
    save_dir.mkdir(parents=True, exist_ok=True)

    tag = f"[fqe-dinov3-{args.mode}]"
    print(f"{tag} suite        = libero_{args.suite}")
    print(f"{tag} encoder      = {args.encoder} ({mode_desc})")
    print(f"{tag} lr / clip    = {args.lr:g} / {grad_clip}")
    print(f"{tag} policy_ckpt  = {policy_ckpt}")
    print(f"{tag} save_dir     = {save_dir}")

    dataset = libero_to_ope(
        dataset_dir=str(P["dataset_dir"]),
        frame_stack=10,
        image_keys=("agentview_rgb", "eye_in_hand_rgb"),
        lowdim_keys=("gripper_states", "joint_states"),
        bert_cache_dir=str(P["bert_cache"]),
    )
    policy = LIBEROPolicyWrapper(str(policy_ckpt))

    sample = dataset[0]
    image_keys = list(dataset.image_keys)
    lowdim_keys = list(dataset.lowdim_keys)
    image_shape = tuple(sample["obs"][image_keys[0]].shape[-3:])
    lowdim_dims = {k: int(sample["obs"][k].shape[-1]) for k in lowdim_keys}
    action_dim = int(sample["actions"].shape[0])
    task_emb_dim = int(sample["task_emb"].shape[0])
    frame_stack = int(sample["obs"][image_keys[0]].shape[0])

    algorithm = FittedQEvaluationVisionPretrained(
        image_keys=image_keys,
        image_shape=image_shape,
        lowdim_keys=lowdim_keys,
        lowdim_dims=lowdim_dims,
        action_dim=action_dim,
        task_emb_dim=task_emb_dim,
        frame_stack=frame_stack,
        gamma=args.gamma,
        learning_rate=args.lr,
        weight_decay=1e-5,
        loss_type="mse",
        target_update_type="soft",
        tau=0.005,
        clip_target=True,
        clip_min=0.0,
        clip_max=1.0,
        grad_clip=grad_clip,
        encoder_name=args.encoder,
        freeze_backbone=True,
        share_encoders=True,
        encoder_out_dim=128,
        token_dim=256,
        temporal_layers=2,
        temporal_heads=4,
        temporal_ff=512,
        head_hidden=(256, 256),
        activation="relu",
        **mode_kwargs,
    )

    # Parameter accounting so train.log carries the verified frozen /
    # trainable split — helpful when comparing across runs / seeds.
    qn = algorithm.q_network
    total = sum(p.numel() for p in qn.parameters())
    trainable = sum(p.numel() for p in qn.parameters() if p.requires_grad)
    bb_trainable = sum(
        p.numel() for n, p in qn.named_parameters()
        if n.startswith("image_encoders") and p.requires_grad
    )
    print(
        f"{tag} q-net params: total={total:,}  trainable={trainable:,}  "
        f"(image_encoders={bb_trainable:,}, head={trainable - bb_trainable:,})  "
        f"frozen={total - trainable:,}"
    )

    trainer = OPETrainer(
        algorithm=algorithm,
        dataset=dataset,
        policy=policy,
        behavior_policy=None,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_dir=str(save_dir),
        use_wandb=False,
        seed=args.seed,
    )

    trainer.train(num_iterations=args.n_steps)
    final = trainer.evaluate()
    print(f"{tag} Final DM estimate: {final}")
    trainer.finish()


if __name__ == "__main__":
    main()
