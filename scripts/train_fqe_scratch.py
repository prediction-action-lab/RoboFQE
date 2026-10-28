"""From-scratch vision FQE (scalar MSE) on any LIBERO suite, with a BC target policy.

All tasks of the suite combined into a single dataset. DM-only estimation.

Adapted from scripts/libero/spatial/fqe.py in RoboOPE main commit 6c2c61a
(author: Sangwoo Shin <jsw7460@gmail.com>). Paths resolved via
fqe.paths.get_paths.

Run (MUJOCO_GL=egl):
    python scripts/train_fqe_scratch.py --suite spatial \\
        --policy-config img128_noprop --ckpt-name multitask_model_ep50.pth
"""
import argparse
import sys
from pathlib import Path

# Make `fqe` importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fqe.paths import get_paths, VALID_SUITES
from fqe.data import libero_to_ope
from fqe.policy import LIBEROPolicyWrapper
from fqe.trainer import OPETrainer
from fqe.scratch import FittedQEvaluationVision


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--suite", choices=list(VALID_SUITES), default="spatial")
    ap.add_argument("--policy-config", default="img128_noprop",
                    help="BC ckpt subdir name (e.g. img128_prop, img128_noprop). "
                         "Empty = flat bc_ckpts/ layout.")
    ap.add_argument("--ckpt-name", default="multitask_model_ep50.pth")
    ap.add_argument("--n-steps", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99,
                    help="Discount factor. Set to 1.0 for undiscounted return "
                         "(V^pi = success rate for sparse 0/1 LIBERO reward).")
    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Upstream LIBERO asserts at least one proprio modality is enabled (see
    # bc_transformer_policy.py:50). No-proprio BC ckpts (img128_noprop) need
    # this runtime monkey-patch before LIBEROPolicyWrapper invokes the
    # policy ctor. No-op for proprio configs.
    from fqe.libero_patches import apply_no_proprio_patch
    apply_no_proprio_patch()

    P = get_paths(args.suite, args.policy_config)
    policy_ckpt = P["bc_ckpt_dir"] / args.ckpt_name

    # Group fqe_results by method/policy-config/ckpt so multiple runs don't
    # collide.
    method_tag = "scratch"
    cfg_tag = args.policy_config or "legacy"
    # γ + seed encoded into save dir so re-runs with different gamma / seed
    # don't overwrite each other (matches train_fqe_dinov3.py).
    ckpt_tag = args.ckpt_name.replace(".pth", "") + f"_gamma{args.gamma:g}_seed{args.seed}"
    save_dir = P["fqe_results"] / method_tag / cfg_tag / ckpt_tag
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fqe-scratch] suite        = libero_{args.suite}")
    print(f"[fqe-scratch] policy_ckpt  = {policy_ckpt}")
    print(f"[fqe-scratch] save_dir     = {save_dir}")

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

    algorithm = FittedQEvaluationVision(
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
        grad_clip=1.0,
        encoder_out_dim=128,
        token_dim=256,
        temporal_layers=2,
        temporal_heads=4,
        temporal_ff=512,
        head_hidden=(256, 256),
        activation="relu",
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
    print(f"[fqe-scratch] {args.ckpt_name}  Final DM estimate: {final}")
    trainer.finish()


if __name__ == "__main__":
    main()
