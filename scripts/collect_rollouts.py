"""Offline-dataset augmentation: LIBERO rollouts of mixed behavior policies.

Paper Sec. 4.1 / App. A.4: the offline dataset of each suite adds rollouts of
other policies to the 50 official (all successful) demos per task; without
their failures FQE is biased upward (demo-only ARE is 5-10x higher). The
paper's LIBERO data adds 25 rollouts per task from partially trained ResNet-T
BC checkpoints. The world-model baselines are post-trained on the same data,
so the behavior policies can be any mix, to broaden state-action coverage:

  --policies SPEC [SPEC ...]      SPEC = NAME[@NOISE][=COUNT]
    NAME    `random` (uniform actions in [-1, 1]^7), or a LIBERO BC
            checkpoint: a file in bc_ckpts/{policy_config}/ (e.g. the
            multitask pipeline's multitask_model_ep10.pth) or a path
    @NOISE  std of Gaussian noise added to the BC actions
    =COUNT  rollouts per task from this policy; give it for every SPEC or for
            none (then --n-rollouts is split evenly)
    The suffixes are only read at the end of a SPEC, so checkpoint paths may
    contain '@' or '=' (e.g. sweep dirs like seed=1/).

Episodes of all policies run in lockstep in parallel simulator processes; see
`fqe.rollouts` for the episode protocol and the HDF5 layout. Output, per task:

    experiments/libero_{suite}/data/libero_{suite}_rollouts/{task}_rollouts.hdf5

The files carry simulator success (`sim_rewards`, `success` attr) but no
`rewards`: label them with scripts/label_rewards_lrm.py before FQE training
(or train on the simulator labels with --rollout-reward-key sim_rewards).

Run (MUJOCO_GL=egl):
    # paper setting: 25 rollouts per task from partially trained BC checkpoints
    python scripts/collect_rollouts.py --suite spatial --policy-config img128_noprop \\
        --policies multitask_model_ep5.pth multitask_model_ep15.pth multitask_model_ep20.pth

    # wider coverage (e.g. for the world models): BC training stages, a noisy
    # BC policy and random actions, kept in a separate rollout dir
    python scripts/collect_rollouts.py --suite spatial --policy-config img128_noprop \\
        --policies multitask_model_ep5.pth=10 multitask_model_ep20.pth=10 \\
                   multitask_model_ep20.pth@0.3=10 random=10 --init-states reset \\
        --out-dir experiments/libero_spatial/data/libero_spatial_rollouts_coverage
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

# Make `fqe` importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np

from fqe.base import set_seed
from fqe.data import compute_libero_task_embeddings
from fqe.paths import get_paths, VALID_SUITES
from fqe.policy import LIBEROPolicyWrapper
from fqe.rollouts import (
    IMAGE_SIZE, MAX_STEPS, NUM_WARMUP_STEPS, BCPolicy, RandomPolicy,
    load_init_states, make_vector_env, run_episodes, write_rollout_file,
)


_SPEC_RE = re.compile(
    r"(?P<name>.+?)(?:@(?P<noise>\d*\.?\d+(?:[eE][-+]?\d+)?))?(?:=(?P<count>-?\d+))?"
)


def parse_policy_spec(spec: str):
    """NAME[@NOISE][=COUNT] → (name, noise_std, count or None)."""
    m = _SPEC_RE.fullmatch(spec)
    name, noise = m["name"], float(m["noise"] or 0.0)
    count = int(m["count"]) if m["count"] is not None else None
    if name == "random" and noise:
        raise ValueError(f"{spec!r}: action noise only applies to BC checkpoints")
    return name, noise, count


def derive_seed(*keys: int) -> int:
    """Independent seed per key tuple, < 2**31 so the vector env's per-worker
    offsets stay valid numpy seeds."""
    return int(np.random.SeedSequence(list(keys)).generate_state(1)[0]) >> 1


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--suite", choices=list(VALID_SUITES), default="spatial")
    ap.add_argument("--policy-config", default="img128_noprop",
                    help="BC ckpt subdir name (e.g. img128_prop, img128_noprop). "
                         "Empty = flat bc_ckpts/ layout.")
    ap.add_argument("--policies", nargs="+", required=True,
                    help="Behavior policies, NAME[@NOISE][=COUNT] each "
                         "(see the module docstring).")
    ap.add_argument("--n-rollouts", type=int, default=25,
                    help="Rollouts per task when no SPEC gives =COUNT.")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Policy steps per episode. Default: the paper's horizon "
                         "of the suite (220 / 280 / 300 / 520; 400 for libero_90).")
    ap.add_argument("--init-states", choices=["fixed", "reset"], default="fixed",
                    help="fixed: the benchmark's init states, as LIBERO "
                         "evaluation (episode e uses state e mod 50); reset: a "
                         "fresh env.reset() sample per episode (more coverage).")
    ap.add_argument("--tasks", type=int, nargs="+", default=None,
                    help="Task indices to collect (default: all tasks of the suite).")
    ap.add_argument("--num-envs", type=int, default=25,
                    help="Parallel simulator processes.")
    ap.add_argument("--out-dir", default=None,
                    help="Default: experiments/libero_{suite}/data/libero_{suite}_rollouts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-collect tasks whose rollout file already exists.")
    args = ap.parse_args()

    try:
        specs = [parse_policy_spec(s) for s in args.policies]
    except ValueError as e:
        ap.error(str(e))
    counts = [count for _, _, count in specs]
    if all(c is None for c in counts):
        k = len(specs)
        counts = [args.n_rollouts // k + (i < args.n_rollouts % k) for i in range(k)]
    elif any(c is None for c in counts):
        ap.error("give =COUNT for every --policies entry or for none")
    if any(c < 0 for c in counts) or sum(counts) == 0:
        ap.error("rollout counts (=COUNT / --n-rollouts) must be >= 0 with a positive total")
    if args.num_envs < 1:
        ap.error("--num-envs must be >= 1")
    if args.max_steps is not None and args.max_steps < 1:
        ap.error("--max-steps must be >= 1")
    # Episode e → policy assignment[e], interleaved so that every policy gets
    # spread-out init states.
    assignment, remaining = [], list(counts)
    while any(r > 0 for r in remaining):
        for i, r in enumerate(remaining):
            if r > 0:
                assignment.append(i)
                remaining[i] -= 1
    labels = [
        name if not noise else f"{name}@{noise:g}" for name, noise, _ in specs
    ]

    # Forked simulator workers fail to create their EGL contexts
    # (EGL_BAD_ALLOC with the NVIDIA EGL driver); LIBERO's SubprocVectorEnv
    # follows the global start method.
    mp.set_start_method("spawn", force=True)

    # Upstream LIBERO asserts at least one proprio modality is enabled (see
    # bc_transformer_policy.py:50). No-proprio BC ckpts (img128_noprop) need
    # this runtime monkey-patch before LIBEROPolicyWrapper invokes the
    # policy ctor. No-op for proprio configs.
    from fqe.libero_patches import apply_no_proprio_patch
    apply_no_proprio_patch()

    import libero.libero.envs.bddl_utils as BDDLUtils
    import robomimic.utils.obs_utils as ObsUtils
    from libero.libero import benchmark, get_libero_path
    from robosuite import macros

    P = get_paths(args.suite, args.policy_config)
    out_dir = Path(args.out_dir) if args.out_dir else P["rollout_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    max_steps = args.max_steps or MAX_STEPS[args.suite]
    bm = benchmark.get_benchmark_dict()[f"libero_{args.suite}"]()
    task_ids = args.tasks if args.tasks is not None else list(range(bm.n_tasks))

    # One policy object per SPEC (a BC policy's history lives in its module,
    # so a ckpt listed twice is loaded twice).
    policies = []
    for (name, noise, _), label in zip(specs, labels):
        if name == "random":
            policies.append(RandomPolicy())
            continue
        ckpt = P["bc_ckpt_dir"] / name
        if not ckpt.is_file() and Path(name).is_file():
            ckpt = Path(name)
        wrapper = LIBEROPolicyWrapper(str(ckpt))
        cfg = wrapper.cfg
        if cfg.task_embedding_format != "bert" or getattr(cfg.data, "max_word_len", 25) != 25:
            raise ValueError(f"{label}: only BERT task embeddings (max_word_len 25) are supported")
        if (cfg.data.img_h, cfg.data.img_w) != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"{label}: policy images are {cfg.data.img_h}x{cfg.data.img_w}, "
                             f"the demos are {IMAGE_SIZE}x{IMAGE_SIZE}")
        policies.append(BCPolicy(wrapper, noise_std=noise))

    bc_policies = [p for p in policies if isinstance(p, BCPolicy)]
    task_embs = None
    if bc_policies:
        # raw_obs_to_tensor_obs → robomimic process_obs needs the obs-modality
        # registry, which LIBERO fills at train time (get_dataset); merge the
        # specs of all checkpoints.
        ObsUtils.initialize_obs_utils_with_obs_specs(
            [{"obs": p.cfg.data.obs.modality} for p in bc_policies]
        )
        # BERT embeddings as LIBERO training / evaluation computes them, from
        # the benchmark's task language (== the demos' language_instruction).
        descriptions = [bm.get_task(i).language for i in range(bm.n_tasks)]
        task_embs = compute_libero_task_embeddings(
            descriptions, cache_dir=str(P["bert_cache"])
        )

    print(f"[collect] suite        = libero_{args.suite} (tasks {task_ids})")
    print("[collect] policies     = " + ", ".join(
        f"{label} x{c}" for label, c in zip(labels, counts)))
    print(f"[collect] episodes     = {len(assignment)} per task, max {max_steps} steps, "
          f"{args.init_states} init states")
    print(f"[collect] out_dir      = {out_dir}")

    summary_path = out_dir / "collect_rollouts_summary.json"
    for task_idx in task_ids:
        task = bm.get_task(task_idx)
        out_path = out_dir / f"{task.name}_rollouts.hdf5"
        if out_path.exists() and not args.overwrite:
            print(f"[collect] skip {out_path.name} (exists; --overwrite to redo)")
            continue
        n_eps = len(assignment)
        if args.init_states == "fixed":
            init_states = load_init_states(task)
            init_index = [e % len(init_states) for e in range(n_eps)]
        else:
            init_states, init_index = None, [-1] * n_eps
        task_emb = task_embs[task_idx] if task_embs is not None else None

        set_seed(derive_seed(args.seed, task_idx))
        t0 = time.time()
        episodes = []
        for start in range(0, n_eps, args.num_envs):
            ids = list(range(start, min(start + args.num_envs, n_eps)))
            env = make_vector_env(task, len(ids))
            try:
                episodes += run_episodes(
                    env, policies, [assignment[e] for e in ids], task_emb, max_steps,
                    init_states=(
                        None if init_states is None
                        else init_states[[init_index[e] for e in ids]]
                    ),
                    seed=derive_seed(args.seed, task_idx, start),
                )
            finally:
                env.close()

        problem_info = BDDLUtils.get_problem_info(
            os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        )
        # The demos' language_instruction is the benchmark's task language;
        # the BDDL's own text differs on some suites ("pick the akita black
        # bowl ..."), so FQE and the success detector would see another task.
        problem_info["language_instruction"] = task.language
        file_attrs = {
            "problem_info": json.dumps(problem_info),
            "bddl_file_name": os.path.join(
                "libero/libero/bddl_files", task.problem_folder, task.bddl_file
            ),
            "macros_image_convention": macros.IMAGE_CONVENTION,
            "suite": f"libero_{args.suite}",
            "task_index": task_idx,
            "task_name": task.name,
            "policy_config": args.policy_config,
            "behavior_policies": [
                {"policy": label, "count": c} for label, c in zip(labels, counts)
            ],
            "init_states": args.init_states,
            "max_steps": max_steps,
            "num_warmup_steps": NUM_WARMUP_STEPS,
            "seed": args.seed,
        }
        # env_name / env_args as in the task's demos (same robot, controller
        # and cameras), which LIBERO's dataset tools read.
        demo_path = P["dataset_dir"] / f"{task.name}_demo.hdf5"
        if demo_path.is_file():
            with h5py.File(demo_path, "r") as f:
                for key in ("env_name", "env_args"):
                    if key in f["data"].attrs:
                        file_attrs[key] = f["data"].attrs[key]
        episode_attrs = [
            {"behavior_policy": labels[assignment[e]], "init_state_index": init_index[e]}
            for e in range(n_eps)
        ]
        write_rollout_file(str(out_path), episodes, episode_attrs, file_attrs)

        success = np.array([ep["success"] for ep in episodes])
        lengths = np.array([len(ep["actions"]) for ep in episodes])
        per_policy = {}
        for i, label in enumerate(labels):
            mask = np.asarray(assignment) == i
            if mask.any():
                per_policy[label] = {
                    "n": int(mask.sum()),
                    "success_rate": float(success[mask].mean()),
                    "mean_length": float(lengths[mask].mean()),
                }
        # Re-read before merging: collectors on disjoint --tasks may share out_dir.
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        summary[task.name] = {
            "success_rate": float(success.mean()),
            "transitions": int(lengths.sum()),
            "per_policy": per_policy,
        }
        tmp = summary_path.with_name(f"{summary_path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(summary, indent=2))
        os.replace(tmp, summary_path)
        print(
            f"[collect] task {task_idx} {task.name}: success {success.mean():.2f} "
            f"({success.sum()}/{n_eps}), {lengths.sum()} transitions, "
            f"{time.time() - t0:.0f}s | "
            + ", ".join(f"{k} {v['success_rate']:.2f}" for k, v in per_policy.items())
        )


if __name__ == "__main__":
    main()
