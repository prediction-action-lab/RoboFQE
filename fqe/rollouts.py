"""Offline-dataset augmentation: LIBERO rollouts of mixed behavior policies.

The official LIBERO demos are all successful. RoboFQE adds rollouts of other
behavior policies (paper Sec. 4.1: 25 per task from partially trained
ResNet-T BC checkpoints), and the world-model baselines are post-trained on
the same data, so broad state-action coverage matters. A collection mixes
  - LIBERO BC checkpoints of any policy type and training stage (e.g. the
    `multitask_model_ep{N}.pth` of LIBERO's multitask pipeline), optionally
    with Gaussian action noise;
  - a uniform random policy over LIBERO's action space [-1, 1]^7.
`scripts/collect_rollouts.py` drives this module.

Episodes follow LIBERO's own evaluator (`libero/lifelong/metric.py::
evaluate_one_task_success`): start from a benchmark init state (or a fresh
env.reset() sample), 5 zero-action physics steps, then the behavior policy
until success (env `done`) or `max_steps` policy steps. BC checkpoints act
through their stateful `get_action` (history reset per episode, action
sampled from the policy head), as in LIBERO evaluation. Actions are clipped
to [-1, 1] before env.step, as robosuite's OSC / gripper controllers do
anyway, so the stored action is the executed command.

Episodes are stored the way `LIBERO/scripts/create_dataset.py` stores the
demos, so `LIBEROOPEDataset`, LIBERO tooling and the world-model data
pipelines read rollouts and demos alike:

    data/demo_{k}/obs/{agentview_rgb, eye_in_hand_rgb, gripper_states,
                       joint_states, ee_states, ee_pos, ee_ori}
    data/demo_{k}/{actions, states, robot_states, dones, sim_rewards}
    data/demo_{k}.attrs{model_file, init_state, num_samples, ...}

with create_dataset.py's alignment: obs[t] (and robot_states[t]) is the
observation returned by env.step(actions[t]), states[t] the MuJoCo state
actions[t] was applied in. obs[-1] is thus the terminal observation o_T a
success detector scores; `model_file` (the episode's MJCF) + `states` re-render
an episode at any resolution, as for the demos. Every episode ends with dones[-1] = 1 (the offline
trajectory stops there, success or timeout); sim_rewards[-1] and the
`success` attr record the simulator's success. The `rewards` dataset FQE
trains on is written by `scripts/label_rewards_lrm.py`.
"""
import json
import os
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch

# File-level `tag` attr of rollout files (the official demos carry "libero-v1").
ROLLOUT_TAG = "robofqe-rollouts-v1"
# Zero-action physics steps before the policy acts (LIBERO metric.py).
NUM_WARMUP_STEPS = 5
# Policy steps per episode: the OpenVLA LIBERO horizons, which the paper uses
# for Spatial / Object / Goal / Long (Sec. 4.1).
MAX_STEPS = {"spatial": 220, "object": 280, "goal": 300, "10": 520, "90": 400}
# Camera resolution of the official demos (and of LIBEROOPEDataset).
IMAGE_SIZE = 128
ACTION_DIM = 7

_IMAGE_KEYS = ("agentview_rgb", "eye_in_hand_rgb")


class RandomPolicy:
    """Uniform random actions over LIBERO's OSC_POSE + gripper space [-1, 1]^7."""

    def reset(self):
        pass

    def act(self, obs, task_emb) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, size=(len(obs), ACTION_DIM))


class BCPolicy:
    """LIBERO BC checkpoint acting as in LIBERO's evaluator.

    Drives the LIBERO policy of a `LIBEROPolicyWrapper` (any LIBERO policy
    type) through its own `get_action`, which keeps the per-episode history
    (latent queue / RNN state) and samples from the policy head.
    `noise_std` > 0 adds Gaussian noise to the sampled actions to spread the
    data around the policy's trajectories. The history lives in the policy
    module, so every BCPolicy needs its own wrapper.
    """

    def __init__(self, wrapper, noise_std: float = 0.0):
        self.policy = wrapper.policy
        self.cfg = wrapper.cfg
        self.noise_std = float(noise_std)

    def reset(self):
        self.policy.reset()

    def act(self, obs, task_emb) -> np.ndarray:
        from libero.lifelong.metric import raw_obs_to_tensor_obs

        data = raw_obs_to_tensor_obs(obs, task_emb, self.cfg)
        actions = self.policy.get_action(data)
        if self.noise_std > 0:
            actions = actions + np.random.normal(0.0, self.noise_std, size=actions.shape)
        return actions


def load_init_states(task) -> np.ndarray:
    """The benchmark's fixed init states of `task`, (N, state_dim).

    Same file as `Benchmark.get_task_init_states`, loaded with
    weights_only=False: the files pickle numpy arrays, which torch>=2.6
    refuses by default (upstream LIBERO calls torch.load bare).
    """
    from libero.libero import get_libero_path

    path = os.path.join(
        get_libero_path("init_states"), task.problem_folder, task.init_states_file
    )
    return np.asarray(torch.load(path, weights_only=False))


def make_vector_env(task, n_envs: int):
    """`n_envs` OffScreenRenderEnv's of `task`, built like LIBERO's evaluator.

    `ignore_done` (as in the demos' env_args) keeps robosuite from ending the
    episode at its 1000-step horizon; success still comes from LIBERO's `done`.
    """
    from libero.libero import get_libero_path
    from libero.libero.envs import DummyVectorEnv, OffScreenRenderEnv, SubprocVectorEnv

    # Exposes the episode MJCF to `get_env_attr` (the sim itself does not
    # pickle across the worker pipe). Defined here so that importing this
    # module does not load the simulator; cloudpickle ships it to the workers.
    class RolloutEnv(OffScreenRenderEnv):
        @property
        def model_xml(self):
            return self.env.sim.model.get_xml()

    env_args = {
        "bddl_file_name": os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        ),
        "camera_heights": IMAGE_SIZE,
        "camera_widths": IMAGE_SIZE,
        "ignore_done": True,
    }
    env_fns = [lambda: RolloutEnv(**env_args) for _ in range(n_envs)]
    return DummyVectorEnv(env_fns) if n_envs == 1 else SubprocVectorEnv(env_fns)


def _demo_frame(obs: Dict) -> Dict[str, np.ndarray]:
    """One frame of obs + robot_states, as create_dataset.py records them."""
    from robosuite.utils.transform_utils import quat2axisangle

    ee_states = np.hstack((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"])))
    return {
        "agentview_rgb": np.array(obs["agentview_image"]),
        "eye_in_hand_rgb": np.array(obs["robot0_eye_in_hand_image"]),
        "gripper_states": np.array(obs["robot0_gripper_qpos"]),
        "joint_states": np.array(obs["robot0_joint_pos"]),
        "ee_states": ee_states,
        "ee_pos": ee_states[:3],
        "ee_ori": ee_states[3:],
        # BDDLBaseDomain.get_robot_state_vector
        "robot_states": np.concatenate(
            [obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]]
        ),
    }


def run_episodes(
    env,
    policies: Sequence,
    policy_ids: Sequence[int],
    task_emb: Optional[torch.Tensor],
    max_steps: int,
    init_states: Optional[np.ndarray] = None,
    seed: int = 0,
) -> List[Dict]:
    """One episode per env of `env`, all stepped in lockstep.

    Env k is driven by policies[policy_ids[k]] and starts from init_states[k],
    or from a fresh env.reset() sample when `init_states` is None. Each policy
    only acts on its own envs, so its history stays per-episode. Returns one
    dict per env with the recorded frames / actions / states and `success`.
    """
    n = len(policy_ids)
    policy_ids = np.asarray(policy_ids)
    groups = [(p, np.flatnonzero(policy_ids == p)) for p in np.unique(policy_ids)]

    env.seed(seed)
    obs = env.reset()
    if init_states is not None:
        obs = env.set_init_state(init_states)
    model_xmls = env.get_env_attr("model_xml")
    for _ in range(NUM_WARMUP_STEPS):
        obs, _, _, _ = env.step(np.zeros((n, ACTION_DIM)))
    for p, _ in groups:
        policies[p].reset()
    states = env.get_sim_state()

    episodes = [
        {"frames": [], "actions": [], "states": [], "success": False,
         "model_file": model_xmls[k]}
        for k in range(n)
    ]
    active = np.ones(n, dtype=bool)
    with torch.no_grad():
        for _ in range(max_steps):
            actions = np.zeros((n, ACTION_DIM))
            for p, ids in groups:
                if active[ids].any():
                    actions[ids] = policies[p].act(obs[ids], task_emb)
            actions = np.clip(actions, -1.0, 1.0)
            obs, _, done, _ = env.step(actions)
            next_states = env.get_sim_state()
            for k in np.flatnonzero(active):
                ep = episodes[k]
                ep["states"].append(np.array(states[k]))
                ep["actions"].append(actions[k].copy())
                ep["frames"].append(_demo_frame(obs[k]))
                if done[k]:
                    ep["success"] = True
                    active[k] = False
            states = next_states
            if not active.any():
                break
    return episodes


def write_rollout_file(
    path: str,
    episodes: Sequence[Dict],
    episode_attrs: Sequence[Dict],
    file_attrs: Dict,
):
    """Write `episodes` as data/demo_{i} of a new LIBERO-format HDF5.

    Goes to `path + ".tmp"` first and is renamed at the end, so an interrupted
    run never leaves a partial file under the final name.
    """
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        grp = f.create_group("data")
        for key, value in file_attrs.items():
            grp.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
        total = 0
        for i, (ep, attrs) in enumerate(zip(episodes, episode_attrs)):
            T = len(ep["actions"])
            g = grp.create_group(f"demo_{i}")
            obs_grp = g.create_group("obs")
            for key in ep["frames"][0]:
                arr = np.stack([fr[key] for fr in ep["frames"]])
                if key == "robot_states":
                    g.create_dataset(key, data=arr)
                elif key in _IMAGE_KEYS:
                    obs_grp.create_dataset(
                        key, data=arr, chunks=(1,) + arr.shape[1:], compression="lzf"
                    )
                else:
                    obs_grp.create_dataset(key, data=arr)
            states = np.stack(ep["states"])
            dones = np.zeros(T, dtype=np.uint8)
            dones[-1] = 1
            sim_rewards = np.zeros(T, dtype=np.uint8)
            sim_rewards[-1] = int(ep["success"])
            g.create_dataset("actions", data=np.stack(ep["actions"]).astype(np.float64))
            g.create_dataset("states", data=states)
            g.create_dataset("dones", data=dones)
            g.create_dataset("sim_rewards", data=sim_rewards)
            g.attrs["num_samples"] = T
            g.attrs["model_file"] = ep["model_file"]
            g.attrs["init_state"] = states[0]
            g.attrs["success"] = bool(ep["success"])
            for key, value in attrs.items():
                g.attrs[key] = value
            total += T
        grp.attrs["num_demos"] = len(episodes)
        grp.attrs["total"] = total
        grp.attrs["tag"] = ROLLOUT_TAG
    os.replace(tmp, path)
