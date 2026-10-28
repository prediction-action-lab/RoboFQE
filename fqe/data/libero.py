"""
LIBERO dataset → OPEDataset-compatible format.

Differences from `robomimic_to_ope`:
  - LIBERO HDF5 has no `next_obs` group (load_next_obs=False at BC train time)
    → next_obs synthesized by shifting (last transition pads with last frame).
  - Image obs is uint8 (T, H, W, C) → float32 (H_stack, C, H, W) in __getitem__.
  - Frame stacking with H=10 (matches BC training); first H-1 frames pad-replicate
    the trajectory's first frame (matches LIBERO's empty-queue rollout convention).
  - task_emb (BERT-encoded language) attached per-transition; required by
    LIBEROPolicyWrapper and the vision Q-network.
  - No `policy_checkpoint_path` metadata (no behavior policy in demos) — DM only.

Ported verbatim from opebench/datasets/libero_dataset.py
of RoboOPE main commit 6c2c61a (author: Sangwoo Shin <jsw7460@gmail.com>).
"""
import glob
import os
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_IMAGE_KEYS = ("agentview_rgb", "eye_in_hand_rgb")
DEFAULT_LOWDIM_KEYS = ("gripper_states", "joint_states")  # matches use_joint=True, use_gripper=True, use_ee=False


def _stack_indices(t: int, T: int, H: int) -> np.ndarray:
    """Indices for an H-length window ending at t, with pad-replicate at edges.

    For t < H-1: leading entries clipped to 0 (replicate first frame).
    For t > T-1: trailing entries clipped to T-1 (replicate last frame).
    """
    raw = np.arange(t - H + 1, t + 1)
    return np.clip(raw, 0, T - 1)


class LIBEROOPEDataset(Dataset):
    """OPE dataset for LIBERO BC-trained policies.

    Each item is a transition with frame-stacked obs / next_obs:
        {
          'obs':       Dict[k, Tensor],  image: (H, C, H_img, W_img); lowdim: (H, D)
          'next_obs':  Dict[k, Tensor],  same shapes
          'actions':   Tensor (A,)
          'rewards':   scalar
          'dones':     scalar
          'valid_mask':scalar
          'task_emb':  Tensor (D_emb,)
        }
    """

    def __init__(
        self,
        hdf5_paths: Sequence[str],
        task_embeddings: torch.Tensor,
        frame_stack: int = 10,
        image_keys: Sequence[str] = DEFAULT_IMAGE_KEYS,
        lowdim_keys: Sequence[str] = DEFAULT_LOWDIM_KEYS,
        image_size: int = 128,
        max_demos_per_task: Optional[int] = None,
    ):
        """
        Args:
            hdf5_paths: Ordered list of LIBERO task HDF5 paths.
                Order MUST match the order used to build `task_embeddings`.
            task_embeddings: (n_tasks, D_emb) tensor of BERT embeddings.
            frame_stack: Sliding window length H (BC training was 10).
            image_keys: RGB obs keys to load (must match policy's image_encoders).
            lowdim_keys: low-dim obs keys (must match policy's extra_encoder modalities).
            image_size: Expected H==W of RGB inputs (sanity check only).
            max_demos_per_task: Optional cap, useful for fast iteration / debugging.
        """
        assert len(hdf5_paths) == task_embeddings.shape[0], (
            f"hdf5_paths ({len(hdf5_paths)}) must match task_embeddings rows "
            f"({task_embeddings.shape[0]})"
        )
        self.hdf5_paths = list(hdf5_paths)
        self.task_embeddings = task_embeddings.float()  # (n_tasks, D_emb)
        self.frame_stack = int(frame_stack)
        self.image_keys = list(image_keys)
        self.lowdim_keys = list(lowdim_keys)
        self.image_size = image_size

        self.trajectories: List[Dict] = []      # per-traj raw arrays
        self.transition_index: List[tuple] = [] # (traj_idx, t)
        self.episode_starts: List[int] = []
        self.task_id_per_traj: List[int] = []

        for task_id, hdf5_path in enumerate(self.hdf5_paths):
            self._load_one_task(task_id, hdf5_path, max_demos_per_task)

        print(
            f"[LIBEROOPEDataset] loaded {len(self.trajectories)} trajectories, "
            f"{len(self.transition_index)} transitions across {len(self.hdf5_paths)} tasks "
            f"(frame_stack={self.frame_stack})"
        )

        # OPEDataset-compatible attrs (used by base trainer / algorithm)
        self.metadata = {
            "dataset_paths": self.hdf5_paths,
            "image_keys": self.image_keys,
            "lowdim_keys": self.lowdim_keys,
            "frame_stack": self.frame_stack,
            "n_tasks": len(self.hdf5_paths),
        }
        self.normalize_states = False
        self.normalize_rewards = False
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.state_mean: Dict = {}
        self.state_std: Dict = {}

    def _load_one_task(self, task_id: int, hdf5_path: str, cap: Optional[int]):
        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(
                [k for k in f["data"].keys() if k.startswith("demo_")],
                key=lambda x: int(x.split("_")[1]),
            )
            if cap is not None:
                demo_keys = demo_keys[:cap]

            for demo_key in demo_keys:
                d = f["data"][demo_key]
                actions = np.asarray(d["actions"][:], dtype=np.float32)
                rewards = np.asarray(d["rewards"][:], dtype=np.float32)
                dones = np.asarray(d["dones"][:], dtype=np.float32)
                T = actions.shape[0]

                obs = {}
                for k in self.image_keys:
                    arr = d["obs"][k][:]   # (T, H, W, C) uint8
                    if arr.shape[1] != self.image_size or arr.shape[2] != self.image_size:
                        raise ValueError(
                            f"Image '{k}' in {os.path.basename(hdf5_path)} has shape "
                            f"{arr.shape}, expected H==W=={self.image_size}"
                        )
                    # Permute to (T, C, H, W) once at load time, keep uint8 to save RAM.
                    obs[k] = np.transpose(arr, (0, 3, 1, 2)).copy()

                for k in self.lowdim_keys:
                    obs[k] = np.asarray(d["obs"][k][:], dtype=np.float32)

                traj_idx = len(self.trajectories)
                self.trajectories.append({
                    "obs": obs,
                    "actions": actions,
                    "rewards": rewards,
                    "dones": dones,
                    "T": T,
                    "task_id": task_id,
                })
                self.task_id_per_traj.append(task_id)
                self.episode_starts.append(len(self.transition_index))
                for t in range(T):
                    self.transition_index.append((traj_idx, t))

    @property
    def num_trajectories(self) -> int:
        return len(self.trajectories)

    def __len__(self) -> int:
        return len(self.transition_index)

    def _stack_obs(self, traj: Dict, t: int, end_offset: int = 0) -> Dict[str, torch.Tensor]:
        """Build a length-H stack of obs ending at t + end_offset."""
        T = traj["T"]
        idx = _stack_indices(t + end_offset, T, self.frame_stack)
        out = {}
        for k in self.image_keys:
            arr = traj["obs"][k][idx]  # (H, C, H_img, W_img) uint8
            t_arr = torch.from_numpy(arr).float() / 255.0
            out[k] = t_arr
        for k in self.lowdim_keys:
            arr = traj["obs"][k][idx]  # (H, D) float32
            out[k] = torch.from_numpy(arr)
        return out

    def __getitem__(self, idx: int) -> Dict:
        traj_idx, t = self.transition_index[idx]
        traj = self.trajectories[traj_idx]

        obs = self._stack_obs(traj, t, end_offset=0)
        next_obs = self._stack_obs(traj, t, end_offset=1)
        action = torch.from_numpy(traj["actions"][t])
        reward = torch.tensor(traj["rewards"][t], dtype=torch.float32)
        done = torch.tensor(traj["dones"][t], dtype=torch.float32)
        # All transitions inside a demo are valid; done=1 only at last step
        # so (1-done)*next_q already neutralizes the synthetic next_obs there.
        valid_mask = torch.tensor(1.0, dtype=torch.float32)
        task_id = int(traj["task_id"])
        task_emb = self.task_embeddings[task_id]

        return {
            "obs": obs,
            "next_obs": next_obs,
            "actions": action,
            "rewards": reward,
            "dones": done,
            "valid_mask": valid_mask,
            "task_emb": task_emb,
            # `task_id` is the per-suite task index (0..n_tasks-1). Not used
            # by the FQE variants in this repo (they read the BERT `task_emb`).
            "task_id": torch.tensor(task_id, dtype=torch.long),
        }

    def get_initial_states(self, num_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Frame-stacked initial obs of every trajectory (or a sample).

        Returns a flat dict where image/lowdim keys are stacked across N trajectories,
        plus a 'task_emb' key (N, D_emb) so the algorithm can pass it through
        to policy and Q-network.
        """
        if num_samples is None or num_samples >= self.num_trajectories:
            traj_indices = list(range(self.num_trajectories))
        else:
            traj_indices = np.random.choice(
                self.num_trajectories, num_samples, replace=False
            ).tolist()

        per_key: Dict[str, List[torch.Tensor]] = {}
        task_embs: List[torch.Tensor] = []
        task_ids: List[int] = []
        for ti in traj_indices:
            stacked = self._stack_obs(self.trajectories[ti], t=0, end_offset=0)
            for k, v in stacked.items():
                per_key.setdefault(k, []).append(v)
            tid = int(self.trajectories[ti]["task_id"])
            task_embs.append(self.task_embeddings[tid])
            task_ids.append(tid)

        out = {k: torch.stack(v, dim=0) for k, v in per_key.items()}
        out["task_emb"] = torch.stack(task_embs, dim=0)
        out["task_id"] = torch.tensor(task_ids, dtype=torch.long)
        return out


def find_libero_task_files(dataset_dir) -> List[str]:
    """All `*_demo.hdf5` files in benchmark directory/directories.

    Args:
        dataset_dir: either a single path string, or a list/tuple of paths.
            Multi-dir combines several LIBERO suites' HDF5 demos into one
            dataset.

    Sorting:
        Single-dir path: files sorted by basename inside that dir
            (preserves per-suite canonical task ordering).
        Multi-dir path: files sorted by (parent-dir-basename, file-basename).
            Parent-dir prefix in the sort key keeps each suite's tasks
            contiguous; the resulting full ordering is deterministic, which
            is what `compute_libero_task_embeddings` and the per-traj
            `task_id` (= index into this list) depend on.
    """
    if isinstance(dataset_dir, (list, tuple)):
        files = []
        for d in dataset_dir:
            d_files = glob.glob(os.path.join(os.path.expanduser(d), "*_demo.hdf5"))
            files.extend(d_files)
        files = sorted(
            files,
            key=lambda p: (os.path.basename(os.path.dirname(p)), os.path.basename(p)),
        )
        if not files:
            raise FileNotFoundError(
                f"No *_demo.hdf5 found under any of {list(dataset_dir)}"
            )
        return files

    files = sorted(glob.glob(os.path.join(os.path.expanduser(dataset_dir), "*_demo.hdf5")))
    if not files:
        raise FileNotFoundError(f"No *_demo.hdf5 found under {dataset_dir}")
    return files


def libero_task_descriptions_from_hdf5(hdf5_paths: Sequence[str]) -> List[str]:
    """Extract the task language description from each HDF5's `problem_info` attr."""
    import json

    out = []
    for p in hdf5_paths:
        with h5py.File(p, "r") as f:
            attrs = dict(f["data"].attrs)
            info = attrs.get("problem_info")
            if isinstance(info, (bytes, str)):
                if isinstance(info, bytes):
                    info = info.decode("utf-8")
                info = json.loads(info)
            lang = info.get("language_instruction") if info else None
            if not lang:
                # Fallback: derive from filename
                name = os.path.basename(p).replace("_demo.hdf5", "").replace("_", " ")
                lang = name
            out.append(lang)
    return out


def compute_libero_task_embeddings(
    descriptions: Sequence[str],
    embedding_format: str = "bert",
    cache_dir: str = "./bert",
) -> torch.Tensor:
    """Replicates LIBERO's `get_task_embs` for an explicit description list.

    Returns (n_tasks, D_emb) float tensor on CPU.
    """
    if embedding_format != "bert":
        raise NotImplementedError(
            f"Only 'bert' embedding_format is supported here (got {embedding_format})"
        )
    from transformers import AutoModel, AutoTokenizer, logging as hf_logging

    hf_logging.set_verbosity_error()
    tz = AutoTokenizer.from_pretrained("bert-base-cased", cache_dir=cache_dir)
    model = AutoModel.from_pretrained("bert-base-cased", cache_dir=cache_dir)
    tokens = tz(
        text=list(descriptions),
        add_special_tokens=True,
        max_length=25,
        padding="max_length",
        return_attention_mask=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        embs = model(**tokens)["pooler_output"].detach()
    return embs.float()


def libero_to_ope(
    dataset_dir,
    frame_stack: int = 10,
    image_keys: Sequence[str] = DEFAULT_IMAGE_KEYS,
    lowdim_keys: Sequence[str] = DEFAULT_LOWDIM_KEYS,
    embedding_format: str = "bert",
    bert_cache_dir: str = "./bert",
    max_demos_per_task: Optional[int] = None,
) -> LIBEROOPEDataset:
    """High-level builder: dataset_dir → LIBEROOPEDataset.

    `dataset_dir` may be a single path string, or a list/tuple of paths
    (multi-suite combined dataset).
    Discovers `*_demo.hdf5` under the given dir(s), reads their language
    descriptions, computes BERT embeddings once, and builds the dataset.
    """
    hdf5_paths = find_libero_task_files(dataset_dir)
    descriptions = libero_task_descriptions_from_hdf5(hdf5_paths)
    print(f"[libero_to_ope] {len(hdf5_paths)} task files; computing task embeddings...")
    task_embeddings = compute_libero_task_embeddings(
        descriptions, embedding_format=embedding_format, cache_dir=bert_cache_dir
    )
    return LIBEROOPEDataset(
        hdf5_paths=hdf5_paths,
        task_embeddings=task_embeddings,
        frame_stack=frame_stack,
        image_keys=image_keys,
        lowdim_keys=lowdim_keys,
        max_demos_per_task=max_demos_per_task,
    )
