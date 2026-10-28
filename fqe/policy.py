"""
LIBERO BC-policy wrapper for OPE.

Bypasses BCTransformerPolicy.get_action's stateful `latent_queue`.  Instead,
expects the caller to provide an explicit length-H frame stack (matching the
policy's training-time input shape exactly).  This is mandatory for FQE which
samples shuffled batches of transitions.

Compatible with `BCTransformerPolicy` and `BCRNNPolicy` (anything inheriting
LIBERO's BasePolicy that exposes `spatial_encode`, `temporal_encode`,
`policy_head`).  For BCRNNPolicy a thinner forward path may be needed; current
implementation targets BCTransformerPolicy specifically since that is the
target policy choice for our experiments.

Ported verbatim from opebench/policies/libero_wrapper.py of RoboOPE main
commit 6c2c61a (author: Sangwoo Shin <jsw7460@gmail.com>). Only the import
path for PolicyWrapper was rewritten to the local relative path.
"""
from typing import Dict, Optional

import torch

from .base import PolicyWrapper


class LIBEROPolicyWrapper(PolicyWrapper):
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        previous_mask_aware: bool = False,
    ):
        """
        Args:
            ckpt_path: Path to a `task{i}_model.pth` (or multitask) checkpoint
                produced by `libero/lifelong/main.py`.
            device: Device for the policy network.
            previous_mask_aware: If True, also load `previous_masks` (only
                meaningful for PackNet checkpoints; ignored otherwise).
        """
        super().__init__()
        # Imports deferred so the rest of the package remains importable
        # without LIBERO installed.
        from libero.lifelong.models import get_policy_class

        self.device = torch.device(device)
        self.ckpt_path = ckpt_path

        # NOTE: LIBERO's torch_load_model() calls torch.load() without
        # weights_only=False, which fails under PyTorch 2.6+ because the
        # checkpoint pickles an EasyDict cfg.  Inline the load here with
        # weights_only=False (LIBERO checkpoints are from a trusted source).
        ckpt = torch.load(
            ckpt_path, map_location=str(self.device), weights_only=False
        )
        sd = ckpt["state_dict"]
        cfg = ckpt.get("cfg")
        previous_mask = ckpt.get("previous_masks")
        if cfg is None:
            raise ValueError(f"Checkpoint {ckpt_path} has no embedded cfg")
        # Override device on the recovered cfg so module construction uses the
        # right device.
        cfg.device = str(self.device)
        if not hasattr(cfg, "shape_meta"):
            raise ValueError(
                f"Checkpoint {ckpt_path} has no shape_meta in cfg; not a "
                f"LIBERO BC checkpoint?"
            )

        policy_cls = get_policy_class(cfg.policy.policy_type)
        self.policy = policy_cls(cfg, cfg.shape_meta).to(self.device)
        self.policy.load_state_dict(sd)
        if previous_mask_aware and previous_mask is not None:
            self.policy.previous_mask = previous_mask
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)

        self.cfg = cfg
        self.shape_meta = cfg.shape_meta
        # Squashed-tanh: GMM means already pass through tanh inside policy_head,
        # so no extra tanh here.

    # ------------------------------------------------------------------
    # Internal: LIBERO-format forward without latent_queue
    # ------------------------------------------------------------------
    def _build_data(
        self,
        obs_seq: Dict[str, torch.Tensor],
        task_emb: torch.Tensor,
    ) -> Dict:
        """Construct the dict shape BCTransformerPolicy.spatial_encode expects.

        obs_seq[image_key]:   (B, H, C, H_img, W_img)
        obs_seq[lowdim_key]:  (B, H, D)
        task_emb:             (B, D_emb)
        """
        out_obs = {}
        for k, v in obs_seq.items():
            out_obs[k] = v.to(self.device, non_blocking=True)
        return {
            "obs": out_obs,
            "task_emb": task_emb.to(self.device, non_blocking=True),
        }

    def _forward_dist(
        self,
        obs_seq: Dict[str, torch.Tensor],
        task_emb: torch.Tensor,
    ):
        """Last-step action distribution given an explicit frame stack."""
        data = self._build_data(obs_seq, task_emb)
        x = self.policy.spatial_encode(data)      # (B, T, num_modalities, E)
        x = self.policy.temporal_encode(x)        # (B, T, E)
        dist = self.policy.policy_head(x[:, -1])  # last step → GMM (or whatever head)
        return dist

    # ------------------------------------------------------------------
    # PolicyWrapper interface
    # ------------------------------------------------------------------
    def get_action(
        self,
        obs_seq: Dict[str, torch.Tensor],
        task_emb: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample (or take mean of) the GMM action distribution.

        For OPE estimation, deterministic=False matches how the policy actually
        behaves at test time inside LIBERO env (`dist.sample()`).
        """
        if task_emb is None:
            raise ValueError(
                "LIBEROPolicyWrapper.get_action requires task_emb (B, D_emb); "
                "pass it explicitly from the batch."
            )
        with torch.no_grad():
            dist = self._forward_dist(obs_seq, task_emb)
            if deterministic:
                # Mixture mean = weighted sum of component means
                # MixtureSameFamily exposes `mean` as the marginal mean.
                action = dist.mean
            else:
                action = dist.sample()
            return action

    def get_log_prob(
        self,
        obs_seq: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        task_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """log π(a | s_seq).  Returns shape (B,).

        Not used for DM-only OPE; provided so DR / IS can be added later.
        """
        if task_emb is None:
            raise ValueError("LIBEROPolicyWrapper.get_log_prob requires task_emb")
        with torch.no_grad():
            dist = self._forward_dist(obs_seq, task_emb)
            log_prob = dist.log_prob(actions.to(self.device))
            if log_prob.ndim > 1:
                log_prob = log_prob.sum(dim=-1)
            return log_prob
