"""FQE specialized for LIBERO frame-stack obs + task-conditioned Q + vision policy.

Ported verbatim from opebench/algorithms/fqe/algorithm_vision.py
of RoboOPE main commit 6c2c61a (author: Sangwoo Shin <jsw7460@gmail.com>).
Only the import for BaseOPEAlgorithm was rewritten to the local relative path.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .networks import VisionQNetwork
from ..base import BaseOPEAlgorithm


class FittedQEvaluationVision(BaseOPEAlgorithm):
    """FQE with frame-stack-aware vision Q-network.

    Differences from `FittedQEvaluation`:
      * obs is a dict of (B, H, ...) frame stacks (image + low-dim).
      * batch carries `task_emb` (B, D_emb); routed to both Q-net and policy.
      * `estimate` returns DM only (no DR; LIBERO demos lack a behavior policy).
    """

    def __init__(
        self,
        image_keys,
        image_shape,
        lowdim_keys,
        lowdim_dims,
        action_dim: int,
        task_emb_dim: int,
        frame_stack: int,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        loss_type: str = "mse",                  # ["mse", "huber"]
        target_update_type: str = "soft",        # ["soft", "hard"]
        tau: float = 0.005,
        target_update_freq: int = 1,
        clip_target: bool = True,                # sparse binary reward → clip [0,1]
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        grad_clip: Optional[float] = 1.0,
        # vision-net hyperparams
        encoder_out_dim: int = 128,
        token_dim: int = 256,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        temporal_ff: int = 512,
        head_hidden=(256, 256),
        activation: str = "relu",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.gamma = gamma
        self.target_update_type = target_update_type
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.clip_target = clip_target
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.grad_clip = grad_clip
        self.device = torch.device(device)

        net_kwargs = dict(
            image_keys=image_keys,
            image_shape=image_shape,
            lowdim_keys=lowdim_keys,
            lowdim_dims=lowdim_dims,
            action_dim=action_dim,
            task_emb_dim=task_emb_dim,
            frame_stack=frame_stack,
            encoder_out_dim=encoder_out_dim,
            token_dim=token_dim,
            temporal_layers=temporal_layers,
            temporal_heads=temporal_heads,
            temporal_ff=temporal_ff,
            head_hidden=head_hidden,
            head_out_dim=1,
            activation=activation,
        )
        self.q_network = VisionQNetwork(**net_kwargs).to(self.device)
        self.target_q_network = VisionQNetwork(**net_kwargs).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        for p in self.target_q_network.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.AdamW(
            self.q_network.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        if loss_type == "mse":
            self.loss_type = "mse"
        elif loss_type == "huber":
            self.loss_type = "huber"
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        self.update_step = 0

    # -- helpers ------------------------------------------------------------
    def _to_device(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in obs.items()}

    def _soft_update(self):
        for p, tp in zip(self.q_network.parameters(), self.target_q_network.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def _hard_update(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    # -- core ---------------------------------------------------------------
    def update(
        self,
        batch: Dict[str, torch.Tensor],
        initial_batch: Dict[str, torch.Tensor],
        policy,
    ) -> Dict[str, float]:
        obs = self._to_device(batch["obs"])
        next_obs = self._to_device(batch["next_obs"])
        actions = batch["actions"].to(self.device, non_blocking=True)
        rewards = batch["rewards"].to(self.device, non_blocking=True)
        dones = batch["dones"].to(self.device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(self.device, non_blocking=True)
        task_emb = batch["task_emb"].to(self.device, non_blocking=True)

        # current Q(s, a)
        q_values = self.q_network(obs, actions, task_emb).squeeze(-1)

        # target: r + γ (1-done) Q_target(s', π(s'))
        with torch.no_grad():
            next_actions = policy.get_action(next_obs, task_emb=task_emb, deterministic=False)
            next_q = self.target_q_network(next_obs, next_actions, task_emb).squeeze(-1)
            target_q = rewards + self.gamma * (1.0 - dones) * next_q
            if self.clip_target:
                target_q = target_q.clamp(self.clip_min, self.clip_max)

        td_error = q_values - target_q
        if self.loss_type == "mse":
            element_loss = td_error ** 2
        else:  # huber, δ=1
            abs_e = td_error.abs()
            element_loss = torch.where(abs_e <= 1.0, 0.5 * td_error ** 2, abs_e - 0.5)
        loss = (element_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()

        self.update_step += 1
        if self.target_update_type == "soft":
            self._soft_update()
        elif self.update_step % self.target_update_freq == 0:
            self._hard_update()

        return {
            "q_loss": loss.item(),
            "q_mean": q_values.mean().item(),
            "target_q_mean": target_q.mean().item(),
            "td_error": td_error.abs().mean().item(),
        }

    # -- estimation: DM only ------------------------------------------------
    def estimate(
        self,
        dataloader,
        policy,
        behavior_policy=None,                   # ignored — DM only for LIBERO
        initial_states: Optional[Dict[str, torch.Tensor]] = None,
        num_action_samples: int = 10,
    ) -> Dict[str, float]:
        self.q_network.eval()
        dataset = dataloader.dataset

        if initial_states is None:
            initial_states = dataset.get_initial_states()
        # Pop task_emb out of the obs dict before passing to Q-net / policy.
        init_state_dict = {k: v.to(self.device) for k, v in initial_states.items()}
        task_emb = init_state_dict.pop("task_emb")
        init_obs_seq = init_state_dict

        with torch.no_grad():
            q_samples = []
            for _ in range(num_action_samples):
                a = policy.get_action(init_obs_seq, task_emb=task_emb, deterministic=False)
                q = self.q_network(init_obs_seq, a, task_emb).squeeze(-1)
                q_samples.append(q)
            v_initial = torch.stack(q_samples, dim=0).mean(dim=0)
            dm = v_initial.mean().item()

        self.q_network.train()
        return {"dm": dm}

    # -- I/O ---------------------------------------------------------------
    def save(self, path: str):
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_q_network_state_dict": self.target_q_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "update_step": self.update_step,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(ckpt["q_network_state_dict"])
        self.target_q_network.load_state_dict(ckpt["target_q_network_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.update_step = ckpt["update_step"]
