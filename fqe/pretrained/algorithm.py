"""FQE with frame-stack obs + task-conditioned Q + frozen pretrained vision encoder.

Forked from `fqe.scratch.algorithm.FittedQEvaluationVision`.
Same training loop and target-update logic; the only difference is:
  - Q-net is `VisionQNetworkPretrained` (pretrained image encoder, default DINOv3).
  - Optimizer is built over `q_network.trainable_parameters()` so a frozen
    backbone doesn't get wired into Adam state.
"""
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim

from .networks import VisionQNetworkPretrained
from ..base import BaseOPEAlgorithm


class FittedQEvaluationVisionPretrained(BaseOPEAlgorithm):
    """FQE with frame-stack-aware Q-net using a pretrained image encoder."""

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
        loss_type: str = "mse",
        target_update_type: str = "soft",
        tau: float = 0.005,
        target_update_freq: int = 1,
        clip_target: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        grad_clip: Optional[float] = 1.0,
        # Pretrained-encoder hyperparams
        encoder_name: str = "dinov3_vits16",
        freeze_backbone: bool = True,
        share_encoders: bool = True,
        # LoRA on the backbone (ignored when use_lora=False).  When True the
        # base backbone stays frozen but trainable rank-r LoRA adapters on
        # the target attn projections are added; FQE TD loss drives them
        # jointly with the head — no separate SSL pretraining step.
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[Sequence[str]] = None,
        # Partial fine-tune knob (mutually exclusive with use_lora). When > 0
        # the top K transformer blocks + final LN + pooler are trainable;
        # bottom blocks + patch embed stay frozen. Paired with `backbone_lr`
        # (typically 10x smaller than `head_lr`) via a per-param-group AdamW.
        n_finetune_blocks: int = 0,
        # When set, `backbone_lr` is used for trainable backbone params
        # (LoRA matrices in LoRA mode; top-K block weights in partial-FT
        # mode) and `head_lr` for everything else (per-image proj, frame
        # proj, temporal Transformer, Q-head MLP). Both default to
        # `learning_rate` for backward compat with the frozen / LoRA paths.
        backbone_lr: Optional[float] = None,
        head_lr: Optional[float] = None,
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
            encoder_name=encoder_name,
            freeze_backbone=freeze_backbone,
            share_encoders=share_encoders,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            n_finetune_blocks=n_finetune_blocks,
            encoder_out_dim=encoder_out_dim,
            token_dim=token_dim,
            temporal_layers=temporal_layers,
            temporal_heads=temporal_heads,
            temporal_ff=temporal_ff,
            head_hidden=head_hidden,
            head_out_dim=1,
            activation=activation,
        )
        self.q_network = VisionQNetworkPretrained(**net_kwargs).to(self.device)
        self.target_q_network = VisionQNetworkPretrained(**net_kwargs).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        for p in self.target_q_network.parameters():
            p.requires_grad_(False)

        # Per-param-group AdamW so partial-FT / LoRA modes can use a lower
        # learning rate on the (trainable slice of) backbone vs. the head.
        # `image_encoders.*` covers per-camera DINOv3 backbones + their proj;
        # everything else is head/aggregator (frame_proj, pos_emb, temporal
        # Transformer, Q-head). Empty groups are silently skipped so the
        # fully-frozen case (no trainable backbone params) still works.
        bb_lr = backbone_lr if backbone_lr is not None else learning_rate
        hd_lr = head_lr if head_lr is not None else learning_rate
        bb_params = [
            p for n, p in self.q_network.named_parameters()
            if n.startswith("image_encoders") and p.requires_grad
        ]
        head_params = [
            p for n, p in self.q_network.named_parameters()
            if not n.startswith("image_encoders") and p.requires_grad
        ]
        param_groups = []
        if bb_params:
            param_groups.append({"params": bb_params, "lr": bb_lr})
        if head_params:
            param_groups.append({"params": head_params, "lr": hd_lr})
        self.optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)

        if loss_type not in ("mse", "huber"):
            raise ValueError(f"Unknown loss_type: {loss_type}")
        self.loss_type = loss_type
        self.update_step = 0

    def _to_device(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in obs.items()}

    def _soft_update(self):
        for p, tp in zip(
            self.q_network.parameters(), self.target_q_network.parameters()
        ):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def _hard_update(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

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

        q_values = self.q_network(obs, actions, task_emb).squeeze(-1)

        with torch.no_grad():
            next_actions = policy.get_action(
                next_obs, task_emb=task_emb, deterministic=False
            )
            next_q = self.target_q_network(
                next_obs, next_actions, task_emb
            ).squeeze(-1)
            target_q = rewards + self.gamma * (1.0 - dones) * next_q
            if self.clip_target:
                target_q = target_q.clamp(self.clip_min, self.clip_max)

        td_error = q_values - target_q
        if self.loss_type == "mse":
            element_loss = td_error ** 2
        else:
            abs_e = td_error.abs()
            element_loss = torch.where(
                abs_e <= 1.0, 0.5 * td_error ** 2, abs_e - 0.5
            )
        loss = (element_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(
                list(self.q_network.trainable_parameters()),
                max_norm=self.grad_clip,
            )
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

    def estimate(
        self,
        dataloader,
        policy,
        behavior_policy=None,
        initial_states: Optional[Dict[str, torch.Tensor]] = None,
        num_action_samples: int = 10,
        eval_chunk_size: int = 64,
    ) -> Dict[str, float]:
        # Chunk over init states so memory peak during eval stays close to the
        # training-step peak. Without chunking, libero_10 (~500 init states)
        # triggers a ~5 GiB allocation in the policy's ResNet-18 forward,
        # which OOMs under concurrent sweep runs sharing one GPU.
        self.q_network.eval()
        dataset = dataloader.dataset

        if initial_states is None:
            initial_states = dataset.get_initial_states()
        init_state_dict = {k: v.to(self.device) for k, v in initial_states.items()}
        task_emb = init_state_dict.pop("task_emb")
        init_obs_seq = init_state_dict
        n_init = next(iter(init_obs_seq.values())).shape[0]

        with torch.no_grad():
            q_samples = []
            for _ in range(num_action_samples):
                chunk_qs = []
                for start in range(0, n_init, eval_chunk_size):
                    end = min(start + eval_chunk_size, n_init)
                    chunk_obs = {k: v[start:end] for k, v in init_obs_seq.items()}
                    chunk_te = task_emb[start:end]
                    a = policy.get_action(
                        chunk_obs, task_emb=chunk_te, deterministic=False
                    )
                    q = self.q_network(chunk_obs, a, chunk_te).squeeze(-1)
                    chunk_qs.append(q)
                q_samples.append(torch.cat(chunk_qs, dim=0))
            v_initial = torch.stack(q_samples, dim=0).mean(dim=0)
            dm = v_initial.mean().item()

        self.q_network.train()
        return {"dm": dm}

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
