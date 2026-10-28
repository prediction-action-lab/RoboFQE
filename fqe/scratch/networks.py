"""
Frame-stack-aware vision Q-network for FQE on LIBERO.

Design (Phase 1, from-scratch):
  per-frame ResNet-18 image encoder (one shared module per camera key)
   → concat per-frame image tokens with low-dim proprio
   → small temporal transformer over H tokens
   → take last token   (matches policy's `x[:, -1]` convention)
   → concat task_emb + action
   → 2-layer MLP → scalar Q

Phase 2 plan: swap `self.image_encoder` for a frozen pretrained encoder
(R3M / VC-1 / DINOv2 / etc.) — the encoder slot is a single module per camera
so the swap is local.

Ported verbatim from opebench/algorithms/fqe/networks_vision.py
of RoboOPE main commit 6c2c61a (author: Sangwoo Shin <jsw7460@gmail.com>).
"""
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torchvision


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU(inplace=True)
    raise ValueError(f"Unknown activation: {name}")


class ResNet18ImageEncoder(nn.Module):
    """ResNet-18 trunk + global avg pool + linear projection to `out_dim`.

    From-scratch (no pretrained weights). Replaces the final FC with a
    custom projection so dimensionality is configurable.
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 128):
        super().__init__()
        net = torchvision.models.resnet18(weights=None)
        if in_channels != 3:
            net.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        # drop the final fc; keep avgpool
        self.backbone = nn.Sequential(*list(net.children())[:-1])  # → (B, 512, 1, 1)
        self.proj = nn.Linear(512, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x).flatten(1)  # (B, 512)
        return self.proj(h)


class VisionQNetwork(nn.Module):
    """Frame-stack-aware Q(obs_seq, action, task_emb)."""

    def __init__(
        self,
        image_keys: Sequence[str],
        image_shape: Sequence[int],            # (C, H, W) per camera
        lowdim_keys: Sequence[str],
        lowdim_dims: Dict[str, int],
        action_dim: int,
        task_emb_dim: int,
        frame_stack: int,
        encoder_out_dim: int = 128,            # per-image token width
        token_dim: int = 256,                  # per-frame fused-token width
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        temporal_ff: int = 512,
        head_hidden: Sequence[int] = (256, 256),
        head_out_dim: int = 1,                 # 1 for scalar FQE; num_bins for FQE-Cat
        activation: str = "relu",
    ):
        super().__init__()
        self.image_keys = list(image_keys)
        self.lowdim_keys = list(lowdim_keys)
        self.frame_stack = int(frame_stack)
        self.head_out_dim = head_out_dim

        c, h, w = image_shape
        # one encoder per camera (independent params; they look at different views)
        self.image_encoders = nn.ModuleDict({
            k: ResNet18ImageEncoder(in_channels=c, out_dim=encoder_out_dim)
            for k in self.image_keys
        })

        # per-frame fused token
        per_frame_in = (
            len(self.image_keys) * encoder_out_dim
            + sum(lowdim_dims[k] for k in self.lowdim_keys)
        )
        self.frame_proj = nn.Linear(per_frame_in, token_dim)
        self.frame_act = _activation(activation)
        self.pos_emb = nn.Parameter(torch.zeros(1, frame_stack, token_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # temporal aggregator
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=temporal_heads,
            dim_feedforward=temporal_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)

        # head: token + task_emb + action → MLP → out
        head_in = token_dim + task_emb_dim + action_dim
        layers: List[nn.Module] = []
        prev = head_in
        for hd in head_hidden:
            layers += [nn.Linear(prev, hd), _activation(activation)]
            prev = hd
        layers += [nn.Linear(prev, head_out_dim)]
        self.head = nn.Sequential(*layers)

        self._init_head()

    def _init_head(self):
        # Small init for the final layer keeps initial Q close to 0 (helpful
        # under sparse rewards).
        last = self.head[-1]
        if isinstance(last, nn.Linear):
            nn.init.orthogonal_(last.weight, gain=0.01)
            nn.init.zeros_(last.bias)

    def encode_obs_seq(
        self,
        obs_seq: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """obs_seq → (B, token_dim)  via per-frame fuse + temporal transformer."""
        B = obs_seq[self.image_keys[0]].shape[0]
        H = self.frame_stack

        per_frame_feats: List[torch.Tensor] = []

        # Image branches: encode all (B*H) frames in one shot per camera
        for k in self.image_keys:
            x = obs_seq[k]                # (B, H, C, H_img, W_img)
            assert x.shape[1] == H, f"obs_seq[{k}] has H={x.shape[1]}, expected {H}"
            c, hi, wi = x.shape[-3:]
            f = self.image_encoders[k](x.reshape(B * H, c, hi, wi))  # (B*H, E_img)
            per_frame_feats.append(f.view(B, H, -1))

        # Lowdim branches: already (B, H, D)
        for k in self.lowdim_keys:
            per_frame_feats.append(obs_seq[k])

        fused = torch.cat(per_frame_feats, dim=-1)             # (B, H, in)
        tokens = self.frame_act(self.frame_proj(fused))        # (B, H, token_dim)
        tokens = tokens + self.pos_emb                          # add learnable PE
        tokens = self.temporal(tokens)                          # (B, H, token_dim)
        return tokens[:, -1]                                    # last frame: (B, token_dim)

    def forward(
        self,
        obs_seq: Dict[str, torch.Tensor],
        action: torch.Tensor,
        task_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Returns (B, head_out_dim)."""
        z = self.encode_obs_seq(obs_seq)
        x = torch.cat([z, task_emb, action], dim=-1)
        return self.head(x)
