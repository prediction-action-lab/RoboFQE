"""Frame-stack-aware vision Q-network with a pretrained image encoder.

Identical to `fqe.scratch.networks.VisionQNetwork` except:
  - The per-camera image encoder is a frozen pretrained backbone (default
    DINOv3 ViT-S/16) + small trainable Linear projection, instead of a
    from-scratch ResNet-18.
  - Cameras may share the same encoder instance (`share_encoders=True`) to
    avoid duplicating ~22M frozen parameters per camera; the from-scratch
    path always keeps independent towers.

The frame_proj + learnable positional embedding + Transformer temporal
aggregator + MLP head are kept identical to `fqe.scratch`, so any swap is local
to the encoder.
"""
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .encoders import build_pretrained_encoder


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


class VisionQNetworkPretrained(nn.Module):
    """Frame-stack-aware Q(obs_seq, action, task_emb) with pretrained encoder."""

    def __init__(
        self,
        image_keys: Sequence[str],
        image_shape: Sequence[int],            # (C, H, W) per camera
        lowdim_keys: Sequence[str],
        lowdim_dims: Dict[str, int],
        action_dim: int,
        task_emb_dim: int,
        frame_stack: int,
        encoder_name: str = "dinov3_vits16",
        freeze_backbone: bool = True,
        share_encoders: bool = True,
        # LoRA on the backbone (ignored when use_lora=False).  When True the
        # base backbone stays frozen but peft inserts trainable rank-r LoRA
        # adapters on the target attn projections; the FQE TD loss drives them.
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[Sequence[str]] = None,
        # Partial fine-tune knob (mutually exclusive with use_lora). When > 0
        # and freeze_backbone=True, the top K transformer blocks + final LN +
        # pooler are trainable; bottom (N-K) blocks + patch embed stay frozen.
        # See `fqe.pretrained.encoders._HFImageEncoder._apply_partial_freeze`.
        n_finetune_blocks: int = 0,
        encoder_out_dim: int = 128,            # post-projection per-image token width
        token_dim: int = 256,                  # per-frame fused-token width
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        temporal_ff: int = 512,
        head_hidden: Sequence[int] = (256, 256),
        head_out_dim: int = 1,
        activation: str = "relu",
    ):
        super().__init__()
        self.image_keys = list(image_keys)
        self.lowdim_keys = list(lowdim_keys)
        self.frame_stack = int(frame_stack)
        self.head_out_dim = head_out_dim
        self.share_encoders = bool(share_encoders)

        enc_kwargs = dict(
            out_dim=encoder_out_dim,
            freeze=freeze_backbone,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            n_finetune_blocks=n_finetune_blocks,
        )
        if self.share_encoders:
            shared = build_pretrained_encoder(encoder_name, **enc_kwargs)
            # ModuleDict mapping camera→encoder so the forward path stays
            # symmetric whether shared or not.  All keys reference the same
            # instance under sharing.
            self.image_encoders = nn.ModuleDict({k: shared for k in self.image_keys})
        else:
            self.image_encoders = nn.ModuleDict({
                k: build_pretrained_encoder(encoder_name, **enc_kwargs)
                for k in self.image_keys
            })

        per_frame_in = (
            len(self.image_keys) * encoder_out_dim
            + sum(lowdim_dims[k] for k in self.lowdim_keys)
        )
        self.frame_proj = nn.Linear(per_frame_in, token_dim)
        self.frame_act = _activation(activation)
        self.pos_emb = nn.Parameter(torch.zeros(1, frame_stack, token_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

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
        # Small init keeps initial Q close to 0 (helpful under sparse rewards).
        last = self.head[-1]
        if isinstance(last, nn.Linear):
            nn.init.orthogonal_(last.weight, gain=0.01)
            nn.init.zeros_(last.bias)

    def encode_obs_seq(
        self,
        obs_seq: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """obs_seq → (B, token_dim) via per-frame fuse + temporal transformer."""
        B = obs_seq[self.image_keys[0]].shape[0]
        H = self.frame_stack

        per_frame_feats: List[torch.Tensor] = []

        for k in self.image_keys:
            x = obs_seq[k]                # (B, H, C, H_img, W_img)
            assert x.shape[1] == H, f"obs_seq[{k}] has H={x.shape[1]}, expected {H}"
            c, hi, wi = x.shape[-3:]
            f = self.image_encoders[k](x.reshape(B * H, c, hi, wi))   # (B*H, E_img)
            per_frame_feats.append(f.view(B, H, -1))

        for k in self.lowdim_keys:
            per_frame_feats.append(obs_seq[k])

        fused = torch.cat(per_frame_feats, dim=-1)             # (B, H, in)
        tokens = self.frame_act(self.frame_proj(fused))        # (B, H, token_dim)
        tokens = tokens + self.pos_emb
        tokens = self.temporal(tokens)
        return tokens[:, -1]

    def forward(
        self,
        obs_seq: Dict[str, torch.Tensor],
        action: torch.Tensor,
        task_emb: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encode_obs_seq(obs_seq)
        x = torch.cat([z, task_emb, action], dim=-1)
        return self.head(x)

    def trainable_parameters(self):
        """Iterator over params whose `.requires_grad` is True — convenience for
        the optimizer when the backbone is frozen.
        """
        return (p for p in self.parameters() if p.requires_grad)
