"""Pretrained vision encoders for FQE.

Three modes, all driven by the FQE TD loss (no separate SSL pretraining):

  (a) Fully frozen     (freeze=True,  use_lora=False, n_finetune_blocks=0)
      Backbone weights have requires_grad=False; forward is wrapped in
      torch.no_grad(); only `self.proj` and downstream modules
      (frame_proj / temporal transformer / Q-head) receive gradients.

  (b) Frozen + LoRA    (freeze=True,  use_lora=True)
      Backbone base weights are frozen, but peft inserts low-rank LoRA
      modules (default rank 16, attn q/k/v only) that ARE trainable.
      Forward is NOT wrapped in no_grad — gradients flow through LoRA so
      the TD loss in FQE can adapt the features to the LIBERO domain.

  (c) Partial fine-tune (freeze=True, use_lora=False, n_finetune_blocks=K>0)
      Patch embedding + bottom (N-K) transformer blocks frozen; top K blocks
      + final LayerNorm + pooler trainable.  Gradients flow through the
      whole backbone (no torch.no_grad()) but `requires_grad=False` zeros
      out updates to the frozen prefix.  Intended for cases where LoRA's
      low-rank attn-only update is too restrictive but full FT is too
      expressive (5/22 result: LoRA didn't shift object/goal overshoot).

Patch-size note:
  - DINOv3 uses patch_size=16: 128² divides cleanly → no resize.
  - DINOv2 uses patch_size=14: 128² does NOT divide → encoder resizes to 224²
    internally (DINOv2's training resolution).
"""
import re
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Default LoRA target modules for HF ViT-style backbones.  q/k/v only is the
# safest / most-studied attn LoRA recipe; expand to o_proj or mlp.{up,down}_proj
# for more capacity at the cost of more trainable params.
DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj")

# Catches `.encoder.layer.7.`, `.layers.3.`, `.blocks.11.` etc. so we can
# identify which transformer block a parameter belongs to across HF naming
# variants (DINOv2 / DINOv3 / SigLIP / CLIP share roughly the same convention).
_LAYER_IDX_RE = re.compile(r"(?:^|\.)(?:encoder\.layer|layer|layers|blocks)\.(\d+)\.")


class _HFImageEncoder(nn.Module):
    """Common scaffold: HF AutoModel backbone (optionally frozen / LoRA-wrapped)
    + Linear projection to `out_dim`.

    Concrete subclasses set `_resize_to` (None = no resize, int N = bilinear
    resize to N×N before backbone) to handle patch-size divisibility.
    """

    _resize_to: int = None  # subclass overrides

    def __init__(
        self,
        model_name: str,
        out_dim: int,
        freeze: bool,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[Sequence[str]] = None,
        # Partial fine-tune (mode (c) in this module's docstring): keep the
        # backbone's patch embed + bottom (n_layers - n_finetune_blocks)
        # transformer blocks frozen, but let the top `n_finetune_blocks`
        # blocks (plus final LayerNorm / pooler) train via the FQE TD loss.
        # 0 = fully frozen (== mode (a)). Ignored when `use_lora=True`.
        n_finetune_blocks: int = 0,
    ):
        super().__init__()
        from transformers import AutoModel

        self.model_name = model_name
        self.backbone = AutoModel.from_pretrained(model_name)
        feat_dim = int(self.backbone.config.hidden_size)
        self.proj = nn.Linear(feat_dim, out_dim)
        self.out_dim = out_dim
        self.use_lora = bool(use_lora)
        self.n_finetune_blocks = int(n_finetune_blocks)
        # `self.frozen` is True only when ZERO backbone params are trainable
        # — that's the only case where we can wrap forward in torch.no_grad()
        # and lock the backbone to eval() mode for free perf. Partial FT and
        # LoRA both leave the backbone in the autograd graph.
        self.frozen = bool(freeze) and self.n_finetune_blocks == 0 and not self.use_lora

        if self.use_lora:
            # peft handles base-param freezing itself; we let it own that
            # responsibility so its bookkeeping (modules_to_save, etc.) stays
            # consistent. Our `frozen` flag still controls `eval()` mode below.
            from peft import LoraConfig, get_peft_model

            targets = list(lora_target_modules or DEFAULT_LORA_TARGETS)
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=targets,
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_cfg)
            self.lora_target_modules = targets
        elif freeze:
            self._apply_partial_freeze(self.n_finetune_blocks)

        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def _apply_partial_freeze(self, n_finetune_blocks: int):
        """Freeze patch_embed + bottom (N - n_finetune_blocks) transformer blocks.

        n_finetune_blocks == 0      → freeze ALL backbone params (mode (a)).
        n_finetune_blocks == N      → freeze only patch_embed; train all blocks
                                       + final LN + pooler.
        n_finetune_blocks in (0, N) → freeze bottom (N - k) blocks + embed;
                                       train top k blocks + final LN + pooler.

        Embedding modules (patch_embed conv, cls_token, register_tokens,
        position_embeddings) are ALWAYS frozen — adapting them would invalidate
        DINOv3's geometric pretraining priors and is essentially never the
        right call for downstream FT.
        """
        n_layers = int(getattr(self.backbone.config, "num_hidden_layers", 0))
        n_ft = max(0, min(int(n_finetune_blocks), n_layers))
        n_frozen_blocks = n_layers - n_ft

        for name, p in self.backbone.named_parameters():
            # Embeddings always frozen (catches embeddings.* / patch_embed /
            # position_embeddings on whatever the HF model calls them).
            if "embed" in name:
                p.requires_grad_(False)
                continue
            m = _LAYER_IDX_RE.search(name)
            if m is not None:
                idx = int(m.group(1))
                # Bottom n_frozen_blocks frozen, top n_ft trainable.
                if idx < n_frozen_blocks:
                    p.requires_grad_(False)
                # else: leave trainable (top block)
            else:
                # Top-level params (final LayerNorm, pooler, etc.) — frozen
                # only when n_ft == 0 (fully frozen mode); trainable when
                # we're doing partial FT (they sit ABOVE the blocks).
                if n_ft == 0:
                    p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen and not self.use_lora:
            # Fully-frozen path: lock backbone to eval (kills any latent BN/dropout).
            self.backbone.eval()
        # LoRA path: let backbone follow parent train/eval so LoRA dropout
        # behaves correctly during training.
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) float in [0, 1]
        if self._resize_to is not None and (
            x.shape[-2] != self._resize_to or x.shape[-1] != self._resize_to
        ):
            x = F.interpolate(
                x, size=(self._resize_to, self._resize_to),
                mode="bilinear", align_corners=False,
            )
        x = (x - self._mean) / self._std
        if self.frozen and not self.use_lora:
            # Fully-frozen path: no autograd through the backbone at all.
            with torch.no_grad():
                out = self.backbone(pixel_values=x)
        else:
            # LoRA path (or fully unfrozen): backbone must be in the autograd
            # graph so gradients reach LoRA matrices / base params.
            out = self.backbone(pixel_values=x)
        feat = getattr(out, "pooler_output", None)
        if feat is None:
            feat = out.last_hidden_state[:, 0]
        return self.proj(feat)


class DINOv3ImageEncoder(_HFImageEncoder):
    """DINOv3 ViT trunk (patch16). 128² LIBERO input works natively (no resize)."""

    _resize_to = None

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        out_dim: int = 128,
        freeze: bool = True,
        **lora_kwargs,
    ):
        super().__init__(model_name, out_dim, freeze, **lora_kwargs)


class DINOv2ImageEncoder(_HFImageEncoder):
    """DINOv2 ViT trunk (patch14). 128² is not /14 → resize to 224² internally.

    Stand-in for DINOv3 while DINOv3 HF access is being granted; same algorithm
    code path, only the encoder tag differs.
    """

    _resize_to = 224

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        out_dim: int = 128,
        freeze: bool = True,
        **lora_kwargs,
    ):
        super().__init__(model_name, out_dim, freeze, **lora_kwargs)


def build_pretrained_encoder(
    name: str,
    out_dim: int = 128,
    freeze: bool = True,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[Sequence[str]] = None,
    n_finetune_blocks: int = 0,
) -> nn.Module:
    """Factory: map a short tag → concrete encoder module.

    Tags are kept short for ergonomic CLI / config use. Add new entries here
    when wiring in VC-1, R3M, SigLIP, CLIP, etc.

    LoRA kwargs are ignored when `use_lora=False`; passing them explicitly
    selects "frozen base + trainable LoRA adapters" mode, trained jointly
    with the FQE head by the same TD loss.

    `n_finetune_blocks` selects partial fine-tune mode (c): when > 0 and
    `use_lora=False`, the top K transformer blocks (+ final LN / pooler)
    are trainable while the bottom (N-K) blocks + patch embed stay frozen.
    Pairs with per-param-group AdamW in `algorithm.py` (lower lr for the
    unfrozen backbone slice than for the Q-head).
    """
    tag = name.lower()
    common = dict(
        out_dim=out_dim,
        freeze=freeze,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        n_finetune_blocks=n_finetune_blocks,
    )
    if tag in ("dinov3_vits16", "dinov3-vits16"):
        return DINOv3ImageEncoder(
            "facebook/dinov3-vits16-pretrain-lvd1689m", **common
        )
    if tag in ("dinov3_vitb16", "dinov3-vitb16"):
        return DINOv3ImageEncoder(
            "facebook/dinov3-vitb16-pretrain-lvd1689m", **common
        )
    if tag in ("dinov3_vitl16", "dinov3-vitl16"):
        return DINOv3ImageEncoder(
            "facebook/dinov3-vitl16-pretrain-lvd1689m", **common
        )
    if tag in ("dinov2_vits14", "dinov2-vits14", "dinov2_small"):
        return DINOv2ImageEncoder("facebook/dinov2-small", **common)
    if tag in ("dinov2_vitb14", "dinov2-vitb14", "dinov2_base"):
        return DINOv2ImageEncoder("facebook/dinov2-base", **common)
    if tag in ("dinov2_vitl14", "dinov2-vitl14", "dinov2_large"):
        return DINOv2ImageEncoder("facebook/dinov2-large", **common)
    raise ValueError(f"Unknown pretrained encoder tag: {name!r}")
