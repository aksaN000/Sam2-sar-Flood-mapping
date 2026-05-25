"""SAM 2 Hiera-Base-Plus with parameter-efficient fine-tuning adapters.

Loads HuggingFace's facebook/sam2-hiera-base-plus, freezes the image and
prompt encoders, injects one of four PEFT mechanisms into every Hiera
transformer block, and adapts the mask decoder via full fine-tuning or
LoRA rank-4 adapters.

The forward path is the same as SAMAdapter (whole-chip box prompt,
single-channel logit map at the input resolution), so the two adapters
are interchangeable as drop-in modules inside the Lightning training
script.

The thesis describes routing the pre-event chip through SAM 2's streaming
memory module and decoding the post-event chip conditional on memory.
For the principal evaluation in this thesis we use the post-event chip
only (matching the single-frame protocol used by DAM-Net's class token
and by CWSAM), and the memory module is documented in `future_work/`
as the natural extension once bi-temporal Sen1Floods11 pairs are
available. The `use_memory` flag here is reserved for that future use.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Sam2Model

from .peft_methods import (
    AdaptFormerFFN,
    PEFTMethod,
    count_trainable,
    replace_submodule,
    wrap_linear,
)

MaskDecoderStrategy = Literal["full-ft", "lora-r4"]

# Hiera transformer blocks expose the same attribute names as SAM ViT-B:
# attn.qkv (fused), attn.proj (output), mlp.proj_in (FFN up), mlp.proj_out (FFN down).
_ATTN_TARGETS = ["attn.qkv", "attn.proj"]


class SAM2Adapter(nn.Module):
    """SAM 2 Hiera-Base-Plus with a chosen PEFT mechanism."""

    def __init__(
        self,
        pretrained_id: str = "facebook/sam2-hiera-base-plus",
        peft_method: PEFTMethod = "lora",
        rank: int = 8,
        adaptformer_dim: int = 64,
        mask_decoder_strategy: MaskDecoderStrategy = "full-ft",
        use_memory: bool = False,
    ) -> None:
        super().__init__()
        self.peft_method = peft_method
        self.rank = rank
        self.adaptformer_dim = adaptformer_dim
        self.mask_decoder_strategy = mask_decoder_strategy
        self.use_memory = use_memory

        self.backbone = Sam2Model.from_pretrained(pretrained_id)
        self._freeze_all()
        self._inject_peft_into_encoder()
        self._configure_mask_decoder()

    def _freeze_all(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def _inject_peft_into_encoder(self) -> None:
        # Hiera blocks live under backbone.vision_encoder.backbone.blocks
        try:
            blocks = self.backbone.vision_encoder.backbone.blocks
        except AttributeError as e:
            raise RuntimeError(
                "Could not locate Hiera blocks at vision_encoder.backbone.blocks; "
                "the HuggingFace Sam2Model layout may have changed."
            ) from e

        for block in blocks:
            if self.peft_method in ("lora", "dora", "conv-lora"):
                for tgt in _ATTN_TARGETS:
                    base = _get_dotted(block, tgt)
                    if not isinstance(base, nn.Linear):
                        continue
                    wrapped = wrap_linear(base, self.peft_method, rank=self.rank)
                    replace_submodule(block, tgt, wrapped)
            elif self.peft_method == "adaptformer":
                # Hiera FFN is `mlp.proj_in` -> activation -> `mlp.proj_out`.
                hidden_dim = block.mlp.proj_in.in_features
                block.adaptformer = AdaptFormerFFN(hidden_dim, bottleneck=self.adaptformer_dim)
                _install_adaptformer_hook(block)
            elif self.peft_method == "none":
                pass  # linear-probe baseline: frozen encoder, decoder-only training
            else:
                raise ValueError(f"Unknown peft_method: {self.peft_method!r}")

    def _configure_mask_decoder(self) -> None:
        dec = self.backbone.mask_decoder
        if self.mask_decoder_strategy == "full-ft":
            for p in dec.parameters():
                p.requires_grad_(True)
        elif self.mask_decoder_strategy == "lora-r4":
            for name, mod in list(dec.named_modules()):
                if isinstance(mod, nn.Linear) and any(
                    name.endswith(s) for s in (
                        "q_proj", "k_proj", "v_proj", "out_proj",
                        "lin1", "lin2", "proj_in", "proj_out",
                    )
                ):
                    wrapped = wrap_linear(mod, "lora", rank=4)
                    replace_submodule(dec, name, wrapped)
        else:
            raise ValueError(f"Unknown mask_decoder_strategy: {self.mask_decoder_strategy!r}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run a single forward pass with a whole-chip box prompt.

        Returns
        -------
        logits : (B, H, W) float32, single-class flood logits at the input
                 resolution.
        """
        B, _, H, W = pixel_values.shape
        if (H, W) != (1024, 1024):
            pixel_values = F.interpolate(
                pixel_values, size=(1024, 1024), mode="bilinear", align_corners=False
            )

        box = torch.tensor([[[0.0, 0.0, 1024.0, 1024.0]]], device=pixel_values.device)
        box = box.expand(B, 1, 4)

        out = self.backbone(
            pixel_values=pixel_values,
            input_boxes=box,
            multimask_output=False,
        )
        logits = out.pred_masks[:, 0, 0]  # (B, h, w)
        logits = F.interpolate(logits.unsqueeze(1), size=(H, W),
                               mode="bilinear", align_corners=False).squeeze(1)
        return logits

    def trainable_parameters(self) -> int:
        return count_trainable(self)

    def trainable_percentage(self) -> float:
        total = sum(p.numel() for p in self.parameters())
        return 100.0 * self.trainable_parameters() / total


def _get_dotted(parent: nn.Module, dotted: str) -> nn.Module:
    obj = parent
    for p in dotted.split("."):
        obj = getattr(obj, p)
    return obj


def _install_adaptformer_hook(block: nn.Module) -> None:
    original_mlp_forward = block.mlp.forward
    adapter = block.adaptformer

    def patched(x):
        return original_mlp_forward(x) + adapter(x)

    block.mlp.forward = patched
