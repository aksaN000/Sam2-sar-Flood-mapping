"""SAM ViT-B with parameter-efficient fine-tuning adapters.

Loads HuggingFace's facebook/sam-vit-base, freezes the image and prompt
encoders, injects one of four PEFT mechanisms into the image encoder,
and adapts the mask decoder via either full fine-tuning at low learning
rate or LoRA rank-4 adapters.

The forward path consumes a normalized pseudo-RGB tensor (the output of
Sen1Floods11Dataset / BangladeshSylhetDataset) plus an implicit
whole-image box prompt, and returns a single-channel logit map of the
same spatial size as the input. We use SAM in its prompt-free mode by
feeding a dummy box covering the whole chip; this matches the protocol
used by CWSAM and by all SAR flood adaptations of SAM in the literature.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SamModel

from .peft_methods import (
    AdaptFormerFFN,
    ConvLoRALinear,
    PEFTMethod,
    count_trainable,
    replace_submodule,
    wrap_linear,
)

MaskDecoderStrategy = Literal["full-ft", "lora-r4"]

# Target linear modules inside each ViT block to which we apply LoRA/DoRA/Conv-LoRA.
# We attach to the fused QKV and to the attention output projection, following
# the thesis methodology (sect. 3.4).
_ATTN_TARGETS = ["attn.qkv", "attn.proj"]


class SAMAdapter(nn.Module):
    """SAM ViT-B with a chosen PEFT mechanism on the image encoder."""

    def __init__(
        self,
        pretrained_id: str = "facebook/sam-vit-base",
        peft_method: PEFTMethod = "lora",
        rank: int = 8,
        adaptformer_dim: int = 64,
        mask_decoder_strategy: MaskDecoderStrategy = "full-ft",
    ) -> None:
        super().__init__()
        self.peft_method = peft_method
        self.rank = rank
        self.adaptformer_dim = adaptformer_dim
        self.mask_decoder_strategy = mask_decoder_strategy

        self.backbone = SamModel.from_pretrained(pretrained_id)
        self._freeze_all()
        self._inject_peft_into_encoder()
        self._configure_mask_decoder()
        self._wire_spatial_hw_for_convlora()

    # ----- setup helpers -----

    def _freeze_all(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def _inject_peft_into_encoder(self) -> None:
        enc = self.backbone.vision_encoder
        for block in enc.layers:
            if self.peft_method in ("lora", "dora", "conv-lora"):
                for tgt in _ATTN_TARGETS:
                    base = _get_dotted(block, tgt)
                    wrapped = wrap_linear(base, self.peft_method, rank=self.rank)
                    replace_submodule(block, tgt, wrapped)
            elif self.peft_method == "adaptformer":
                hidden_dim = block.mlp.lin1.in_features  # 768 for ViT-B
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
                    name.endswith(s) for s in ("q_proj", "k_proj", "v_proj", "out_proj", "lin1", "lin2")
                ):
                    wrapped = wrap_linear(mod, "lora", rank=4)
                    replace_submodule(dec, name, wrapped)
        else:
            raise ValueError(f"Unknown mask_decoder_strategy: {self.mask_decoder_strategy!r}")

    def _wire_spatial_hw_for_convlora(self) -> None:
        """Push the encoder's spatial grid into every ConvLoRALinear before forward."""
        if self.peft_method != "conv-lora":
            return
        self._convlora_modules = [m for m in self.modules() if isinstance(m, ConvLoRALinear)]

        def set_grid(_module, _inputs):
            for m in self._convlora_modules:
                m.spatial_hw = (64, 64)  # SAM ViT-B sees a 64x64 token grid at 1024x1024 input

        self.backbone.vision_encoder.register_forward_pre_hook(set_grid)

    # ----- inference / training -----

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run a single forward pass with a whole-chip box prompt.

        Parameters
        ----------
        pixel_values : (B, 3, H, W) float32, already normalized.

        Returns
        -------
        logits : (B, H, W) float32, single-class flood logits at the input
                 resolution (upsampled from SAM's native low-res mask).
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
        logits = out.pred_masks[:, 0, 0]  # (B, 256, 256)
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
    """Patch the block's mlp.forward so the adapter output is added in parallel."""
    original_mlp_forward = block.mlp.forward
    adapter = block.adaptformer

    def patched(x):
        return original_mlp_forward(x) + adapter(x)

    block.mlp.forward = patched
