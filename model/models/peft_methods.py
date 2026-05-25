"""The four parameter-efficient fine-tuning methods compared in the thesis.

Each is implemented as a thin wrapper around an existing nn.Linear, so
we can swap one for another without touching the surrounding architecture.
The base linear's weights are frozen; only the small added parameters
are trainable. This keeps the parameter count below 5% of the backbone
for typical ranks (4-16).

  LoRAlinear        : standard low-rank update    BA on top of W
                      (Hu et al. ICLR 2022)
  DoRALinear        : magnitude/direction split   m * (W + BA) / ||W + BA||
                      (Liu et al. ICML 2024)
  ConvLoRALinear    : LoRA + parallel depthwise conv on the spatial layout
                      of the low-rank intermediate; expects the inputs to
                      a transformer block to be reshapeable to (B, C, H, W)
                      (Zhong et al. ICLR 2024)
  AdaptFormerFFN    : parallel bottleneck adapter on a feed-forward block
                      (Chen et al. NeurIPS 2022) -- this one wraps two
                      linears (the up and down projections of the FFN)
                      rather than a single linear.

All four expose `.replace_for(parent, attr_name, ...)` to perform in-place
substitution of the original module.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

PEFTMethod = Literal["lora", "dora", "conv-lora", "adaptformer", "none"]


class LoRALinear(nn.Module):
    """y = W x + alpha/r * B A x, with W frozen and (B, A) trained.

    Following Hu et al. ICLR 2022:
      A is initialized with Kaiming-uniform (small random)
      B is initialized to zero so the initial output equals the base linear
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int | None = None) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


class DoRALinear(nn.Module):
    """DoRA: decompose W into magnitude m (per output channel) and direction V.

    The direction is updated via a LoRA-style low-rank delta; the magnitude
    is trained as a vector. The forward is:
        V_eff = W + BA
        out   = (V_eff / ||V_eff||_col) * m

    where ||.||_col is the per-output-channel L2 norm of the columns.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int | None = None) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        # Initialize magnitude from the base weight's column norms (so the
        # initial forward matches the base linear exactly).
        with torch.no_grad():
            mag = base.weight.norm(p=2, dim=1, keepdim=False)  # (out,)
        self.magnitude = nn.Parameter(mag.clone())
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.lora_B @ self.lora_A * self.scaling  # (out, in)
        V_eff = self.base.weight + delta                  # (out, in)
        col_norm = V_eff.norm(p=2, dim=1, keepdim=True) + 1e-8
        W_eff = V_eff / col_norm * self.magnitude.unsqueeze(1)
        return F.linear(x, W_eff, self.base.bias)


class ConvLoRALinear(nn.Module):
    """LoRA augmented with a parallel depthwise conv on the low-rank intermediate.

    Per Zhong et al. ICLR 2024, the conv injects locality bias into a plain
    ViT encoder. The intermediate Ax has shape (B, N, r) for a token
    sequence; we reshape to (B, r, H, W), apply a 3x3 depthwise conv with
    `groups=r`, flatten back to (B, N, r), then go through B as in LoRA.

    The conv path's parameter count is r*3*3 = 9r, which is negligible
    compared to the LoRA path (r*(in+out)).

    The spatial grid (H, W) is taken from the `spatial_hw` attribute set
    by the parent module before forward; if unset, ConvLoRALinear falls
    back to plain LoRA behavior (the conv is skipped).
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int | None = None) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.conv = nn.Conv2d(rank, rank, kernel_size=3, padding=1, groups=rank, bias=False)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.conv.weight)
        # Set externally by the parent encoder before each forward, when the
        # tokens correspond to a regular HxW grid.
        self.spatial_hw: tuple[int, int] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward auto-detects the spatial layout of x.

        Handles two common cases:
          (B, N, C) where sqrt(N) is integer -> reshape to (B, C, sqrt(N), sqrt(N))
          (B, H, W, C)                       -> permute to (B, C, H, W) directly
        Otherwise the conv path is skipped and ConvLoRA degrades to plain LoRA.
        """
        base_out = self.base(x)
        ax = F.linear(x, self.lora_A)  # same shape as x but last dim = r

        applied_conv = False
        if ax.ndim == 4:
            # (B, H, W, r) -> (B, r, H, W) -> conv -> back
            B, H, W, r = ax.shape
            conv_in = ax.permute(0, 3, 1, 2).contiguous()
            ax = ax + self.conv(conv_in).permute(0, 2, 3, 1)
            applied_conv = True
        elif ax.ndim == 3:
            B, N, r = ax.shape
            side = int(round(N ** 0.5))
            if side * side == N:
                conv_in = ax.transpose(1, 2).reshape(B, r, side, side)
                ax = ax + self.conv(conv_in).reshape(B, r, N).transpose(1, 2)
                applied_conv = True
        # else: leave ax unchanged; conv path is a no-op for other shapes

        lora_out = F.linear(ax, self.lora_B) * self.scaling
        return base_out + lora_out


class AdaptFormerFFN(nn.Module):
    """Parallel bottleneck adapter on a feed-forward block.

    Wraps the FFN's up-projection (lin1) and down-projection (lin2) and
    adds a parallel path:
        adapter(x) = down(GELU(up(x))) * s
        ffn_out    = original_FFN(x) + adapter(x)

    The original FFN is frozen; only the adapter and scalar `s` are trained.

    Use replace_ffn(parent_block, bottleneck=...) to install on a block
    that exposes `mlp.lin1` and `mlp.lin2` and an `mlp.act` activation.
    """

    def __init__(self, hidden_dim: int, bottleneck: int = 64) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_dim)
        self.act = nn.GELU()
        # Following Chen et al. NeurIPS 2022 reference implementation:
        # scale is initialized to 1.0 (NOT zero), and up.weight is zero.
        # That gives adapter_out=0 at init (identity preserved) but a nonzero
        # gradient w.r.t. up.weight so the adapter can start learning. If
        # scale were also zero, both gradient paths would be zero and the
        # adapter would be permanently frozen at zero (the original bug in
        # this codebase that caused AdaptFormer to underperform).
        self.scale = nn.Parameter(torch.ones(1))
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x))) * self.scale


def wrap_linear(
    base: nn.Linear,
    method: PEFTMethod,
    rank: int = 8,
    alpha: int | None = None,
) -> nn.Module:
    """Return a LoRA/DoRA/ConvLoRA wrapper around an nn.Linear.

    AdaptFormer is not a per-linear wrapper -- it is installed at the FFN
    block level via AdaptFormerFFN; use install_adaptformer() instead.
    """
    if method == "lora":
        return LoRALinear(base, rank=rank, alpha=alpha)
    if method == "dora":
        return DoRALinear(base, rank=rank, alpha=alpha)
    if method == "conv-lora":
        return ConvLoRALinear(base, rank=rank, alpha=alpha)
    raise ValueError(f"wrap_linear does not support method={method!r}; "
                     f"use install_adaptformer for AdaptFormer.")


def replace_submodule(parent: nn.Module, dotted: str, new: nn.Module) -> None:
    """Replace `parent.<a>.<b>.<c>` with `new` given the dotted attribute name."""
    parts = dotted.split(".")
    obj = parent
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], new)


def count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
