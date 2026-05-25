"""Confidence-aware predictors over a trained adapter.

Two variants matching the thesis methodology (Section 3.5):

  MCDropoutPredictor    Monte Carlo dropout, N forward passes per chip,
                        per-pixel std as the uncertainty estimate.
  DeepEnsemblePredictor Aggregation over K independently trained
                        adapter checkpoints (different random seeds).

Both emit a tuple `(mean_probs, std_probs)` where `mean_probs` is the
final flood probability map and `std_probs` is the per-pixel epistemic
uncertainty. The two are fed into expected_calibration_error() for the
calibration plots in the thesis Results chapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn


class MCDropoutPredictor(nn.Module):
    """Monte Carlo dropout over a single trained adapter.

    At inference, every nn.Dropout submodule is forced into train mode so
    its randomness stays active, while batch-norm / layer-norm / the rest
    of the model stays in eval mode. `n_passes` forward passes are run on
    each batch and the per-pixel mean and standard deviation of the sigmoid
    probabilities are returned.
    """

    def __init__(self, model: nn.Module, n_passes: int = 20) -> None:
        super().__init__()
        self.model = model
        self.n_passes = n_passes

    def _set_dropout_train(self) -> None:
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        self._set_dropout_train()
        probs = []
        for _ in range(self.n_passes):
            logits = self.model(pixel_values)
            probs.append(torch.sigmoid(logits))
        stack = torch.stack(probs, dim=0)  # (N, B, H, W)
        return stack.mean(dim=0), stack.std(dim=0)


class DeepEnsemblePredictor(nn.Module):
    """Deep ensemble over K trained adapter checkpoints.

    `model_factory()` should return a fresh adapter instance ready to
    receive a state_dict. Each checkpoint is loaded into its own member,
    set to eval mode, and predictions are aggregated as mean and std of
    the sigmoid probabilities.
    """

    def __init__(
        self,
        checkpoints: Iterable[str | Path],
        model_factory: Callable[[], nn.Module],
    ) -> None:
        super().__init__()
        members = []
        for ckpt_path in checkpoints:
            model = model_factory()
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = state.get("state_dict", state) if isinstance(state, dict) else state
            # Strip a possible "model." prefix added by LightningModule.
            sd = {k.removeprefix("model."): v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
            model.eval()
            members.append(model)
        self.members = nn.ModuleList(members)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = []
        for m in self.members:
            logits = m(pixel_values)
            probs.append(torch.sigmoid(logits))
        stack = torch.stack(probs, dim=0)  # (K, B, H, W)
        return stack.mean(dim=0), stack.std(dim=0)


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
    ignore_index: int = 255,
) -> float:
    """Compute ECE on a binary segmentation task.

    Partitions the predicted probability range [0, 1] into `n_bins` equal-width
    bins, computes the absolute difference between average predicted
    probability and empirical accuracy within each bin, and returns the
    sample-weighted mean (Guo et al. ICML 2017).

    Parameters
    ----------
    probs   (B, H, W) sigmoid probabilities for the flood class.
    labels  (B, H, W) integer labels with values 0, 1, or `ignore_index`.
    n_bins  number of probability bins.
    """
    probs = probs.detach().flatten()
    labels = labels.detach().flatten()
    valid = labels != ignore_index
    probs = probs[valid]
    labels = labels[valid].float()
    if probs.numel() == 0:
        return 0.0

    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece = torch.tensor(0.0, device=probs.device)
    total = probs.numel()
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if mask.any():
            avg_conf = probs[mask].mean()
            avg_acc = labels[mask].mean()
            weight = mask.float().sum() / total
            ece = ece + (avg_conf - avg_acc).abs() * weight
    return ece.item()
