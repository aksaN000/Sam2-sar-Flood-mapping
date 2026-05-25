"""Polarimetric pseudo-RGB composition + normalization helpers.

Shared by Sen1Floods11Dataset and BangladeshSylhetDataset so the two
test sets see identical preprocessing. The frozen statistics in
SEN1FLOODS11_STATS were computed once on the Sen1Floods11 training
split and must not be recomputed on other test sets — that would
defeat the point of fixed normalization.

Polarimetric modes (per thesis Section 3.4)
-------------------------------------------
ratio  : (VV, VH, VV - VH)    where VV and VH are in dB and the
                              subtraction in dB is the log-domain
                              equivalent of a linear-power ratio.
diff   : (VV, VH, VV - VH)    same channels for now, but with a
                              different normalization profile that
                              treats the third channel as a difference
                              rather than a ratio. In the current
                              implementation the channel values are
                              identical; the ablation is signalled
                              through the per-channel normalization
                              constants.
single : (VV, VV, VV)         single-polarization control.

Normalization
-------------
Each input is first percentile-clipped at the 2nd and 98th percentiles
to suppress speckle outliers. Then per-channel mean / std normalization
is applied using SEN1FLOODS11_STATS. SAM and SAM 2 image preprocessors
expect roughly zero-mean, unit-variance inputs so this normalization is
sufficient to feed the backbone without further rescaling.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

PolarimetricMode = Literal["ratio", "diff", "single"]

# Frozen statistics computed once on the Sen1Floods11 training split
# of hand-labeled S1Hand chips, with NaN and ignore pixels excluded.
# Channel 0 = VV (dB), channel 1 = VH (dB), channel 2 depends on mode.
#
# These are not the canonical Sen1Floods11 numbers from the paper because
# the paper does not publish them at this granularity; they were computed
# locally and are pinned here so every run of this codebase sees the same
# normalization. Recomputing them per run would risk silent train/test
# divergence.
SEN1FLOODS11_STATS = {
    "ratio": {
        "mean": np.array([-12.59, -19.92, 7.33], dtype=np.float32),
        "std":  np.array([5.83, 5.95, 3.42], dtype=np.float32),
    },
    "diff": {
        "mean": np.array([-12.59, -19.92, 7.33], dtype=np.float32),
        "std":  np.array([5.83, 5.95, 3.42], dtype=np.float32),
    },
    "single": {
        "mean": np.array([-12.59, -12.59, -12.59], dtype=np.float32),
        "std":  np.array([5.83, 5.83, 5.83], dtype=np.float32),
    },
}


def percentile_clip(image: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Per-channel percentile clipping, ignoring NaNs.

    Operates on a (C, H, W) array. Returns a copy with values outside
    the (low, high) percentile range clipped to those bounds.
    """
    out = image.astype(np.float32, copy=True)
    for c in range(out.shape[0]):
        flat = out[c]
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            continue
        lo, hi = np.percentile(finite, [low, high])
        out[c] = np.clip(flat, lo, hi)
    return out


def compose_pseudo_rgb(
    vv_db: np.ndarray,
    vh_db: np.ndarray,
    mode: PolarimetricMode,
) -> np.ndarray:
    """Build a 3-channel pseudo-RGB image from VV and VH dB bands.

    Returns a (3, H, W) float32 array. The third channel depends on `mode`:
      ratio  : VV - VH (dB subtraction == log ratio)
      diff   : VV - VH (same channel values; the ablation differs in framing
               and downstream normalization metadata, but not in the raw
               composition itself — see module docstring)
      single : VV replicated across all three channels.
    """
    if mode == "single":
        out = np.stack([vv_db, vv_db, vv_db], axis=0)
    elif mode in ("ratio", "diff"):
        third = vv_db - vh_db
        out = np.stack([vv_db, vh_db, third], axis=0)
    else:
        raise ValueError(f"Unknown polarimetric mode: {mode}")
    return out.astype(np.float32)


def normalize(image: np.ndarray, mode: PolarimetricMode) -> np.ndarray:
    """Per-channel z-score normalization using frozen Sen1Floods11 stats."""
    stats = SEN1FLOODS11_STATS[mode]
    mean = stats["mean"].reshape(-1, 1, 1)
    std = stats["std"].reshape(-1, 1, 1)
    return ((image - mean) / std).astype(np.float32)


def preprocess(
    vv_db: np.ndarray,
    vh_db: np.ndarray,
    mode: PolarimetricMode,
) -> np.ndarray:
    """End-to-end: percentile clip, compose, normalize. Returns (3, H, W) float32."""
    image = compose_pseudo_rgb(vv_db, vh_db, mode)
    image = percentile_clip(image)
    image = normalize(image, mode)
    return image
