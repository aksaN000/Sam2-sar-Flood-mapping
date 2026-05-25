"""PyTorch Dataset for the custom Bangladesh-2022-Sylhet SAR flood test set.

Loads chips produced by scripts/07_tile_to_chips.py from the
data/final/ directory layout. The chip layout matches Sen1Floods11
(pre_event/, post_event/, label/, chip_index.csv), so the polarimetric
composition and normalization logic is shared with the Sen1Floods11
dataset class.

This dataset is evaluation-only: it has no training split.

Returned dict keys
------------------
image    torch.float32, shape (3, H, W), normalized to Sen1Floods11 stats
label    torch.uint8,   shape (H, W),    binary flood mask
chip_id  str,                              chip identifier from chip_index.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

PolarimetricMode = Literal["ratio", "diff", "single"]


class BangladeshSylhetDataset(Dataset):
    """Bangladesh-2022-Sylhet OOD test set chips.

    Parameters
    ----------
    root
        Path to data/final/ as produced by scripts/07_tile_to_chips.py.
    polarimetric_mode
        Channel composition for the pseudo-RGB input. Must match the
        composition used during training.
    use_post_event
        If True, return post-event imagery (the standard evaluation
        configuration). If False, return pre-event imagery (used for
        sanity-checking that the model does not predict flood on dry
        baseline imagery).
    """

    def __init__(
        self,
        root: str | Path,
        polarimetric_mode: PolarimetricMode = "ratio",
        use_post_event: bool = True,
    ) -> None:
        self.root = Path(root)
        self.polarimetric_mode = polarimetric_mode
        self.use_post_event = use_post_event
        self.index: list[dict] = self._read_chip_index()

    def _read_chip_index(self) -> list[dict]:
        index_path = self.root / "chip_index.csv"
        with open(index_path) as f:
            return list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        row = self.index[idx]
        sar_key = "post_path" if self.use_post_event else "pre_path"
        vv_db, vh_db = self._load_sar_chip(row[sar_key])
        label = self._load_label_chip(row["label_path"])
        image = self._compose_pseudo_rgb(vv_db, vh_db, self.polarimetric_mode)
        image = self._normalize(image)
        return {
            "image": torch.from_numpy(image).float(),
            "label": torch.from_numpy(label).to(torch.uint8),
            "chip_id": row["chip_id"],
        }

    def _load_sar_chip(self, relpath: str) -> tuple[np.ndarray, np.ndarray]:
        # TODO: open the GeoTIFF and return VV, VH bands as float32 dB.
        #       The chip_index.csv stores paths relative to the project root,
        #       so resolve against self.root.parent.parent.
        raise NotImplementedError

    def _load_label_chip(self, relpath: str) -> np.ndarray:
        # TODO: open the binary label GeoTIFF and return a uint8 mask.
        raise NotImplementedError

    @staticmethod
    def _compose_pseudo_rgb(
        vv_db: np.ndarray,
        vh_db: np.ndarray,
        mode: PolarimetricMode,
    ) -> np.ndarray:
        # TODO: shared with Sen1Floods11Dataset; refactor into a common helper
        raise NotImplementedError

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        # TODO: apply the same percentile clipping and per-channel
        #       mean/std normalization as the Sen1Floods11 training pipeline.
        #       Statistics must be loaded from the same frozen JSON.
        raise NotImplementedError
