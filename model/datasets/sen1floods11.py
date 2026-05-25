"""PyTorch Dataset for Sen1Floods11 SAR flood chips.

Loads hand-labeled (or weakly-labeled) chips from the Sen1Floods11 v1.1
directory layout, applies polarimetric pseudo-RGB composition, and
returns a dict suitable for the SAM and SAM 2 image preprocessors.

Sen1Floods11 layout assumed:
    <root>/v1.1/data/flood_events/HandLabeled/S1Hand/<name>_S1Hand.tif
    <root>/v1.1/data/flood_events/HandLabeled/LabelHand/<name>_LabelHand.tif
    <root>/v1.1/splits/flood_handlabeled/flood_<split>_data.csv

Each split CSV has rows: `<s1_filename>,<label_filename>` (no header).

S1Hand chips are 2-band float32 GeoTIFFs in dB (per the original paper).
LabelHand chips are 1-band int8 GeoTIFFs with values:
    -1  ignore / cloud / nodata    -> mapped to 255 in the returned uint8 mask
     0  non-water
     1  water (flood)

Returned dict keys
------------------
image    torch.float32, shape (3, H, W), normalized to Sen1Floods11 stats
label    torch.uint8,   shape (H, W),    0=non-water, 1=water, 255=ignore
chip_id  str,                              filename stem (e.g. "Ghana_103272")
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from .polarimetric import PolarimetricMode, preprocess

Sen1FloodsSplit = Literal["train", "valid", "test", "bolivia", "pakistan", "pakistan2022"]


class Sen1Floods11Dataset(Dataset):
    """Sen1Floods11 SAR flood chips with polarimetric pseudo-RGB composition.

    Parameters
    ----------
    root
        Path to the Sen1Floods11 root (the directory that contains the
        `v1.1/` subdirectory; pass the parent, not v1.1 itself).
    split
        One of "train", "valid", "test", "bolivia", "pakistan". The first
        four map directly to the official Sen1Floods11 split CSVs under
        root/v1.1/splits/flood_handlabeled/. The "pakistan" split is a
        regional subset carved from the union of test+valid Pakistan_*
        chips (12 hand-labeled chips from the 2010 Pakistan flood event,
        regionally analogous to the Bangladesh Indo-Gangetic deltaic
        monsoon focus of this thesis). Pakistan chips are present in
        the training set as well, so the pakistan split is reported as
        an additional regional in-distribution slice, not a strict OOD test.
    polarimetric_mode
        Channel composition for the pseudo-RGB input. See polarimetric.py.
    """

    def __init__(
        self,
        root: str | Path,
        split: Sen1FloodsSplit = "train",
        polarimetric_mode: PolarimetricMode = "ratio",
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.polarimetric_mode = polarimetric_mode
        self.pairs: list[tuple[str, str]] = self._load_split_index()
        self.s1_dir = self.root / "v1.1" / "data" / "flood_events" / "HandLabeled" / "S1Hand"
        self.label_dir = self.root / "v1.1" / "data" / "flood_events" / "HandLabeled" / "LabelHand"

    def _load_split_index(self) -> list[tuple[str, str]]:
        splits_dir = self.root / "v1.1" / "splits" / "flood_handlabeled"
        if self.split == "pakistan":
            # Filter test + valid rows whose chip name starts with "Pakistan_".
            rows: list[tuple[str, str]] = []
            for src in ("test", "valid"):
                src_path = splits_dir / f"flood_{src}_data.csv"
                if not src_path.exists():
                    raise FileNotFoundError(f"Split CSV not found: {src_path}")
                with open(src_path, newline="") as f:
                    rows.extend(
                        tuple(r) for r in csv.reader(f)
                        if len(r) == 2 and r[0].startswith("Pakistan_")
                    )
            return rows

        if self.split == "pakistan2022":
            # Custom test set we build from TU Wien Pakistan-2022 masks + S1 RTC.
            # The root should point at the pakistan2022 chip tree, not Sen1Floods11.
            split_path = splits_dir / "flood_pakistan2022_data.csv"
            if not split_path.exists():
                raise FileNotFoundError(
                    f"Pakistan-2022 split CSV not found: {split_path}\n"
                    f"Did you run data_pipelines/pakistan_2022/acquire.py first, and "
                    f"are you passing the right --root (the chips root, not the "
                    f"Sen1Floods11 root)?"
                )
            with open(split_path, newline="") as f:
                rows = [tuple(r) for r in csv.reader(f) if len(r) == 2]
            return rows

        split_path = splits_dir / f"flood_{self.split}_data.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {split_path}")
        with open(split_path, newline="") as f:
            rows = [tuple(r) for r in csv.reader(f) if len(r) == 2]
        return rows

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        s1_name, label_name = self.pairs[idx]
        s1_path = self.s1_dir / s1_name
        label_path = self.label_dir / label_name

        with rasterio.open(s1_path) as src:
            bands = src.read().astype(np.float32)  # (2, H, W) in dB
        vv_db, vh_db = bands[0], bands[1]

        with rasterio.open(label_path) as src:
            raw_label = src.read(1).astype(np.int16)  # values -1, 0, 1

        # Map -1 (ignore) -> 255 for downstream loss functions.
        label = np.where(raw_label < 0, 255, raw_label).astype(np.uint8)

        # On the pakistan2022 split only, also mark pixels with -100 dB sentinel
        # values (produced by acquire.py's linear_to_db clamp at 1e-10 when the
        # MS Planetary Computer S1 RTC product has no acquisition coverage) as
        # ignore=255 in the label. Apply per-pixel: any pixel where either VV
        # or VH is below -50 dB gets masked from loss/metric. This threshold is
        # tight enough to exclude no-coverage pixels but loose enough to keep
        # all legitimate Sentinel-1 dB values. Sen1Floods11 chips occasionally
        # have legitimate values down to roughly -90 dB at swath edges that the
        # trained models have already been exposed to, so the threshold is NOT
        # applied to Sen1Floods11 splits to preserve checkpoint compatibility.
        if self.split == "pakistan2022":
            nodata_pix = (~np.isfinite(vv_db)) | (~np.isfinite(vh_db)) | (vv_db < -50) | (vh_db < -50)
            if nodata_pix.any():
                label = np.where(nodata_pix, 255, label).astype(np.uint8)

        # Replace NaN/inf in SAR with the channel mean so percentile-clip
        # behaves sensibly. Sen1Floods11 chips occasionally have NaN at
        # swath edges. On the pakistan2022 split also replace the -100 dB
        # sentinel so it doesn't dominate the percentile clip.
        replace_thresh = -50 if self.split == "pakistan2022" else -np.inf
        for arr in (vv_db, vh_db):
            mask = (~np.isfinite(arr)) | (arr < replace_thresh)
            if mask.any():
                valid = arr[~mask]
                finite_mean = float(valid.mean()) if valid.size > 0 else 0.0
                arr[mask] = finite_mean

        image = preprocess(vv_db, vh_db, self.polarimetric_mode)
        chip_id = s1_name.replace("_S1Hand.tif", "")

        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),
            "chip_id": chip_id,
        }
