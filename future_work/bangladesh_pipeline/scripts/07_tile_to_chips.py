"""
Cut the filtered Sentinel-1 imagery and the label raster into 512 by 512
pixel chips matching the Sen1Floods11 convention.

Why 512x512
-----------
Sen1Floods11 uses 512x512 chips, so adopting the same tile size means
your test set can be loaded by the exact same PyTorch Dataset class as
the training set. It also fits comfortably into a single GPU forward
pass for SAM ViT-B and SAM 2 Hiera-Base-Plus on a 12 GB consumer card.

What this script does
---------------------
1. Reads the pre-event imagery, post-event imagery, and label raster.
2. Walks a regular grid of (row_offset, col_offset) starting positions
   across the rasters.
3. At each position, extracts a 512x512 window from all three rasters.
4. Discards windows that are mostly nodata or that fall partially off
   the edge of the image.
5. Writes each window to disk as three GeoTIFFs in a Sen1Floods11-style
   directory layout under data/final/.

Output directory layout
-----------------------
data/final/
    pre_event/
        sylhet_2022_chip_000001.tif
        sylhet_2022_chip_000002.tif
        ...
    post_event/
        sylhet_2022_chip_000001.tif
        ...
    label/
        sylhet_2022_chip_000001.tif
        ...
    chip_index.csv  (one row per chip with file paths and statistics)
"""

import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FINAL_DIR = PROJECT_ROOT / "data" / "final"

PRE_INPUT = PROCESSED_DIR / "sylhet_pre_event_filtered_db.tif"
POST_INPUT = PROCESSED_DIR / "sylhet_post_event_filtered_db.tif"
LABEL_INPUT = PROCESSED_DIR / "flood_label.tif"

CHIP_SIZE = 512
STRIDE = 512  # set < CHIP_SIZE for overlapping chips, = CHIP_SIZE for non-overlapping
MIN_VALID_FRACTION = 0.95  # discard chips that have more than 5% nodata


def write_chip(src_dataset, window, out_path):
    """Write a windowed read from src_dataset to out_path as a small GeoTIFF."""
    data = src_dataset.read(window=window)
    profile = src_dataset.profile.copy()
    profile.update({
        "height": window.height,
        "width": window.width,
        "transform": rasterio.windows.transform(window, src_dataset.transform),
        "compress": "lzw",
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)


def is_valid_chip(pre_data, post_data, label_data, min_valid_fraction):
    """Check that the chip is mostly within the valid imagery footprint."""
    total = pre_data.size + post_data.size
    valid_pre = np.sum(np.isfinite(pre_data) & (pre_data != 0))
    valid_post = np.sum(np.isfinite(post_data) & (post_data != 0))
    return (valid_pre + valid_post) / total >= min_valid_fraction


def main() -> None:
    if not all(p.exists() for p in [PRE_INPUT, POST_INPUT, LABEL_INPUT]):
        print("Cannot find one or more input rasters. Expected:")
        print(f"  {PRE_INPUT}")
        print(f"  {POST_INPUT}")
        print(f"  {LABEL_INPUT}")
        print("Run scripts 02 through 06 first.")
        return

    pre_src = rasterio.open(PRE_INPUT)
    post_src = rasterio.open(POST_INPUT)
    label_src = rasterio.open(LABEL_INPUT)

    h, w = pre_src.height, pre_src.width
    print(f"Source raster: {w} x {h} pixels")

    # Build the list of starting positions for chips on a regular grid
    starts = []
    for row in range(0, h - CHIP_SIZE + 1, STRIDE):
        for col in range(0, w - CHIP_SIZE + 1, STRIDE):
            starts.append((row, col))
    print(f"Generated {len(starts)} candidate chip positions")

    # Open the chip index CSV for writing as we go
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    index_path = FINAL_DIR / "chip_index.csv"
    written = 0
    with open(index_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chip_id", "row_off", "col_off", "pre_path", "post_path", "label_path", "flood_pixel_fraction"])

        for chip_idx, (row_off, col_off) in enumerate(starts):
            window = Window(col_off, row_off, CHIP_SIZE, CHIP_SIZE)
            pre_data = pre_src.read(window=window)
            post_data = post_src.read(window=window)
            label_data = label_src.read(1, window=window)

            if not is_valid_chip(pre_data, post_data, label_data, MIN_VALID_FRACTION):
                continue

            chip_id = f"sylhet_2022_chip_{chip_idx:06d}"
            pre_path = FINAL_DIR / "pre_event" / f"{chip_id}.tif"
            post_path = FINAL_DIR / "post_event" / f"{chip_id}.tif"
            label_path = FINAL_DIR / "label" / f"{chip_id}.tif"

            write_chip(pre_src, window, pre_path)
            write_chip(post_src, window, post_path)
            write_chip(label_src, window, label_path)

            flood_fraction = float(label_data.mean()) if label_data.size > 0 else 0.0
            writer.writerow([chip_id, row_off, col_off,
                             pre_path.relative_to(PROJECT_ROOT),
                             post_path.relative_to(PROJECT_ROOT),
                             label_path.relative_to(PROJECT_ROOT),
                             f"{flood_fraction:.4f}"])
            written += 1

    pre_src.close()
    post_src.close()
    label_src.close()

    print(f"Wrote {written} chips to {FINAL_DIR}")
    print(f"Chip index saved to {index_path}")


if __name__ == "__main__":
    main()
