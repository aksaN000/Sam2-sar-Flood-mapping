"""
Visual quality assurance for the final chip set.

Why this matters
----------------
Automated pipelines can produce silently wrong results in ways that
summary statistics will not catch. A tile where the radar imagery is
correctly aligned but the flood label is shifted by a few pixels will
look fine in any numerical metric you compute, yet will systematically
poison model training. The only reliable way to catch these errors is
to look at the actual images side by side.

This script picks a random sample of N chips from the final set and
produces a quad-panel figure for each one showing:
  Top-left:    pre-event Sentinel-1 VV (dB)
  Top-right:   post-event Sentinel-1 VV (dB)
  Bottom-left: post-event - pre-event difference (highlights new water)
  Bottom-right: ground-truth flood label

You then visually scan through the figures and flag any chip where the
flood label does not look correctly aligned to the obvious dark patches
in the post-event imagery. Flagged chips can be manually removed from
the final test set.

Output: data/final/qa_panels/chip_*.png
"""

import csv
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio

PROJECT_ROOT = Path(__file__).parent.parent
FINAL_DIR = PROJECT_ROOT / "data" / "final"
QA_DIR = FINAL_DIR / "qa_panels"
INDEX_PATH = FINAL_DIR / "chip_index.csv"

SAMPLE_SIZE = 20
RANDOM_SEED = 42


def render_chip_panel(chip_id, pre_path, post_path, label_path, output_path):
    """Generate a 2x2 panel of pre, post, difference, and label."""
    with rasterio.open(pre_path) as src:
        pre_vv = src.read(1)
    with rasterio.open(post_path) as src:
        post_vv = src.read(1)
    with rasterio.open(label_path) as src:
        label = src.read(1)

    diff = post_vv - pre_vv

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"QA panel: {chip_id}", fontsize=12)

    axes[0, 0].imshow(pre_vv, cmap="gray", vmin=-25, vmax=0)
    axes[0, 0].set_title("Pre-event VV (dB)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(post_vv, cmap="gray", vmin=-25, vmax=0)
    axes[0, 1].set_title("Post-event VV (dB)")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(diff, cmap="RdBu_r", vmin=-10, vmax=10)
    axes[1, 0].set_title("Post - Pre (dB), red=darker post")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(label, cmap="Blues", vmin=0, vmax=1)
    axes[1, 1].set_title(f"Flood label ({100*label.mean():.1f}% flooded)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not INDEX_PATH.exists():
        print(f"Cannot find {INDEX_PATH}. Run script 07 first.")
        return

    with open(INDEX_PATH) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("Chip index is empty. Run scripts 02 through 07 to populate it.")
        return

    print(f"Found {len(rows)} chips in index")
    random.seed(RANDOM_SEED)
    sample_size = min(SAMPLE_SIZE, len(rows))
    sample = random.sample(rows, sample_size)
    print(f"Generating QA panels for {sample_size} randomly sampled chips")

    for row in sample:
        pre_path = PROJECT_ROOT / row["pre_path"]
        post_path = PROJECT_ROOT / row["post_path"]
        label_path = PROJECT_ROOT / row["label_path"]
        out_path = QA_DIR / f"{row['chip_id']}.png"
        render_chip_panel(row["chip_id"], pre_path, post_path, label_path, out_path)
        print(f"  Wrote {out_path.name} (flood fraction {row['flood_pixel_fraction']})")

    print(f"\nDone. Open the PNG files in {QA_DIR}")
    print("and scan for chips where the flood label does not match the dark")
    print("regions in the post-event imagery. Flag any obvious misalignments.")


if __name__ == "__main__":
    main()
