"""Quad-panel chip visualization, refactored out of script 08.

Both `08_quality_check.py` and `notebooks/inspect_chips.ipynb` call
`render_chip_panel` with the same arguments, so the rendering code lives
here as a single source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def render_chip_panel(
    chip_id: str,
    pre_path: Path,
    post_path: Path,
    label_path: Path,
    output_path: Optional[Path] = None,
    show_inline: bool = False,
) -> None:
    """Render a 2x2 panel of pre, post, post-minus-pre, and label.

    Parameters
    ----------
    chip_id
        Identifier displayed in the figure title.
    pre_path
        Path to the pre-event SAR GeoTIFF (band 1 = VV in dB).
    post_path
        Path to the post-event SAR GeoTIFF (band 1 = VV in dB).
    label_path
        Path to the binary flood label GeoTIFF.
    output_path
        If provided, save the figure as a PNG at this path.
    show_inline
        If True, leave the figure open so a Jupyter front-end can render
        it inline. If False, close the figure after saving (the default
        used by the headless quality-check script).
    """
    import matplotlib.pyplot as plt
    import rasterio

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
    axes[1, 0].set_title("Post minus Pre (dB), red=darker post")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(label, cmap="Blues", vmin=0, vmax=1)
    axes[1, 1].set_title(f"Flood label ({100*label.mean():.1f}% flooded)")
    axes[1, 1].axis("off")

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=110, bbox_inches="tight")

    if show_inline:
        plt.show()
    else:
        plt.close(fig)
