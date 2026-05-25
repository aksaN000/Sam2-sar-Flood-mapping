"""
Apply the Refined Lee speckle filter to the Sentinel-1 GRD imagery and
convert from linear power to decibels (dB).

Why we need this
----------------
Synthetic Aperture Radar imagery naturally contains speckle noise, which
is a salt-and-pepper graininess intrinsic to how radar physics works. The
signal returned by the satellite is the coherent sum of many tiny
reflections within each resolution cell, and those reflections interfere
constructively or destructively depending on the exact arrangement of
scatterers. Even a perfectly uniform field will therefore look grainy.

The Refined Lee filter, developed by Jong-Sen Lee in the 1980s, smooths
this graininess while preserving sharp edges like the boundary between
flooded and dry land. It does this by computing local statistics in a
sliding window and adapting the smoothing strength based on whether the
local statistics suggest a homogeneous region (smooth strongly) or a
boundary (preserve detail).

We then convert the filtered linear-power values to decibels because
flood mapping models are typically trained on dB values, and Sen1Floods11
itself distributes its imagery in dB.

Inputs:  data/raw/sylhet_pre_event.tif   (linear power, VV+VH)
         data/raw/sylhet_post_event.tif  (linear power, VV+VH)
Outputs: data/processed/sylhet_pre_event_filtered_db.tif  (dB, VV+VH)
         data/processed/sylhet_post_event_filtered_db.tif (dB, VV+VH)
"""

from pathlib import Path

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PRE_INPUT = RAW_DIR / "sylhet_pre_event.tif"
POST_INPUT = RAW_DIR / "sylhet_post_event.tif"
PRE_OUTPUT = PROCESSED_DIR / "sylhet_pre_event_filtered_db.tif"
POST_OUTPUT = PROCESSED_DIR / "sylhet_post_event_filtered_db.tif"

WINDOW_SIZE = 7  # pixels, must be odd
DAMPING = 1.0    # filter damping factor; higher means more smoothing


def refined_lee(image: np.ndarray, window_size: int = 7, damping: float = 1.0) -> np.ndarray:
    """
    A pure-NumPy implementation of the Refined Lee adaptive speckle filter.

    The algorithm computes local mean and variance in a sliding window, then
    blends the local mean with the original pixel value using a weight that
    depends on the local coefficient of variation. In smooth regions the
    weight goes to 1, replacing the noisy pixel with the local mean. Near
    edges the weight goes to 0, preserving the original pixel value.
    """
    from scipy.ndimage import uniform_filter

    # Compute local mean and local mean-of-squares with a uniform window
    mean_local = uniform_filter(image.astype(np.float64), size=window_size)
    mean_sq_local = uniform_filter(image.astype(np.float64) ** 2, size=window_size)
    var_local = mean_sq_local - mean_local ** 2

    # Estimate the noise variance assuming the noise is multiplicative
    # speckle with a known 'equivalent number of looks' (ENL). For Sentinel-1
    # GRD products the ENL is typically about 4 or 5; we use 4 as a robust
    # default that works well across the dynamic range of Bangladesh imagery.
    enl = 4.0
    noise_var = (mean_local ** 2) / enl

    # The weight balances local smoothing against edge preservation
    weight = (var_local - noise_var * damping) / (var_local + 1e-10)
    weight = np.clip(weight, 0.0, 1.0)

    # Filtered output is a weighted blend of the original and the local mean
    filtered = mean_local + weight * (image.astype(np.float64) - mean_local)
    return filtered.astype(image.dtype)


def linear_to_db(image: np.ndarray) -> np.ndarray:
    """Convert linear-power values to decibels with a small floor to avoid log(0)."""
    floor = 1e-10
    safe = np.maximum(image, floor)
    return 10.0 * np.log10(safe).astype(np.float32)


def process_image(input_path: Path, output_path: Path, label: str) -> None:
    print(f"\nProcessing {label}: {input_path.name}")
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        bands = src.read()  # shape (n_bands, height, width), in linear power
        print(f"  Loaded {bands.shape[0]} band(s), {bands.shape[1]}x{bands.shape[2]} pixels")

        out_bands = np.zeros_like(bands, dtype=np.float32)
        for i in range(bands.shape[0]):
            band = bands[i]
            filtered = refined_lee(band, window_size=WINDOW_SIZE, damping=DAMPING)
            out_bands[i] = linear_to_db(filtered)
            band_name = src.descriptions[i] if i < len(src.descriptions) else f"band_{i+1}"
            print(f"  Band {band_name}: range {out_bands[i].min():.1f} to {out_bands[i].max():.1f} dB")

    # Update profile for float32 output
    profile.update(dtype=rasterio.float32, compress="lzw")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out_bands)
    print(f"  Wrote filtered dB output to {output_path}")


def main() -> None:
    if not PRE_INPUT.exists() or not POST_INPUT.exists():
        print(f"Cannot find input GeoTIFFs. Expected:")
        print(f"  {PRE_INPUT}")
        print(f"  {POST_INPUT}")
        print(f"Run script 02 and download the exports from Google Drive first.")
        return

    process_image(PRE_INPUT, PRE_OUTPUT, "PRE-EVENT")
    process_image(POST_INPUT, POST_OUTPUT, "POST-EVENT")


if __name__ == "__main__":
    main()
