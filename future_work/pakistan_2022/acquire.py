"""Acquire matching Sentinel-1 GRD imagery for the TU Wien Pakistan-2022 flood masks.

The TU Wien release ships flood-extent masks on Equi7Grid (EPSG:27703) at
20 m resolution. Each filename encodes the originating S1 acquisition
(date, time, orbit, tile). This script:

1. Parses the mask filenames to recover acquisition metadata.
2. Picks a representative sample of mask tiles across dates and orbits.
3. Queries the Microsoft Planetary Computer Sentinel-1 GRD STAC catalog
   for each acquisition. Anonymous SAS tokens are issued by PC's free
   `/api/sas/v1/token/...` endpoint -- no signup required.
4. For each (mask, S1) pair, uses rasterio's WarpedVRT to read the S1
   VV+VH bands lazily over HTTP, reprojecting on the fly to the mask
   grid (same CRS, transform, shape).
5. Converts S1 linear power values to dB (matching the Sen1Floods11
   convention) and writes a pair of pixel-aligned GeoTIFFs to disk:
       <out>/s1/<chip_id>_S1.tif    2-band float32 VV/VH dB
       <out>/label/<chip_id>_Label.tif    1-band uint8 (0/1/255)
6. Tiles each pair to 512x512 chips, discarding chips with >5% nodata
   in either source, and writes an index CSV listing the surviving chips.

Output directory layout (mirrors Sen1Floods11 conventions):
    <out>/v1.1/data/flood_events/HandLabeled/S1Hand/<chip>_S1Hand.tif
    <out>/v1.1/data/flood_events/HandLabeled/LabelHand/<chip>_LabelHand.tif
    <out>/v1.1/splits/flood_handlabeled/flood_pakistan2022_data.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.windows import Window

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel1euwestrtc/sentinel1-grd-rtc"
PC_COLLECTION = "sentinel-1-rtc"  # Radiometric Terrain Corrected, fully georeferenced

# Equi7Grid Asia CRS used by the TU Wien Pakistan-2022 release.
EQUI7_AS = CRS.from_epsg(27703)

MASK_RE = re.compile(
    r"FLOOD-HM-MASKED_(\d{8})T(\d{6})__(\w+)_([AD]\d+)_(E\d+N\d+T\d)_.*\.tif"
)


def parse_mask_name(name: str) -> dict | None:
    m = MASK_RE.match(name)
    if not m:
        return None
    date, hms, pol, orbit, tile = m.groups()
    return {
        "date": date,
        "hms": hms,
        "iso": f"{date[:4]}-{date[4:6]}-{date[6:8]}T{hms[:2]}:{hms[2:4]}:{hms[4:6]}Z",
        "pol": pol,
        "orbit": orbit,
        "orbit_dir": "ASCENDING" if orbit.startswith("A") else "DESCENDING",
        "tile_id": tile,
    }


def get_sas_token() -> str:
    """Get a fresh anonymous SAS token from Planetary Computer."""
    with urllib.request.urlopen(PC_SAS, timeout=30) as r:
        import json
        return json.load(r)["token"]


def stac_search_s1(iso_time: str, bbox: tuple[float, float, float, float],
                   window_minutes: int = 5) -> Optional[dict]:
    """Find the S1 GRD product whose acquisition window contains `iso_time`."""
    import json
    from datetime import datetime, timedelta, timezone

    t = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
    lo = (t - timedelta(minutes=window_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
    hi = (t + timedelta(minutes=window_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = {
        "collections": [PC_COLLECTION],
        "datetime": f"{lo}/{hi}",
        "bbox": list(bbox),
        "limit": 5,
    }
    req = urllib.request.Request(
        PC_STAC, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    features = data.get("features", [])
    if not features:
        return None
    # Prefer the feature with start_datetime closest to iso_time.
    best = min(features, key=lambda f: abs(
        datetime.fromisoformat(f["properties"]["datetime"].replace("Z", "+00:00")) - t
    ).total_seconds())
    return best


def signed_url(href: str, token: str) -> str:
    sep = "&" if "?" in href else "?"
    return f"{href}{sep}{token}"


def linear_to_db(arr: np.ndarray) -> np.ndarray:
    """Convert linear power to dB, with a small floor."""
    return 10.0 * np.log10(np.maximum(arr.astype(np.float32), 1e-10)).astype(np.float32)


def acquire_pair(mask_path: Path, out_s1: Path, out_label: Path, token: str) -> bool:
    """Build a (mask, S1) pair on the same grid, cropped to the labelled sub-bbox.

    The TU Wien masks are 15000x15000 but only ~5-15% of pixels are non-255
    (labelled). To avoid reprojecting hundreds of millions of empty pixels,
    we crop both the mask and the S1 warp window to the bounding rectangle
    of the valid-label region, snapped to a 512-pixel grid for clean tiling.
    """
    info = parse_mask_name(mask_path.name)
    if info is None:
        print(f"  skip {mask_path.name}: filename did not match pattern")
        return False

    # Read mask + find labelled sub-bbox.
    with rasterio.open(mask_path) as msrc:
        grid_crs = msrc.crs
        full_transform = msrc.transform
        full_h, full_w = msrc.height, msrc.width
        mask = msrc.read(1)

    valid_idx = np.where(mask != 255)
    if len(valid_idx[0]) == 0:
        print(f"  mask has no labelled pixels, skipping")
        return False

    r0, r1 = int(valid_idx[0].min()), int(valid_idx[0].max()) + 1
    c0, c1 = int(valid_idx[1].min()), int(valid_idx[1].max()) + 1
    # Snap to 512-pixel multiples for clean chip tiling.
    chip = 512
    r0 = (r0 // chip) * chip
    c0 = (c0 // chip) * chip
    r1 = ((r1 + chip - 1) // chip) * chip
    c1 = ((c1 + chip - 1) // chip) * chip
    r0, c0 = max(0, r0), max(0, c0)
    r1 = min(full_h, r1)
    c1 = min(full_w, c1)
    sub_h, sub_w = r1 - r0, c1 - c0
    valid_frac = ((mask[r0:r1, c0:c1] != 255).sum() / (sub_h * sub_w))
    print(f"  labelled sub-bbox: rows[{r0}:{r1}] cols[{c0}:{c1}]  "
          f"shape=({sub_h},{sub_w})  valid-label fraction={valid_frac:.3f}")
    if sub_h < chip or sub_w < chip:
        print(f"  labelled region smaller than one chip, skipping")
        return False

    # Sub-grid transform.
    from rasterio.transform import Affine
    sub_transform = full_transform * Affine.translation(c0, r0)

    # STAC search using sub-bbox in lat/lon.
    from rasterio.warp import transform_bounds
    sub_bounds_native = (
        full_transform * (c0, r1),       # left, bottom
        full_transform * (c1, r0),       # right, top
    )
    left, bottom = sub_bounds_native[0]
    right, top = sub_bounds_native[1]
    bbox_4326 = transform_bounds(grid_crs, "EPSG:4326", left, bottom, right, top)

    feat = stac_search_s1(info["iso"], bbox_4326)
    if feat is None:
        print(f"  no S1 RTC product found for {info['iso']}")
        return False
    print(f"  matched S1 RTC: {feat['id']}")

    vv_href = feat["assets"]["vv"]["href"]
    vh_href = feat["assets"]["vh"]["href"]

    sub_mask = mask[r0:r1, c0:c1].astype(np.uint8)
    s1_dst = np.zeros((2, sub_h, sub_w), dtype=np.float32)
    s1_nodata = np.zeros((sub_h, sub_w), dtype=bool)

    rasterio_env = {
        "GDAL_HTTP_TIMEOUT": "120",
        "GDAL_HTTP_MAX_RETRY": "5",
        "GDAL_HTTP_RETRY_DELAY": "2",
        "CPL_VSIL_CURL_USE_HEAD": "NO",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    }
    current_token = token
    for band_idx, href in enumerate((vv_href, vh_href)):
        # Retry on TIFF read failures by refreshing the SAS token. Mid-stream
        # token expiry is the dominant failure mode.
        last_exc = None
        for attempt in range(3):
            try:
                url = signed_url(href, current_token)
                with rasterio.Env(**rasterio_env):
                    with rasterio.open(url) as src:
                        with WarpedVRT(
                            src, crs=grid_crs, transform=sub_transform,
                            height=sub_h, width=sub_w,
                            resampling=Resampling.bilinear,
                            src_nodata=0, nodata=0,
                        ) as vrt:
                            data = vrt.read(1).astype(np.float32)
                last_exc = None
                break
            except (rasterio.errors.RasterioIOError, OSError) as e:
                last_exc = e
                print(f"  band {band_idx} read failed (attempt {attempt+1}/3): {e}; refreshing token")
                current_token = get_sas_token()
        if last_exc is not None:
            print(f"  giving up on this acquisition after 3 retries")
            return False
        s1_dst[band_idx] = linear_to_db(data)
        # Broaden nodata detection: zero, NaN, and near-zero linear values
        # all produce -100 dB (or worse) after log10 clamp at 1e-10, which
        # is far outside Sen1Floods11's training distribution (-40 to +12 dB).
        # If we don't mask them here they pass through linear_to_db and end
        # up as -100 dB in the final chip, biasing inference.
        s1_nodata |= (data == 0) | ~np.isfinite(data) | (data < 1e-9)

    s1_valid_frac = 1.0 - s1_nodata.sum() / s1_nodata.size
    print(f"  S1 valid fraction in sub-bbox: {s1_valid_frac:.3f}")
    if s1_valid_frac < 0.05:
        print(f"  S1 does not overlap labelled region meaningfully, skipping")
        return False

    # Apply S1 nodata to the label so we don't score where SAR is missing.
    sub_mask = np.where(s1_nodata, 255, sub_mask)

    # Write paired GeoTIFFs at the sub-grid resolution.
    out_s1.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_s1, "w", driver="GTiff", height=sub_h, width=sub_w, count=2,
        dtype="float32", crs=grid_crs, transform=sub_transform,
        nodata=-9999, compress="lzw", tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        for b in range(2):
            band = s1_dst[b].copy()
            band[s1_nodata] = -9999
            dst.write(band, b + 1)
        dst.set_band_description(1, "VV_dB")
        dst.set_band_description(2, "VH_dB")

    out_label.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_label, "w", driver="GTiff", height=sub_h, width=sub_w, count=1,
        dtype="uint8", crs=grid_crs, transform=sub_transform,
        nodata=255, compress="lzw", tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(sub_mask, 1)
    return True


def tile_pair(s1_path: Path, label_path: Path, out_dir: Path,
              chip_size: int = 512, min_valid_fraction: float = 0.95,
              max_chips_per_pair: int = 200) -> list[str]:
    """Tile a (S1, mask) pair to 512x512 chips. Returns the list of chip names."""
    out_s1 = out_dir / "v1.1" / "data" / "flood_events" / "HandLabeled" / "S1Hand"
    out_lb = out_dir / "v1.1" / "data" / "flood_events" / "HandLabeled" / "LabelHand"
    out_s1.mkdir(parents=True, exist_ok=True)
    out_lb.mkdir(parents=True, exist_ok=True)

    stem = s1_path.stem.replace("_S1", "")  # "Pakistan2022_<acqid>_<tile>"
    written = []
    with rasterio.open(s1_path) as s1, rasterio.open(label_path) as lab:
        H, W = s1.height, s1.width
        s1_nodata = s1.nodata if s1.nodata is not None else -9999

        # Walk a grid of non-overlapping chip starts.
        chip_idx = 0
        for row in range(0, H - chip_size + 1, chip_size):
            for col in range(0, W - chip_size + 1, chip_size):
                if chip_idx >= max_chips_per_pair:
                    break
                window = Window(col, row, chip_size, chip_size)
                s1_chunk = s1.read(window=window)
                lab_chunk = lab.read(1, window=window)

                # Treat both the declared nodata sentinel (-9999) AND any
                # dB value <= -50 as nodata. The latter catches pixels that
                # slipped past acquire_pair's mask because their linear
                # backscatter was tiny-but-nonzero (< 1e-9 W/m^2).
                bad = (s1_chunk == s1_nodata) | (s1_chunk <= -50)
                valid_s1 = 1.0 - bad.sum() / s1_chunk.size
                valid_lab = np.sum(lab_chunk != 255) / lab_chunk.size
                # Require valid imagery AND at least some real label coverage.
                if valid_s1 < min_valid_fraction or valid_lab < 0.5:
                    continue

                chip_name = f"{stem}_chip{chip_idx:04d}"
                s1_chip_path = out_s1 / f"{chip_name}_S1Hand.tif"
                lb_chip_path = out_lb / f"{chip_name}_LabelHand.tif"

                # Write chips with per-window transform so geo metadata is correct.
                from rasterio.windows import transform as window_transform
                s1_tr = window_transform(window, s1.transform)
                lab_tr = window_transform(window, lab.transform)
                with rasterio.open(
                    s1_chip_path, "w", driver="GTiff",
                    height=chip_size, width=chip_size, count=2,
                    dtype="float32", crs=s1.crs, transform=s1_tr,
                    nodata=-9999, compress="lzw",
                ) as dst:
                    dst.write(s1_chunk)
                with rasterio.open(
                    lb_chip_path, "w", driver="GTiff",
                    height=chip_size, width=chip_size, count=1,
                    dtype="uint8", crs=lab.crs, transform=lab_tr,
                    nodata=255, compress="lzw",
                ) as dst:
                    dst.write(lab_chunk, 1)

                written.append(chip_name)
                chip_idx += 1
    return written


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--masks-dir", type=Path,
                   default=Path("D:/datasets/pakistan-2022/FLOOD-HM-MASKED"),
                   help="Root containing TU Wien Pakistan-2022 mask GeoTIFFs.")
    p.add_argument("--out-dir", type=Path,
                   default=Path("D:/datasets/pakistan-2022-chips"),
                   help="Destination root, mirroring Sen1Floods11 layout.")
    p.add_argument("--max-acquisitions", type=int, default=8,
                   help="Sample at most this many unique (date, time, orbit) acquisitions.")
    p.add_argument("--max-masks-per-acq", type=int, default=1,
                   help="Per acquisition, take at most this many mask tiles.")
    args = p.parse_args()

    all_masks = sorted(args.masks_dir.rglob("*.tif"))
    print(f"[acquire] discovered {len(all_masks)} mask files under {args.masks_dir}")

    grouped: dict[tuple, list[Path]] = defaultdict(list)
    for m in all_masks:
        info = parse_mask_name(m.name)
        if info is None:
            continue
        key = (info["date"], info["hms"], info["orbit"])
        grouped[key].append(m)

    acqs = sorted(grouped.keys())
    print(f"[acquire] {len(acqs)} unique acquisitions; sampling {min(args.max_acquisitions, len(acqs))}")
    # Sample evenly spaced acquisitions for time-diversity coverage.
    if len(acqs) > args.max_acquisitions:
        step = len(acqs) / args.max_acquisitions
        acqs = [acqs[int(i * step)] for i in range(args.max_acquisitions)]

    workdir = args.out_dir / "_workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    token = get_sas_token()
    token_time = time.time()

    all_chips = []
    for k in acqs:
        date, hms, orbit = k
        # Rank this acquisition's mask tiles by labelled-pixel count so we
        # pick the densest ones first. Reading 15000x15000 uint8 is ~225 MB
        # per tile so we stream-count instead of holding all in RAM.
        ranked = []
        for m in grouped[k]:
            try:
                with rasterio.open(m) as src:
                    sample = src.read(1)
                    labelled = int((sample != 255).sum())
                    if labelled > 0:
                        ranked.append((labelled, m))
            except Exception as e:
                print(f"  could not read {m.name}: {e}")
        ranked.sort(reverse=True)  # most-labelled first
        masks = [m for _, m in ranked[: args.max_masks_per_acq]]
        if not masks:
            print(f"[acquire] === {date}T{hms} {orbit}: no labelled tiles, skipping ===")
            continue
        for mask_path in masks:
            tile_id = parse_mask_name(mask_path.name)["tile_id"]
            acq_id = f"{date}T{hms}_{orbit}_{tile_id}"
            print(f"[acquire] === {acq_id} ===")
            # Refresh SAS token every 20 min to be safe (45 min lifetime).
            if time.time() - token_time > 20 * 60:
                token = get_sas_token()
                token_time = time.time()
            s1_path = workdir / f"Pakistan2022_{acq_id}_S1.tif"
            lb_path = workdir / f"Pakistan2022_{acq_id}_Label.tif"
            if s1_path.exists() and lb_path.exists():
                print(f"  pair already on disk, skipping acquisition")
            else:
                ok = acquire_pair(mask_path, s1_path, lb_path, token)
                if not ok:
                    continue
            print(f"  tiling to 512x512 chips...")
            chips = tile_pair(s1_path, lb_path, args.out_dir)
            print(f"  wrote {len(chips)} chips")
            all_chips.extend(chips)

    # Write the split CSV in Sen1Floods11 layout so Sen1Floods11Dataset can read it.
    splits_dir = args.out_dir / "v1.1" / "splits" / "flood_handlabeled"
    splits_dir.mkdir(parents=True, exist_ok=True)
    csv_path = splits_dir / "flood_pakistan2022_data.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        for chip in all_chips:
            w.writerow([f"{chip}_S1Hand.tif", f"{chip}_LabelHand.tif"])
    print(f"[acquire] wrote split CSV with {len(all_chips)} chips -> {csv_path}")


if __name__ == "__main__":
    main()
