"""Pakistan-2022 chip acquisition v2 — keep S1 at native 10 m, upsample labels.

This is a corrected version of acquire.py. The v1 pipeline reprojected
Sentinel-1 imagery from its native 10 m resolution onto the TU Wien mask
grid at 20 m, producing chips whose pixel scale was 2x the Sen1Floods11
training distribution. That mismatch confounded Pakistan-2022 IoU numbers
with a scale-shift artifact that is not part of the intended OOD test.

v2 inverts the resampling direction so the chips match training:
  - S1 is read at its native UTM CRS and 10 m pixel grid (no reprojection).
  - The TU Wien label is reprojected from Equi7Grid (EPSG:27703) 20 m
    onto the S1 UTM 10 m grid via nearest-neighbor.
  - Chips are 512x512 at 10 m, so each chip covers 5.12 x 5.12 km, matching
    Sen1Floods11 chips exactly.

Everything else (STAC search, SAS-token refresh, nodata handling, chip-level
validity filter, output layout) is preserved from v1 so the dataset loader
and downstream evaluation code do not need to change.

Run:
    python -m data_pipelines.pakistan_2022.acquire_v2 \
        --masks-dir D:/datasets/pakistan-2022/FLOOD-HM-MASKED \
        --out-dir   D:/datasets/pakistan-2022-chips-v2

After acquisition, point the dataset loader at the new chip root:
    python -m model.eval --split pakistan2022 \
        --pakistan2022-root D:/datasets/pakistan-2022-chips-v2 ...
"""

from __future__ import annotations

import argparse
import csv
import json
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
from rasterio.transform import Affine, rowcol, from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel1euwestrtc/sentinel1-grd-rtc"
PC_COLLECTION = "sentinel-1-rtc"

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
    with urllib.request.urlopen(PC_SAS, timeout=30) as r:
        return json.load(r)["token"]


def stac_search_s1(iso_time: str, bbox: tuple, window_minutes: int = 5) -> Optional[dict]:
    from datetime import datetime, timedelta

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
    best = min(features, key=lambda f: abs(
        datetime.fromisoformat(f["properties"]["datetime"].replace("Z", "+00:00")) - t
    ).total_seconds())
    return best


def signed_url(href: str, token: str) -> str:
    sep = "&" if "?" in href else "?"
    return f"{href}{sep}{token}"


def linear_to_db(arr: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(arr.astype(np.float32), 1e-10)).astype(np.float32)


RASTERIO_ENV = {
    "GDAL_HTTP_TIMEOUT": "120",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "CPL_VSIL_CURL_USE_HEAD": "NO",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
}


def acquire_pair(mask_path: Path, out_s1: Path, out_label: Path, token: str,
                 chip: int = 512) -> bool:
    """Build a (mask, S1) pair on the S1 native grid (UTM, 10 m).

    Output transform matches the S1 acquisition's native CRS and 10 m pixel
    size; the mask is upsampled from 20 m via nearest-neighbor to the same
    grid. This preserves S1 sampling and matches the Sen1Floods11 training
    chip resolution.
    """
    info = parse_mask_name(mask_path.name)
    if info is None:
        print(f"  skip {mask_path.name}: filename did not match pattern")
        return False

    with rasterio.open(mask_path) as msrc:
        mask_crs = msrc.crs
        mask_transform = msrc.transform
        mask_h, mask_w = msrc.height, msrc.width
        mask = msrc.read(1)

    valid_idx = np.where(mask != 255)
    if len(valid_idx[0]) == 0:
        print("  mask has no labelled pixels, skipping")
        return False

    r0, r1 = int(valid_idx[0].min()), int(valid_idx[0].max()) + 1
    c0, c1 = int(valid_idx[1].min()), int(valid_idx[1].max()) + 1

    # Labelled-region bounds in mask CRS (Equi7Grid).
    left, top = mask_transform * (c0, r0)
    right, bottom = mask_transform * (c1, r1)
    bbox_4326 = transform_bounds(mask_crs, "EPSG:4326",
                                  left, bottom, right, top)

    feat = stac_search_s1(info["iso"], bbox_4326)
    if feat is None:
        print(f"  no S1 RTC product found for {info['iso']}")
        return False
    print(f"  matched S1 RTC: {feat['id']}")

    vv_href = feat["assets"]["vv"]["href"]
    vh_href = feat["assets"]["vh"]["href"]

    # Open S1 VV once to learn its native CRS, transform, dimensions.
    current_token = token
    last_exc = None
    for attempt in range(3):
        try:
            with rasterio.Env(**RASTERIO_ENV):
                with rasterio.open(signed_url(vv_href, current_token)) as src:
                    s1_crs = src.crs
                    s1_transform = src.transform
                    s1_h, s1_w = src.height, src.width
            last_exc = None
            break
        except (rasterio.errors.RasterioIOError, OSError) as e:
            last_exc = e
            current_token = get_sas_token()
    if last_exc is not None:
        print(f"  could not open S1 metadata: {last_exc}")
        return False

    # Sanity: S1 RTC products are typically UTM at 10 m.
    s1_xres = abs(s1_transform.a)
    s1_yres = abs(s1_transform.e)
    print(f"  S1 native CRS={s1_crs.to_string()}  pixel={s1_xres:.2f}x{s1_yres:.2f}")

    # Convert labelled-region bbox from mask CRS to S1 native CRS.
    bbox_in_s1 = transform_bounds(mask_crs, s1_crs,
                                   left, bottom, right, top)
    sb_left, sb_bottom, sb_right, sb_top = bbox_in_s1

    # Pixel coords of bbox corners in S1.
    row_top, col_left = rowcol(s1_transform, sb_left, sb_top)
    row_bot, col_right = rowcol(s1_transform, sb_right, sb_bottom)
    row_min, row_max = min(row_top, row_bot), max(row_top, row_bot)
    col_min, col_max = min(col_left, col_right), max(col_left, col_right)

    # Snap to chip grid.
    row_min = (row_min // chip) * chip
    col_min = (col_min // chip) * chip
    row_max = ((row_max + chip - 1) // chip) * chip
    col_max = ((col_max + chip - 1) // chip) * chip
    row_min = max(0, row_min)
    col_min = max(0, col_min)
    row_max = min(s1_h, row_max)
    col_max = min(s1_w, col_max)
    sub_h, sub_w = row_max - row_min, col_max - col_min
    if sub_h < chip or sub_w < chip:
        print(f"  S1 overlap window too small ({sub_h}x{sub_w}), skipping")
        return False

    # Output transform: S1's native, translated to the labelled sub-window.
    target_transform = s1_transform * Affine.translation(col_min, row_min)

    # Read S1 bands at the labelled window (native, no reprojection).
    s1_dst = np.zeros((2, sub_h, sub_w), dtype=np.float32)
    s1_nodata = np.zeros((sub_h, sub_w), dtype=bool)
    window = Window(col_min, row_min, sub_w, sub_h)

    for band_idx, href in enumerate((vv_href, vh_href)):
        last_exc = None
        for attempt in range(3):
            try:
                with rasterio.Env(**RASTERIO_ENV):
                    with rasterio.open(signed_url(href, current_token)) as src:
                        data = src.read(1, window=window).astype(np.float32)
                last_exc = None
                break
            except (rasterio.errors.RasterioIOError, OSError) as e:
                last_exc = e
                print(f"  band {band_idx} read failed (attempt {attempt+1}/3): refreshing token")
                current_token = get_sas_token()
        if last_exc is not None:
            print(f"  giving up on this acquisition after 3 retries")
            return False
        s1_dst[band_idx] = linear_to_db(data)
        s1_nodata |= (data == 0) | ~np.isfinite(data) | (data < 1e-9)

    s1_valid_frac = 1.0 - s1_nodata.sum() / s1_nodata.size
    print(f"  S1 valid fraction: {s1_valid_frac:.3f}")
    if s1_valid_frac < 0.05:
        print(f"  S1 does not overlap labelled region meaningfully, skipping")
        return False

    # Reproject mask from Equi7Grid 20 m to S1 UTM 10 m via nearest-neighbor.
    label_dst = np.full((sub_h, sub_w), 255, dtype=np.uint8)
    reproject(
        source=mask,
        destination=label_dst,
        src_transform=mask_transform,
        src_crs=mask_crs,
        dst_transform=target_transform,
        dst_crs=s1_crs,
        resampling=Resampling.nearest,
        src_nodata=255,
        dst_nodata=255,
    )
    # Apply S1 nodata to the label so we don't score where SAR is missing.
    label_dst = np.where(s1_nodata, 255, label_dst)

    # Write outputs.
    out_s1.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_s1, "w", driver="GTiff", height=sub_h, width=sub_w, count=2,
        dtype="float32", crs=s1_crs, transform=target_transform,
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
        dtype="uint8", crs=s1_crs, transform=target_transform,
        nodata=255, compress="lzw", tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(label_dst, 1)
    return True


def tile_pair(s1_path: Path, label_path: Path, out_dir: Path,
              chip_size: int = 512, min_valid_fraction: float = 0.95,
              min_label_fraction: float = 0.5,
              max_chips_per_pair: int = 200) -> list[str]:
    """Tile a (S1, mask) pair to 512x512 chips at 10 m."""
    out_s1 = out_dir / "v1.1" / "data" / "flood_events" / "HandLabeled" / "S1Hand"
    out_lb = out_dir / "v1.1" / "data" / "flood_events" / "HandLabeled" / "LabelHand"
    out_s1.mkdir(parents=True, exist_ok=True)
    out_lb.mkdir(parents=True, exist_ok=True)

    stem = s1_path.stem.replace("_S1", "")
    written: list[str] = []
    with rasterio.open(s1_path) as s1, rasterio.open(label_path) as lab:
        H, W = s1.height, s1.width
        s1_nodata = s1.nodata if s1.nodata is not None else -9999

        chip_idx = 0
        for row in range(0, H - chip_size + 1, chip_size):
            for col in range(0, W - chip_size + 1, chip_size):
                if chip_idx >= max_chips_per_pair:
                    break
                window = Window(col, row, chip_size, chip_size)
                s1_chunk = s1.read(window=window)
                lab_chunk = lab.read(1, window=window)

                bad = (s1_chunk == s1_nodata) | (s1_chunk <= -50)
                valid_s1 = 1.0 - bad.sum() / s1_chunk.size
                valid_lab = float((lab_chunk != 255).sum()) / lab_chunk.size
                if valid_s1 < min_valid_fraction or valid_lab < min_label_fraction:
                    continue

                chip_name = f"{stem}_chip{chip_idx:04d}"
                s1_chip_path = out_s1 / f"{chip_name}_S1Hand.tif"
                lb_chip_path = out_lb / f"{chip_name}_LabelHand.tif"

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
                   default=Path("D:/datasets/pakistan-2022/FLOOD-HM-MASKED"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("D:/datasets/pakistan-2022-chips-v2"))
    p.add_argument("--max-acquisitions", type=int, default=8)
    p.add_argument("--max-masks-per-acq", type=int, default=1)
    args = p.parse_args()

    all_masks = sorted(args.masks_dir.rglob("*.tif"))
    print(f"[acquire-v2] {len(all_masks)} mask files under {args.masks_dir}")

    grouped: dict[tuple, list[Path]] = defaultdict(list)
    for m in all_masks:
        info = parse_mask_name(m.name)
        if info is None:
            continue
        key = (info["date"], info["hms"], info["orbit"])
        grouped[key].append(m)

    acqs = sorted(grouped.keys())
    print(f"[acquire-v2] {len(acqs)} unique acquisitions; sampling {min(args.max_acquisitions, len(acqs))}")
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
        ranked = []
        for m in grouped[k]:
            try:
                with rasterio.open(m) as src:
                    labelled = int((src.read(1) != 255).sum())
                    if labelled > 0:
                        ranked.append((labelled, m))
            except Exception as e:
                print(f"  could not read {m.name}: {e}")
        ranked.sort(reverse=True)
        masks = [m for _, m in ranked[: args.max_masks_per_acq]]
        if not masks:
            print(f"[acquire-v2] {date}T{hms} {orbit}: no labelled tiles, skipping")
            continue

        for mask_path in masks:
            tile_id = parse_mask_name(mask_path.name)["tile_id"]
            acq_id = f"{date}T{hms}_{orbit}_{tile_id}"
            print(f"[acquire-v2] === {acq_id} ===")
            if time.time() - token_time > 20 * 60:
                token = get_sas_token()
                token_time = time.time()
            s1_path = workdir / f"Pakistan2022_{acq_id}_S1.tif"
            lb_path = workdir / f"Pakistan2022_{acq_id}_Label.tif"
            if not (s1_path.exists() and lb_path.exists()):
                ok = acquire_pair(mask_path, s1_path, lb_path, token)
                if not ok:
                    continue
            else:
                print(f"  pair already on disk, skipping acquisition")
            chips = tile_pair(s1_path, lb_path, args.out_dir)
            print(f"  wrote {len(chips)} chips at 10 m")
            all_chips.extend(chips)

    splits_dir = args.out_dir / "v1.1" / "splits" / "flood_handlabeled"
    splits_dir.mkdir(parents=True, exist_ok=True)
    csv_path = splits_dir / "flood_pakistan2022_data.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        for chip in all_chips:
            w.writerow([f"{chip}_S1Hand.tif", f"{chip}_LabelHand.tif"])
    print(f"[acquire-v2] wrote split CSV with {len(all_chips)} chips -> {csv_path}")


if __name__ == "__main__":
    main()
