"""Download Sen1Floods11 from the public GCS bucket without gcloud.

The Sen1Floods11 bucket `gs://sen1floods11/v1.1/` is anonymously listable
via the GCS JSON API, so this script uses plain HTTP and stdlib only
(no gcloud, no google-cloud-storage dependency) to pull the data.

Default mode downloads only the hand-labeled portion (446 chips, ~1.5 GB)
which is enough for code development and the main fine-tuning phase.
Pass --target weak to also pull the 4,385 weakly-labeled chips
(~12 GB) needed for the warmup phase.

Usage
-----
    python -m model.download_sen1floods11 --dest data/sen1floods11
    python -m model.download_sen1floods11 --dest data/sen1floods11 --target weak
    python -m model.download_sen1floods11 --dest data/sen1floods11 --target both
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET = "sen1floods11"
API = "https://storage.googleapis.com/storage/v1/b"
MEDIA = "https://storage.googleapis.com"  # public download endpoint

# Subdirs of v1.1/ that we pull. Sen1Floods11 has both SAR (S1) and
# optical (S2) chips; this project is SAR-only so we skip S2.
HAND_PREFIXES = [
    "v1.1/splits/flood_handlabeled/",
    "v1.1/data/flood_events/HandLabeled/S1Hand/",
    "v1.1/data/flood_events/HandLabeled/LabelHand/",
    "v1.1/data/flood_events/HandLabeled/JRCWaterHand/",
]

WEAK_PREFIXES = [
    "v1.1/data/flood_events/WeaklyLabeled/",
]


def list_objects(prefix: str) -> list[dict]:
    """List all objects under a prefix via the public GCS JSON API."""
    out = []
    token = None
    while True:
        params = {"prefix": prefix, "maxResults": "1000", "fields": "items(name,size),nextPageToken"}
        if token:
            params["pageToken"] = token
        url = f"{API}/{BUCKET}/o?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=30) as r:
            page = json.load(r)
        out.extend(page.get("items", []))
        token = page.get("nextPageToken")
        if not token:
            break
    return out


def download_one(obj: dict, dest_root: Path) -> tuple[str, int, str]:
    """Download a single object if it's missing or wrong size. Returns (name, bytes, status)."""
    name = obj["name"]
    size = int(obj["size"])
    local_path = dest_root / name
    if local_path.exists() and local_path.stat().st_size == size:
        return (name, 0, "skip")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{MEDIA}/{BUCKET}/{urllib.parse.quote(name)}"
    with urllib.request.urlopen(url, timeout=60) as r, open(local_path, "wb") as f:
        while True:
            chunk = r.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            f.write(chunk)
    return (name, size, "ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dest", type=Path, required=True,
                        help="Local destination directory.")
    parser.add_argument("--target", choices=["hand", "weak", "both"], default="hand",
                        help="Which portion to download.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel download workers.")
    args = parser.parse_args()

    prefixes: list[str] = []
    if args.target in ("hand", "both"):
        prefixes.extend(HAND_PREFIXES)
    if args.target in ("weak", "both"):
        prefixes.extend(WEAK_PREFIXES)

    print(f"Listing {len(prefixes)} prefix(es) under gs://{BUCKET}/...")
    all_objects: list[dict] = []
    for p in prefixes:
        items = list_objects(p)
        total_bytes = sum(int(it["size"]) for it in items)
        print(f"  {p}: {len(items)} files, {total_bytes/1024/1024:.1f} MB")
        all_objects.extend(items)

    total_bytes = sum(int(it["size"]) for it in all_objects)
    print(f"\nTotal: {len(all_objects)} files, {total_bytes/1024/1024:.1f} MB")
    print(f"Downloading to: {args.dest}")
    print(f"Workers: {args.workers}\n")

    args.dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = 0
    bytes_done = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(download_one, obj, args.dest) for obj in all_objects]
        for fut in as_completed(futs):
            name, size, status = fut.result()
            done += 1
            if status == "ok":
                bytes_done += size
            elif status == "skip":
                skipped += 1
            if done % 20 == 0 or done == len(futs):
                rate = bytes_done / max(time.time() - t0, 1e-6) / 1024 / 1024
                print(f"  [{done:4d}/{len(futs):4d}] {bytes_done/1024/1024:7.1f} MB "
                      f"@ {rate:5.1f} MB/s (skipped {skipped})")
    print(f"\nDone in {time.time() - t0:.1f} s. {bytes_done/1024/1024:.1f} MB downloaded, "
          f"{skipped} already present.")


if __name__ == "__main__":
    main()
