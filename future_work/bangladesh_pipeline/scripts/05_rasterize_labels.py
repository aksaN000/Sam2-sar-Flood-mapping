"""
Rasterize the reference flood polygons onto the same grid as the
Sentinel-1 imagery, then subtract the permanent water mask to produce
the final binary flood-label raster.

This version reads from three possible source directories and unions
whatever it finds. The primary source is UNOSAT FL20220525BGD shapefiles
in data/raw/unosat_FL20220525BGD/. The secondary source is Copernicus
GFM rasters in data/raw/gfm_ofe/. The tertiary source is International
Charter shapefiles in data/raw/charter_762/. The script gracefully
handles missing sources, so you can run it as soon as you have any one
of them and add others later.

Pipeline
--------
1. Load the post-event Sentinel-1 image and read its grid metadata
   (CRS, transform, height, width). All later outputs will use these
   exact values so that pixel (i, j) in the image corresponds to
   pixel (i, j) in every other output raster.
2. Walk each source directory, collect any shapefiles or rasters present,
   and union them into a single binary mask aligned to the reference grid.
3. Subtract the permanent water mask from the union, leaving only
   pixels that are flooded AND not normally wet.
4. Save the final binary label raster.

Output: data/processed/flood_label.tif
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from pandas import concat as pd_concat

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

POST_EVENT_TIF = RAW_DIR / "sylhet_post_event.tif"
PERMANENT_WATER_TIF = RAW_DIR / "permanent_water_mask.tif"

UNOSAT_DIR = RAW_DIR / "unosat_FL20220525BGD"
GFM_DIR = RAW_DIR / "gfm_ofe"
CHARTER_DIR = RAW_DIR / "charter_762"

OUTPUT_LABEL_TIF = PROCESSED_DIR / "flood_label.tif"


def load_grid_reference():
    """Read the Sentinel-1 post-event GeoTIFF for its grid metadata."""
    if not POST_EVENT_TIF.exists():
        sys.exit(
            f"Cannot find {POST_EVENT_TIF}. Run script 02 first and "
            f"download the export from your Google Drive into data/raw/."
        )
    with rasterio.open(POST_EVENT_TIF) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "height": src.height,
            "width": src.width,
            "shape": (src.height, src.width),
        }


def rasterize_shapefile_dir(directory, grid, label):
    """Find all .shp files in a directory, union, reproject, and rasterize."""
    if not directory.exists():
        print(f"  {label}: directory does not exist, skipping")
        return np.zeros(grid["shape"], dtype=np.uint8)

    shapefiles = list(directory.rglob("*.shp"))
    if not shapefiles:
        print(f"  {label}: no shapefiles found in {directory}, skipping")
        return np.zeros(grid["shape"], dtype=np.uint8)

    print(f"  {label}: found {len(shapefiles)} shapefile(s)")
    pieces = []
    for shp in shapefiles:
        try:
            gdf = gpd.read_file(shp)
            if gdf.crs != grid["crs"]:
                gdf = gdf.to_crs(grid["crs"])
            pieces.append(gdf)
        except Exception as e:
            print(f"    Warning: could not read {shp}: {e}")

    if not pieces:
        return np.zeros(grid["shape"], dtype=np.uint8)

    merged_df = pd_concat(pieces, ignore_index=True)
    merged = gpd.GeoDataFrame(merged_df, crs=grid["crs"])

    if len(merged) == 0:
        return np.zeros(grid["shape"], dtype=np.uint8)

    shapes = ((geom, 1) for geom in merged.geometry if geom is not None and not geom.is_empty)
    raster = rasterize(
        shapes=shapes,
        out_shape=grid["shape"],
        transform=grid["transform"],
        fill=0,
        dtype=np.uint8,
    )
    print(f"  {label}: {int(raster.sum())} flooded pixels")
    return raster


def rasterize_gfm_directory(directory, grid):
    """Find any GeoTIFFs in the GFM directory and reproject to grid."""
    if not directory.exists():
        print(f"  GFM: directory does not exist, skipping")
        return np.zeros(grid["shape"], dtype=np.uint8)
    tifs = list(directory.rglob("*.tif")) + list(directory.rglob("*.tiff"))
    if not tifs:
        print(f"  GFM: no GeoTIFFs found, skipping")
        return np.zeros(grid["shape"], dtype=np.uint8)

    print(f"  GFM: found {len(tifs)} raster(s)")
    out = np.zeros(grid["shape"], dtype=np.uint8)
    for tif in tifs:
        with rasterio.open(tif) as src:
            buf = np.zeros(grid["shape"], dtype=np.uint8)
            reproject(
                source=rasterio.band(src, 1),
                destination=buf,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=grid["transform"], dst_crs=grid["crs"],
                resampling=Resampling.nearest,
            )
            out = np.maximum(out, (buf > 0).astype(np.uint8))
    print(f"  GFM: {int(out.sum())} flooded pixels")
    return out


def load_permanent_water(grid):
    """Load permanent-water mask and reproject to the reference grid."""
    if not PERMANENT_WATER_TIF.exists():
        sys.exit(f"Cannot find {PERMANENT_WATER_TIF}. Run script 03 first.")
    with rasterio.open(PERMANENT_WATER_TIF) as src:
        if src.crs == grid["crs"] and src.shape == grid["shape"]:
            return (src.read(1) > 0).astype(np.uint8)
        out = np.zeros(grid["shape"], dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=grid["transform"], dst_crs=grid["crs"],
            resampling=Resampling.nearest,
        )
        return (out > 0).astype(np.uint8)


def save_label(label, grid, path):
    """Write the binary label raster as a GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "height": grid["height"],
        "width": grid["width"],
        "crs": grid["crs"],
        "transform": grid["transform"],
        "compress": "lzw",
        "nodata": 255,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(label, 1)
    print(f"\nSaved final label raster to {path}")


def main() -> None:
    grid = load_grid_reference()
    print(f"Reference grid: {grid['width']} x {grid['height']} pixels in {grid['crs']}")
    print()

    print("Loading reference labels from up to three sources:")
    unosat_raster = rasterize_shapefile_dir(UNOSAT_DIR, grid, "UNOSAT FL20220525BGD")
    gfm_raster = rasterize_gfm_directory(GFM_DIR, grid)
    charter_raster = rasterize_shapefile_dir(CHARTER_DIR, grid, "Charter Activation 762")

    # Logical OR across all available sources
    flood_union = ((unosat_raster > 0) | (gfm_raster > 0) | (charter_raster > 0)).astype(np.uint8)
    n_total = int(flood_union.sum())
    print(f"\nUnion of all sources: {n_total} flooded pixels")
    if n_total == 0:
        sys.exit(
            "No flood pixels found in any source. Make sure you have downloaded "
            "at least one of the UNOSAT, GFM, or Charter products into data/raw/."
        )

    permanent_water = load_permanent_water(grid)
    print(f"Permanent water mask excludes: {int(permanent_water.sum())} pixels")
    final_label = (flood_union & (1 - permanent_water)).astype(np.uint8)
    print(f"Final flood label: {int(final_label.sum())} pixels classified as flooded")

    save_label(final_label, grid, OUTPUT_LABEL_TIF)


if __name__ == "__main__":
    main()
