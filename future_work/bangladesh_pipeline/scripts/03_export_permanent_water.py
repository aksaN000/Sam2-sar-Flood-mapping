"""
Export the JRC Global Surface Water permanent-water mask for the AOI.

Background
----------
The Joint Research Centre of the European Commission, in collaboration with
Google, has produced a global dataset that records how often each 30-meter
pixel on Earth's surface was covered by water between 1984 and 2021. The
dataset has several layers; we use the 'seasonality' layer, which counts
the number of months per year (out of 12) that a given pixel was wet.

Why we need this
----------------
Our Sentinel-1 imagery makes water pixels appear dark, but radar cannot
distinguish "this pixel is dark because of the flood" from "this pixel
is dark because there is normally a river here". If we did not correct
for this, our model would learn to label permanent rivers as 'flood'
which is technically incorrect and would inflate evaluation scores in a
misleading way. By masking out pixels that are normally wet (seasonality
greater than 5 months per year), we ensure that the labels reflect only
flood-induced inundation, not baseline river extent.

What this script does
---------------------
1. Loads the AOI polygon from data/aoi/bangladesh_2022_aoi.geojson
2. Loads the JRC Global Surface Water v1.4 seasonality layer
3. Thresholds it at 5 months per year to define 'permanent water'
4. Exports the result as a single-band GeoTIFF to your Google Drive

Output
------
A GeoTIFF at data/raw/permanent_water_mask.tif (after you download it
from Drive). The file has one band where 1 means 'normally wet' and 0
means 'sometimes dry or always dry'. Pixel grid matches the Sentinel-1
exports, which lets us simply multiply or subtract them in later scripts.
"""

import ee
import json
from pathlib import Path

JRC_GSW_ASSET = "JRC/GSW1_4/GlobalSurfaceWater"
SEASONALITY_THRESHOLD = 5  # months per year wet
EXPORT_SCALE_M = 10


def main() -> None:
    print("Initializing Earth Engine session...")
    ee.Initialize()

    aoi_path = Path(__file__).parent.parent / "data" / "aoi" / "bangladesh_2022_aoi.geojson"
    with open(aoi_path) as f:
        aoi_geojson = json.load(f)
    aoi_fc = ee.FeatureCollection(aoi_geojson["features"])
    aoi_geom = aoi_fc.geometry()

    # Load the JRC global surface water dataset and pull the seasonality band
    gsw = ee.Image(JRC_GSW_ASSET)
    seasonality = gsw.select("seasonality")

    # Threshold to produce a binary permanent-water mask. Pixels where
    # seasonality > 5 months are treated as 'normally wet' and excluded
    # from flood labels later in the pipeline.
    permanent_water = seasonality.gt(SEASONALITY_THRESHOLD).clip(aoi_geom)

    print("Submitting export task for permanent water mask...")
    task = ee.batch.Export.image.toDrive(
        image=permanent_water,
        description="permanent_water_mask",
        folder="bangladesh_sylhet_2022_s1",
        fileNamePrefix="permanent_water_mask",
        region=aoi_geom,
        scale=EXPORT_SCALE_M,
        crs="EPSG:4326",
        maxPixels=1e10,
    )
    task.start()
    print(f"  Task ID: {task.id}")
    print()
    print("Once complete, download the file from Google Drive to ./data/raw/")
    print("Monitor at https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
