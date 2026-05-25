"""
Export Sentinel-1 GRD imagery from Google Earth Engine for the Bangladesh
2022 flood area-of-interest, for both the pre-event and post-event time
windows. The exports are routed to your Google Drive in a folder called
'bangladesh_sylhet_2022_s1', from which you will download them to disk
manually after the export tasks complete.

PREREQUISITES
-------------
1. A Google account.
2. Earth Engine access. If you have not used Earth Engine before, sign up
   for free at https://earthengine.google.com/signup/. Approval is usually
   instant for academic use; account confirmation takes a few minutes.
3. The earthengine-api Python package, installed via:
       pip install earthengine-api
4. Authenticate once per machine by running:
       earthengine authenticate
   in your terminal. This opens a browser, you sign in with the same
   Google account that has Earth Engine access, and a credentials file
   is stored in your home directory.

WHAT THIS SCRIPT DOES
---------------------
The script performs four logical steps. First it loads the AOI polygon
that was pre-built by 01_build_aoi.py. Second it queries the Sentinel-1
ImageCollection on Earth Engine, filtering by date, polarization, and
acquisition mode to retrieve only the IW-mode descending-pass acquisitions
that intersect the AOI in two specific time windows. Third it computes
median composites for each window so that you have one clean pre-event
image and one clean post-event image rather than dozens of overlapping
acquisitions. Fourth it submits two export tasks to Earth Engine which
will run on Google's servers in the background and write the results
to your Google Drive.

OUTPUTS
-------
Two GeoTIFF files in your Google Drive folder 'bangladesh_sylhet_2022_s1':
    sylhet_pre_event.tif   (median of 1-10 June 2022 acquisitions)
    sylhet_post_event.tif  (median of 17-25 June 2022 acquisitions)

Each GeoTIFF has two bands: VV and VH polarizations, in linear power
units rather than decibels. The conversion to decibels happens later in
script 06_apply_speckle_filter.py because the Refined Lee speckle filter
is most accurately applied to linear power values.

TYPICAL RUNTIME
---------------
Earth Engine task submission is instantaneous. The actual export jobs
typically take 30-90 minutes each depending on the queue. You can monitor
their progress at https://code.earthengine.google.com/tasks while they run.
"""

import ee
from pathlib import Path

# Pre-event window covers the days before the major flooding began,
# giving us a baseline radar image of the dry-state landscape.
PRE_EVENT_START = "2022-06-01"
PRE_EVENT_END = "2022-06-10"

# Post-event window brackets the peak inundation according to UNOSAT
# and the International Charter satellite assessments dated 18-21 June.
# Going to 25 June gives us a few extra acquisitions in case some of
# them are obscured by acquisition geometry issues.
POST_EVENT_START = "2022-06-17"
POST_EVENT_END = "2022-06-25"

# Output resolution. Sentinel-1 IW GRD products have a native ground
# resolution of about 20 meters and a pixel spacing of 10 meters, so
# 10 m export is appropriate.
EXPORT_SCALE_M = 10

# We restrict to descending passes only because mixing ascending and
# descending acquisitions causes radiometric inconsistencies that would
# add noise to your evaluation. Most Sen1Floods11 chips are also
# descending-pass only.
PASS_DIRECTION = "DESCENDING"


def main() -> None:
    # The first time you run this in a Python session you have to call
    # ee.Initialize(). This reads the credentials saved by
    # 'earthengine authenticate' and opens a session with Google's servers.
    print("Initializing Earth Engine session...")
    ee.Initialize()

    # Load the AOI polygon that 01_build_aoi.py wrote out. We convert it
    # to an Earth Engine FeatureCollection so EE can use it as a spatial
    # filter on the imagery query.
    aoi_path = Path(__file__).parent.parent / "data" / "aoi" / "bangladesh_2022_aoi.geojson"
    print(f"Loading AOI from {aoi_path}")
    import json
    with open(aoi_path) as f:
        aoi_geojson = json.load(f)
    aoi_fc = ee.FeatureCollection(aoi_geojson["features"])
    aoi_geom = aoi_fc.geometry()

    # Query Sentinel-1 for the pre-event window. The Earth Engine
    # collection 'COPERNICUS/S1_GRD' contains Ground Range Detected
    # products which are the standard input for flood mapping work.
    def fetch_window(start_date: str, end_date: str, label: str) -> ee.Image:
        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterDate(start_date, end_date)
            .filterBounds(aoi_geom)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("orbitProperties_pass", PASS_DIRECTION))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VV", "VH"])
        )
        n_images = collection.size().getInfo()
        print(f"  {label}: found {n_images} images between {start_date} and {end_date}")

        # Median compositing reduces speckle and smooths over inconsistencies
        # between adjacent acquisitions.
        composite = collection.median().clip(aoi_geom)
        return composite

    print("Querying Sentinel-1 for pre-event window...")
    pre_image = fetch_window(PRE_EVENT_START, PRE_EVENT_END, "PRE-EVENT")

    print("Querying Sentinel-1 for post-event window...")
    post_image = fetch_window(POST_EVENT_START, POST_EVENT_END, "POST-EVENT")

    # Submit export tasks. These run asynchronously on Google's servers.
    # The function returns immediately after submission. You then monitor
    # progress at https://code.earthengine.google.com/tasks
    print("Submitting export tasks to Earth Engine...")

    pre_task = ee.batch.Export.image.toDrive(
        image=pre_image,
        description="sylhet_pre_event",
        folder="bangladesh_sylhet_2022_s1",
        fileNamePrefix="sylhet_pre_event",
        region=aoi_geom,
        scale=EXPORT_SCALE_M,
        crs="EPSG:4326",
        maxPixels=1e10,
    )
    pre_task.start()
    print(f"  Pre-event export task ID: {pre_task.id}")

    post_task = ee.batch.Export.image.toDrive(
        image=post_image,
        description="sylhet_post_event",
        folder="bangladesh_sylhet_2022_s1",
        fileNamePrefix="sylhet_post_event",
        region=aoi_geom,
        scale=EXPORT_SCALE_M,
        crs="EPSG:4326",
        maxPixels=1e10,
    )
    post_task.start()
    print(f"  Post-event export task ID: {post_task.id}")

    print()
    print("Both export tasks submitted. They will run in the background on")
    print("Google's servers, typically completing in 30-90 minutes each.")
    print("Monitor progress at https://code.earthengine.google.com/tasks")
    print("When done, download the GeoTIFFs from your Google Drive folder")
    print("'bangladesh_sylhet_2022_s1' into ./data/raw/ on this machine.")


if __name__ == "__main__":
    main()
