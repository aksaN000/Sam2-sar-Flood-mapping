"""
Fetch reference flood polygons for the June 2022 Bangladesh flood event.

This script collects expert-validated flood-extent polygons from up to
three independent sources, in order of priority. The primary source is
UNOSAT FL20220525BGD distributed through the Humanitarian Data Exchange,
which is the easiest to access and the most temporally comprehensive.
The secondary source is the Copernicus EMS Global Flood Monitoring
service. The tertiary source is the International Charter Activation 762.
You only strictly need the primary source for the test set to work; the
secondary and tertiary are useful for cross-validation and for filling
gaps in coverage.

PRIMARY SOURCE: UNOSAT FL20220525BGD on HumData
-----------------------------------------------
UNOSAT, the United Nations Satellite Centre, processed the event under
activation code FL20220525BGD and published six dated flood-extent
products spanning 25 May to 21 June 2022. Three observations cover the
post-event window used by this pipeline (the primary downloads), and
three additional observations cover the pre-event and onset phases of
the flood (optional, useful for pre-event reference or cross-validation).

Primary downloads for the post-event window (17 to 25 June 2022):

  19 June 2022 product (Radarsat Constellation Mission acquisition):
    Covers 22,300 km^2 across all four affected divisions; 9,500 km^2
    flooded. This is the principal post-event reference because it
    captures the peak of the second flood wave and matches our AOI
    extent closely.
    URL: https://data.humdata.org/dataset/water-extent-over-sylhet-mymensingh-dhaka-and-chattogram-divisions-bangladesh-as-of-19-jun

  21 June 2022 product (Sentinel-1 acquisition):
    Covers 20,400 km^2; 2,950 km^2 flooded. This captures the receding
    phase of the flood and is useful for cross-validation of the peak
    observation against a different sensor.
    URL: https://data.humdata.org/dataset/water-extent-in-rajshahi-rangur-mymensingh-dhaka-and-khulna-divisions-bangladesh-as-of-21

  18 June 2022 product (RCM-1 acquisition):
    Covers 1,325 km^2 with high spatial detail in Sylhet and Sunamganj;
    840 km^2 flooded.
    URL: https://data.humdata.org/dataset/water-extent-and-impact-over-sylhet-and-sunamganj-districts-sylhet-division-bangladesh-as-

Optional pre-event and onset observations (useful for cross-validation):

  25 May 2022 product (Chaohu-1 acquisition):
    Covers 730 km^2 in Sylhet and Sunamganj districts only; 420 km^2
    flooded. Captures the very first onset of the May flood wave.
    URL: https://data.humdata.org/dataset/water-extent-over-sylhet-and-sunamganj-districts-bangladesh-as-of-25-may-2022

  26 May 2022 product (Sentinel-1 acquisition):
    Covers 16,000 km^2 across the four affected divisions; 6,800 km^2
    flooded. Captures the peak of the first (May) flood wave.
    URL: https://data.humdata.org/dataset/water-extent-over-sylhet-mymensingh-dhaka-and-chattogram-divisions-bangladesh-as-of-26-may

  28 May 2022 product (Sentinel-1 acquisition):
    Covers 12,000 km^2; 4,500 km^2 flooded. Captures the receding phase
    of the first wave between the two flood peaks.
    URL: https://data.humdata.org/dataset/water-extent-over-sylhet-mymensingh-dhaka-and-chattogram-divisions-bangladesh-as-of-28-may

How to download from HDX
------------------------
Each HDX dataset page lists the available resources at the bottom. For
each of the URLs above, find the resource that ends in .zip or has the
'Geodatabase' or 'Shapefile' format tag, click 'Download', and save the
zip file to data/raw/unosat_FL20220525BGD/. After all downloads are done,
unzip them all in place.

SECONDARY SOURCE: Copernicus GFM
--------------------------------
Open https://global-flood.emergency.copernicus.eu/news/102-floods-in-bangladesh-may-2022/
and follow the data download links to the GFM observed flood extent
products covering the same dates. Save these to data/raw/gfm_ofe/.

TERTIARY SOURCE (OPTIONAL): International Charter Activation 762
-----------------------------------------------------------------
Open https://disasterscharter.org/web/guest/activations/-/article/flood-large-in-bangladesh-activation-762-
and download any delivered shapefile products from the 'Delivered
Products' section. Save these to data/raw/charter_762/. Skip this step
if you have already obtained the UNOSAT and GFM polygons and want to
proceed; the Charter polygons are useful for cross-validation but not
strictly required.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

UNOSAT_DIR = RAW_DIR / "unosat_FL20220525BGD"
GFM_DIR = RAW_DIR / "gfm_ofe"
CHARTER_DIR = RAW_DIR / "charter_762"

UNOSAT_DATASETS = [
    # Primary downloads for the post-event window (17 to 25 June 2022)
    {
        "label": "19 Jun 2022 (RCM-1, all 4 divisions, peak)",
        "url": "https://data.humdata.org/dataset/water-extent-over-sylhet-mymensingh-dhaka-and-chattogram-divisions-bangladesh-as-of-19-jun",
        "covers_km2": 22300,
        "flooded_km2": 9500,
        "priority": "primary download for post-event window",
    },
    {
        "label": "21 Jun 2022 (Sentinel-1, 5 divisions, receding)",
        "url": "https://data.humdata.org/dataset/water-extent-in-rajshahi-rangur-mymensingh-dhaka-and-khulna-divisions-bangladesh-as-of-21",
        "covers_km2": 20400,
        "flooded_km2": 2950,
        "priority": "primary download for post-event window",
    },
    {
        "label": "18 Jun 2022 (RCM-1, Sylhet+Sunamganj high detail)",
        "url": "https://data.humdata.org/dataset/water-extent-and-impact-over-sylhet-and-sunamganj-districts-sylhet-division-bangladesh-as-",
        "covers_km2": 1325,
        "flooded_km2": 840,
        "priority": "primary download for post-event window",
    },
    # Optional pre-event and onset observations (useful for pre-event reference or cross-validation)
    {
        "label": "25 May 2022 (Chaohu-1, Sylhet+Sunamganj only, onset)",
        "url": "https://data.humdata.org/dataset/water-extent-over-sylhet-and-sunamganj-districts-bangladesh-as-of-25-may-2022",
        "covers_km2": 730,
        "flooded_km2": 420,
        "priority": "optional, useful for pre-event reference or cross-validation",
    },
    {
        "label": "26 May 2022 (Sentinel-1, all 4 divisions, first peak)",
        "url": "https://data.humdata.org/dataset/water-extent-over-sylhet-mymensingh-dhaka-and-chattogram-divisions-bangladesh-as-of-26-may",
        "covers_km2": 16000,
        "flooded_km2": 6800,
        "priority": "optional, useful for pre-event reference or cross-validation",
    },
    {
        "label": "28 May 2022 (Sentinel-1, receding first wave)",
        "url": "https://data.humdata.org/dataset/water-extent-over-sylhet-mymensingh-dhaka-and-chattogram-divisions-bangladesh-as-of-28-may",
        "covers_km2": 12000,
        "flooded_km2": 4500,
        "priority": "optional, useful for pre-event reference or cross-validation",
    },
]


def print_unosat_instructions():
    print("=" * 78)
    print("PRIMARY SOURCE: UNOSAT FL20220525BGD on HumData")
    print("=" * 78)
    print()
    print(f"Save all downloads under: {UNOSAT_DIR}")
    print()
    print("Six dated observations are available, spanning 25 May to 21 June 2022.")
    print("Three are primary downloads for the post-event window (17 to 25 June);")
    print("the other three are optional pre-event and onset observations useful")
    print("for cross-validation or building a pre-event reference.")
    print()
    for i, ds in enumerate(UNOSAT_DATASETS, start=1):
        print(f"  {i}. {ds['label']}")
        print(f"     Priority: {ds['priority']}")
        print(f"     URL: {ds['url']}")
        print(f"     Stats: {ds['covers_km2']:,} km^2 analyzed, {ds['flooded_km2']:,} km^2 flooded")
        print()
    print("On each HDX dataset page:")
    print("  a. Scroll to the 'Resources' section near the bottom.")
    print("  b. Click the resource with format 'SHP' or 'Geodatabase'.")
    print("  c. The download is a zip file containing .shp/.shx/.dbf/.prj.")
    print("  d. Save it under the directory above and unzip in place.")
    print()


def print_gfm_instructions():
    print("=" * 78)
    print("SECONDARY SOURCE: Copernicus GFM Observed Flood Extent")
    print("=" * 78)
    print()
    print(f"Save all downloads under: {GFM_DIR}")
    print()
    print("1. Open the GFM news item for the May 2022 Bangladesh floods:")
    print("   https://global-flood.emergency.copernicus.eu/news/102-floods-in-bangladesh-may-2022/")
    print("2. Follow the data download links provided in the article body.")
    print("3. If the article only links to the live dashboard, instead use the")
    print("   GFM main map at https://global-flood.emergency.copernicus.eu/")
    print("   and use the 'Download data' tool with date filter 17-25 June 2022.")
    print()


def print_charter_instructions():
    print("=" * 78)
    print("TERTIARY SOURCE (OPTIONAL): International Charter Activation 762")
    print("=" * 78)
    print()
    print(f"Save all downloads under: {CHARTER_DIR}")
    print()
    print("1. Open the activation page:")
    print("   https://disasterscharter.org/web/guest/activations/-/article/flood-large-in-bangladesh-activation-762-")
    print("2. Scroll to 'Delivered Products' and download any .zip products that")
    print("   contain shapefiles. Skip products that are only PDF reference maps.")
    print("3. Unzip in place inside the directory above.")
    print()
    print("If access is blocked or no shapefile products are available, you can")
    print("safely skip this source. The UNOSAT data alone is sufficient.")
    print()


def main():
    UNOSAT_DIR.mkdir(parents=True, exist_ok=True)
    GFM_DIR.mkdir(parents=True, exist_ok=True)
    CHARTER_DIR.mkdir(parents=True, exist_ok=True)

    print_unosat_instructions()
    print_gfm_instructions()
    print_charter_instructions()

    print("=" * 78)
    print("STATUS: Once you have downloaded UNOSAT FL20220525BGD shapefiles")
    print("        (and optionally GFM and Charter), proceed to script 05")
    print("        which will rasterize them onto the Sentinel-1 grid.")
    print("=" * 78)


if __name__ == "__main__":
    main()
