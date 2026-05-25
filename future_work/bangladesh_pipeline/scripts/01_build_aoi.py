"""
Build the area-of-interest (AOI) polygon for the Bangladesh-2022-Sylhet test set.

This script reads the GADM 4.1 level-2 (district) administrative boundaries for
Bangladesh, selects the nine districts that the UN OCHA situation report and
the IFRC final report MDRBD028 identify as having been affected by the
June 2022 flood event, and merges them into a single multipolygon geometry
saved as GeoJSON.

The nine affected districts span three administrative divisions:
  Sylhet division     -> Sylhet, Sunamganj, Habiganj, Maulvibazar
  Mymensingh division -> Mymensingh, Sherpur, Netrakona
  Dhaka division      -> Kishoreganj
  Chittagong division -> Brahamanbaria

Note that several district names in GADM use transliterations that differ
from Bangladeshi government usage. We therefore key on the GADM spellings:
  Brahamanbaria (commonly Brahmanbaria)
  Maulvibazar   (commonly Moulvibazar)
  Netrakona     (commonly Netrokona)

Output: data/aoi/bangladesh_2022_aoi.geojson
"""

import json
from pathlib import Path

# Paths relative to repo root
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_GADM = PROJECT_ROOT / "data" / "raw" / "gadm41_BGD_2.json"
OUTPUT_AOI = PROJECT_ROOT / "data" / "aoi" / "bangladesh_2022_aoi.geojson"

# District names exactly as they appear in GADM 4.1 level-2 for Bangladesh
TARGET_DISTRICTS = {
    "Sylhet", "Sunamganj", "Habiganj", "Maulvibazar",     # Sylhet division
    "Mymensingh", "Sherpur", "Netrakona",                  # Mymensingh division
    "Kishoreganj",                                         # Dhaka division
    "Brahamanbaria",                                       # Chittagong division
}

def main() -> None:
    with open(INPUT_GADM) as f:
        gadm = json.load(f)

    # Filter to just the nine affected districts
    selected_features = []
    for feature in gadm["features"]:
        district_name = feature["properties"].get("NAME_2", "")
        if district_name in TARGET_DISTRICTS:
            # Keep the geometry but simplify the properties to what we need
            simplified = {
                "type": "Feature",
                "properties": {
                    "district": district_name,
                    "division": feature["properties"].get("NAME_1", ""),
                },
                "geometry": feature["geometry"],
            }
            selected_features.append(simplified)

    # Sanity check that we found all nine districts
    found_names = {f["properties"]["district"] for f in selected_features}
    missing = TARGET_DISTRICTS - found_names
    if missing:
        raise RuntimeError(f"Could not find these districts in GADM: {missing}")

    # Wrap in a feature collection so the output is a valid GeoJSON file
    output = {
        "type": "FeatureCollection",
        "name": "bangladesh_2022_sylhet_flood_aoi",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": selected_features,
    }

    OUTPUT_AOI.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_AOI, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(selected_features)} district features to {OUTPUT_AOI}")
    print("Districts included:")
    for feat in selected_features:
        p = feat["properties"]
        print(f"  - {p['district']:15s} ({p['division']} division)")


if __name__ == "__main__":
    main()
