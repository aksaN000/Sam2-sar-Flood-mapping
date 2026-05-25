# Pakistan-2022 (TU Wien) — deferred

Released by Roth et al. alongside their NHESS 2023 paper "Sentinel-1 based Analysis of the Severe Flood over Pakistan 2022".

- URL: https://researchdata.tuwien.at/records/zvvmh-nan78
- License: CC-BY 4.0
- Local path: `D:\datasets\pakistan-2022\` (911 MB, downloaded)
- Contents: `FLOOD-HM-MASKED.zip` (164 flood-extent rasters, Equi7Grid AS020M, EPSG:27703), `flood_frequency.tif`, `first_flood_detection.tif`.

## Why this is deferred

The TU Wien release contains only flood-extent label rasters. The matching Sentinel-1 GRD imagery used to derive them is not redistributed; it has to be acquired separately from the Copernicus Data Space ecosystem (free, no auth gate). Until those S1 scenes are downloaded and tiled to 512×512 chips matching the Sen1Floods11 layout, the masks alone are not usable for evaluating a SAR flood model.

## To activate

1. Identify the date and Equi7Grid cell of each mask in `FLOOD-HM-MASKED/`.
2. Pull matching S1 GRD scenes from Copernicus DataSpace for each (date, footprint).
3. Reproject the S1 scenes to the EPSG:27703 grid that the masks use, or vice versa.
4. Tile both into 512×512 chips, drop chips with too much nodata.
5. Add a `Pakistan2022Dataset` class under `model/datasets/` mirroring `Sen1Floods11Dataset`.

Estimated effort: 3-5 hours of code + several hours of download time. Not in scope for the current thesis.
