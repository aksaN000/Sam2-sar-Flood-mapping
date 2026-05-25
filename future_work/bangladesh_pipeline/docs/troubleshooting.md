# Troubleshooting

Common failures encountered while running the dataset construction pipeline, in roughly the order in which a fresh user is likely to hit them.

## GADM download fails or times out

The GADM redirect occasionally returns a 503 or hangs for several minutes. Fetch the file manually from `https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BGD_2.json.zip` and place the unzipped JSON in `data/raw/gadm41_BGD_2.json` yourself. Verify with `python -c "import json; print(len(json.load(open('data/raw/gadm41_BGD_2.json'))['features']))"`, which should print a number greater than 60 (Bangladesh has 64 districts at level 2).

## Earth Engine authentication fails

If `earthengine authenticate` opens the browser but the resulting `ee.Initialize()` raises an authentication error, two issues are common. First, the Google account you signed in with may not yet have Earth Engine access. Sign up for free at https://earthengine.google.com/signup/ and wait a few minutes for academic approval. Second, the credentials file may have been written to the wrong directory if you ran the command inside a restricted shell. Re-run from a normal user shell and confirm that `~/.config/earthengine/credentials` (Linux/macOS) or `%USERPROFILE%\.config\earthengine\credentials` (Windows) exists.

## Earth Engine export takes longer than two hours

The export queue at Earth Engine is shared across all users globally and occasionally backs up. Check https://code.earthengine.google.com/tasks. If a task has been queued for more than two hours, cancel and resubmit with a smaller AOI by editing `01_build_aoi.py` to keep only Sylhet division (four districts), regenerating `data/aoi/bangladesh_2022_aoi.geojson`, and re-running scripts 02 and 03. The smaller export typically completes in 15 to 30 minutes.

## UNOSAT shapefile loads but contains no geometry

If `05_rasterize_labels.py` reports zero flooded pixels for a UNOSAT source that you definitely downloaded, the most common cause is a missing `.prj` file. The HumData zip occasionally drops the projection file, in which case `geopandas.read_file` opens the shapefile but cannot reproject it. Re-download the zip and unzip again; if the `.prj` is still missing, copy the matching one from another UNOSAT product in the same activation, since they all share the same coordinate system.

## Permanent-water mask download missing or wrong shape

`05_rasterize_labels.py` reprojects the permanent-water mask onto the Sentinel-1 grid, so it does not strictly require an exact shape match, but a missing file will cause the script to exit. Confirm `data/raw/permanent_water_mask.tif` exists and has shape consistent with the AOI bounding box. If the export from script 03 is partly missing nodata values along the edges, that is normal and harmless; the reprojection step pads out-of-bounds pixels as zero.

## Refined Lee filter takes much longer than two minutes per image

The pure-NumPy Refined Lee implementation in `06_apply_speckle_filter.py` runs comfortably on any laptop with at least eight gigabytes of RAM. If it takes much longer, your machine is probably swapping to disk because the AOI is large. Reduce the AOI to one division as described above and re-run the pipeline end to end.

## Chip count is below the expected 150 to 200

`07_tile_to_chips.py` discards any chip with more than five percent nodata pixels. If you are getting fewer chips than expected, inspect `data/raw/sylhet_post_event.tif` in QGIS and look for large nodata regions. The most common cause is that one of the Earth Engine exports clipped the AOI more aggressively than expected because of a `maxPixels` overflow. Resubmit the export with a larger `maxPixels` value (the script already uses 1e10) or split the AOI in half and merge the two exports before tiling.

## QA panels show flood label misaligned with dark patches

If a meaningful fraction of the QA panels show the flood label shifted by several pixels relative to the obvious dark regions in the post-event imagery, the cause is almost always a coordinate-reference-system mismatch. Confirm that all of `sylhet_post_event.tif`, `permanent_water_mask.tif`, and the UNOSAT shapefiles share the same CRS after `05_rasterize_labels.py` runs. The pipeline reprojects everything to the post-event raster's CRS, so any mismatch indicates that the post-event raster itself was exported in a non-standard projection. Re-run script 02 with `crs="EPSG:4326"` explicitly set if you previously edited it.

## Disk fills up before the pipeline finishes

The intermediate files in `data/raw/` and `data/processed/` together can reach about 6 to 8 gigabytes. If you are tight on disk, delete `data/processed/sylhet_pre_event_filtered_db.tif` and `data/processed/sylhet_post_event_filtered_db.tif` after script 07 finishes; they are no longer needed once the chips are written.
