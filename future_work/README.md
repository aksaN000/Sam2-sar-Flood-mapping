# Future Work — Deferred Artifacts

This directory contains two artifacts that the thesis describes and ships, but does not empirically evaluate. They are kept here so the principal `model/` tree stays focused on the Sen1Floods11 evaluation surface.

## `bangladesh_pipeline/`

The complete Bangladesh-2022-Sylhet dataset construction pipeline, originally Phase 1 of the thesis project. Contains:

- `scripts/01_build_aoi.py` through `08_quality_check.py` — the eight-stage pipeline that builds 512×512 Sen1Floods11-compatible chips for the June 2022 Sylhet flood event.
- `scripts/utils/gee_auth.py`, `viz.py` — shared helpers.
- `data/aoi/bangladesh_2022_aoi.geojson` — pre-built 9-district area-of-interest polygon.
- `data/raw/gadm41_BGD_2.json` — GADM 4.1 admin boundaries.
- `docs/data_sources.md`, `licensing.md`, `troubleshooting.md` — pipeline documentation.
- `notebooks/inspect_chips.ipynb` — visual QA notebook.
- `bangladesh_sylhet.py` — the deferred PyTorch Dataset class. To activate it, move back to `../../model/datasets/` and re-add the export to `__init__.py`.
- `requirements.txt` — the geospatial dependency stack (rasterio, geopandas, earthengine-api, etc.) needed only by the pipeline; the main `model/` does not need any of these.

To run the pipeline:

```bash
conda activate sam2-sar          # or any Python 3.11 env
pip install -r future_work/bangladesh_pipeline/requirements.txt
cd future_work/bangladesh_pipeline
python scripts/01_build_aoi.py   # ... through 08_quality_check.py
```

Running the pipeline requires Earth Engine authentication, manual HDX downloads, and ~3 hours of wall-clock time. See `docs/` for the full procedure.

## `pakistan_2022/`

The Pakistan-2022 SAR flood label rasters from TU Wien (Roth et al. 2023). The dataset itself sits on disk at `D:\datasets\pakistan-2022\` (911 MB). This subdirectory only contains a README pointing at it because:

- The TU Wien release ships only flood-extent masks on the Equi7Grid coordinate system.
- The matching Sentinel-1 GRD imagery is not redistributed and must be pulled from the Copernicus Data Space ecosystem for the actual evaluation.
- The thesis cites this dataset but defers empirical evaluation to subsequent work for the reason above.
