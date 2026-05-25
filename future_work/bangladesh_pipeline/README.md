## Bangladesh-2022-Sylhet SAR Flood Test Set

This package builds a custom out-of-distribution test set for evaluating Synthetic Aperture Radar flood-mapping models on the June 2022 monsoon flood event in northeastern Bangladesh. The output is a set of 512 by 512 pixel chips that match the Sen1Floods11 directory layout, which means they are plug-compatible with any model trained on Sen1Floods11.

### What you get when you finish

When the pipeline runs to completion, the directory `data/final/` contains roughly one hundred and fifty to two hundred chips covering the nine districts identified by UN OCHA and IFRC as affected by the event, namely Sylhet, Sunamganj, Habiganj and Maulvibazar in Sylhet division, Mymensingh, Sherpur and Netrakona in Mymensingh division, Kishoreganj in Dhaka division, and Brahamanbaria in Chittagong division. Each chip is represented by three GeoTIFF files: a pre-event Sentinel-1 image taken between 1 and 10 June 2022, a post-event Sentinel-1 image taken between 17 and 25 June 2022 covering the peak of the flood, and a binary flood label raster. The chips contain dual-polarization VV and VH channels in decibel scale, with permanent water bodies masked out so that the labels reflect only flood-induced inundation rather than baseline river extent.

### Sources of input data

The flood polygons used to produce the labels come from three independent expert sources, in order of priority. The primary source is the UNOSAT activation FL20220525BGD, which is published through the Humanitarian Data Exchange at data.humdata.org and which provides ready-to-use shapefiles for six dated observations spanning 25 May to 21 June 2022. The secondary source is the Copernicus Emergency Management Service Global Flood Monitoring service, which publishes continuous Sentinel-1-based observations of all floods globally. The tertiary source is the International Charter Space and Major Disasters Activation 762, which was activated on 18 June 2022 at the request of UNITAR on behalf of UN OCHA Asia-Pacific. The pipeline gracefully handles whatever subset of these three sources you actually obtain, so the easiest workflow is to start with UNOSAT alone and add the other two only if you want extra cross-validation.

The Sentinel-1 imagery itself comes from the European Space Agency under the Copernicus open and free data policy, accessed through Google Earth Engine. The administrative boundaries used to define the area of interest come from the GADM 4.1 dataset. Permanent water bodies are masked out using the Joint Research Centre Global Surface Water dataset version 1.4 with a seasonality threshold of five months per year.

### Hardware and time budget

The pipeline runs comfortably on any laptop with at least eight gigabytes of RAM and ten gigabytes of free disk space. The actual compute cost on your machine is small because the heavy lifting happens on Google Earth Engine's servers. End-to-end wall-clock time is roughly three to six hours, dominated by Earth Engine export queue times rather than active work on your part.

### Setup once

Install Python 3.11, create a virtual environment, and install the pinned dependencies. There are two requirements files: `requirements.txt` covers the dataset construction pipeline (the eight `scripts/0*.py` files), and `requirements-train.txt` covers the modelling stack (PyTorch, Lightning, HuggingFace Transformers, PEFT, W&B). Both files install cleanly into the same Python 3.11 virtualenv. From the project root run `pip install -r requirements.txt` if you only need to build the test set, or `pip install -r requirements.txt -r requirements-train.txt` if you also want to train the SAM 2 adapters under `model/`. After dependencies install, run `earthengine authenticate` to establish your Earth Engine credentials, which opens a browser window where you sign in with the Google account that has Earth Engine access. If you do not already have Earth Engine access, sign up for free at `https://earthengine.google.com/signup/`; academic approval is usually instant.

### The pipeline, step by step

Run the eight scripts in numerical order. Each one reads from `data/raw/` or `data/processed/` and writes to the next directory along the chain.

The first script, `01_build_aoi.py`, reads a copy of the GADM 4.1 administrative boundary file that ships in the `data/raw/` directory and assembles the area-of-interest polygon by selecting the nine flood-affected districts. This script runs locally in about one second and produces `data/aoi/bangladesh_2022_aoi.geojson`. The pre-built AOI is included in the package so you do not need to run this script unless you want to verify the construction.

The second script, `02_export_sentinel1.py`, asks Earth Engine to retrieve all Sentinel-1 GRD images covering the AOI in the two relevant time windows, computes a median composite for each window to suppress per-acquisition noise, and submits two export tasks that write the results to your Google Drive. The script returns immediately after submitting the tasks; the actual export typically completes in thirty to ninety minutes for each task. You monitor progress at `https://code.earthengine.google.com/tasks` and download the resulting GeoTIFFs from your Google Drive folder `bangladesh_sylhet_2022_s1` into the local `data/raw/` directory once they are ready.

The third script, `03_export_permanent_water.py`, performs a similar export for the JRC Global Surface Water permanent-water mask. This is a smaller export that usually completes in under fifteen minutes. Save the result alongside the Sentinel-1 GeoTIFFs in `data/raw/`.

The fourth script, `04_fetch_reference_labels.py`, prints clear manual download instructions for the UNOSAT FL20220525BGD shapefiles on HumData and for the optional GFM and Charter products. UNOSAT is the only source you strictly need; the others are useful for cross-validation but the pipeline runs fine with UNOSAT alone. Each HumData dataset page lists the shapefile resource at the bottom and downloads as a single zip file. The whole download takes about fifteen minutes.

The fifth script, `05_rasterize_labels.py`, reads the polygon shapefiles and any GFM rasters you downloaded, projects them into the same coordinate system as the Sentinel-1 imagery, burns them onto an exact-aligned raster grid, unions all sources through a logical OR operation, subtracts the permanent water mask, and writes the final binary flood-label raster to `data/processed/flood_label.tif`. This script runs in about one minute.

The sixth script, `06_apply_speckle_filter.py`, applies the Refined Lee speckle filter to the raw Sentinel-1 imagery and converts the filtered output from linear power values to decibels. This is the radar equivalent of running noise reduction on a grainy photograph. The script runs in about two minutes per image.

The seventh script, `07_tile_to_chips.py`, walks a regular grid of starting positions across the filtered images and the label raster, extracts a 512 by 512 pixel window at each position, discards any window that has too much nodata, and writes the surviving windows as separate GeoTIFF files in the Sen1Floods11 directory layout. It also writes an index CSV that records which chip corresponds to which spatial location and what fraction of each chip is flooded.

The eighth script, `08_quality_check.py`, picks a random sample of twenty chips and renders quad-panel PNG figures that show the pre-event imagery, post-event imagery, post-minus-pre difference, and the ground-truth flood label side by side. You scan through these figures by eye to confirm that the labels are correctly aligned to the obvious dark patches in the post-event imagery and to flag any chips with obvious errors.

### Licensing and attribution

If you publish your final test set, you must attribute the underlying data sources. The Sentinel-1 imagery is provided under the Copernicus open and free data policy. The JRC Global Surface Water dataset is provided under Creative Commons Attribution 4.0. UNOSAT shapefiles distributed through HumData are released under HumData's data sharing terms with attribution to UNOSAT and UNITAR. The Copernicus Emergency Management Service Global Flood Monitoring observations follow the standard Copernicus open data policy. The GADM administrative boundaries are freely distributable for academic use under the GADM license.

The construction scripts in this repository are released under the MIT License; see the `LICENSE` file at the project root for the full text. A more thorough discussion of the licensing chain across all source datasets lives in `docs/licensing.md`. The recommended approach for redistributing the dataset is to release the construction scripts publicly and let downstream users assemble the data themselves from the original sources, which sidesteps any ambiguity about whether you have the right to redistribute the underlying labels.

### Troubleshooting

If the GADM download fails, you can fetch the file manually from `https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BGD_2.json.zip` and place it in `data/raw/` yourself. If the Earth Engine export takes more than two hours, check the task status page; sometimes the queue is congested and you can resubmit with a smaller AOI. If a UNOSAT shapefile fails to load in script five, it is usually because the .prj file did not download cleanly, in which case re-downloading the zip file and unzipping again typically resolves it. If the speckle filter takes much longer than two minutes per image, your machine is probably swapping to disk because the AOI is large; reduce the AOI to a single division for testing before running the full version.
