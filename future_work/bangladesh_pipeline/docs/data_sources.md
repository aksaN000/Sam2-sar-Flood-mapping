# Data Sources

This document describes every external dataset used by the construction pipeline, along with its access URL, license, and recommended citation.

## Sentinel-1 GRD imagery (radar input)

The Sentinel-1 mission is operated by the European Space Agency under the Copernicus programme and provides C-band Synthetic Aperture Radar (SAR) imagery globally on a 6-day revisit cycle. The pipeline pulls IW-mode descending-pass GRD products through Google Earth Engine collection `COPERNICUS/S1_GRD` for two windows: 1 to 10 June 2022 (pre-event) and 17 to 25 June 2022 (post-event). The native ground resolution is approximately 20 m and the pixel spacing is 10 m, so we export at 10 m. Sentinel-1 imagery is released under the Copernicus open and free data policy, which permits unrestricted use including redistribution. Citation: European Space Agency, Copernicus Sentinel-1 GRD, accessed via Google Earth Engine.

## GADM 4.1 administrative boundaries (AOI definition)

The Database of Global Administrative Areas (GADM) version 4.1 supplies the polygon for each Bangladesh district. The pipeline reads the level-2 (district-level) JSON file `gadm41_BGD_2.json`, which ships in `data/raw/`. The GADM license permits free academic use; commercial redistribution requires permission. Source URL: https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BGD_2.json.zip. Citation: Hijmans, R., Database of Global Administrative Areas (GADM) version 4.1, 2022.

## JRC Global Surface Water v1.4 (permanent-water mask)

The Joint Research Centre of the European Commission, in collaboration with Google, produced a global record of how often each 30 m pixel was covered by water between 1984 and 2021. The pipeline uses the seasonality layer thresholded at 5 months per year to mask out pixels that are normally inundated, ensuring that flood labels reflect only flood-induced inundation rather than baseline river extent. Asset ID on Earth Engine: `JRC/GSW1_4/GlobalSurfaceWater`. License: Creative Commons Attribution 4.0. Citation: Pekel, J.-F., Cottam, A., Gorelick, N., Belward, A. S., High-resolution mapping of global surface water and its long-term changes, Nature 540, 418 to 422, 2016.

## UNOSAT FL20220525BGD (primary flood polygons)

The United Nations Satellite Centre processed the May to June 2022 Bangladesh flood event under activation FL20220525BGD and published six dated flood-extent products spanning 25 May to 21 June 2022. Three observations cover the post-event window used by this pipeline (18, 19, and 21 June) and three additional observations cover the pre-event and onset phases (25, 26, and 28 May). All six are distributed as shapefiles through the Humanitarian Data Exchange at data.humdata.org. License: HumData terms of use with attribution to UNOSAT and UNITAR. Citation: UNOSAT, Water Extent over Bangladesh as part of activation FL20220525BGD, 2022.

## Copernicus EMS Global Flood Monitoring (secondary flood rasters)

The Copernicus Emergency Management Service Global Flood Monitoring (GFM) service publishes continuous Sentinel-1-based observations of all floods globally. The May to June 2022 Bangladesh flood is covered by the news item at https://global-flood.emergency.copernicus.eu/news/102-floods-in-bangladesh-may-2022/. The data are released under the standard Copernicus open data policy. Citation: Copernicus Emergency Management Service Global Flood Monitoring, 2022.

## International Charter Activation 762 (tertiary flood polygons, optional)

The International Charter Space and Major Disasters was activated on 18 June 2022 at the request of UNITAR on behalf of UN OCHA Asia-Pacific. The activation page lists delivered shapefile products. Activation URL: https://disasterscharter.org/web/guest/activations/-/article/flood-large-in-bangladesh-activation-762-. The Charter products are useful for cross-validation but are optional; the pipeline runs correctly with UNOSAT alone. License: per the Charter's standard terms for the relevant product author.
