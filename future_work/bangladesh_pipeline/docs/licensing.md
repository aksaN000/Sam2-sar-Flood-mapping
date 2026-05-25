# Licensing Chain

The Bangladesh-2022-Sylhet test set is assembled from open data released by several different providers, each with its own license. The construction scripts in this repository have a single uniform license, but the data they consume and produce inherit the constraints of the underlying sources. This document explains the chain so that downstream users can decide what they may redistribute and under what terms.

## Construction scripts (this repository)

All Python code under `scripts/`, `model/`, and `notebooks/`, the `LICENSE` file at the project root, and the `requirements*.txt` files are released under the MIT License. See `LICENSE` for the full text. In practice, this means anyone may use, modify, and redistribute the scripts for any purpose subject only to retaining the copyright notice. The MIT License covers only the scripts, not the data they download or generate.

## Sentinel-1 imagery

Sentinel-1 GRD products are released by the European Space Agency under the Copernicus open and free data policy. Redistribution is permitted as long as Copernicus and the European Space Agency are credited. There is no copyleft clause and no commercial-use restriction.

## GADM administrative boundaries

GADM 4.1 is freely distributable for academic use under the GADM license. The license restricts commercial redistribution, so a user who builds a commercial product on this dataset must contact GADM or substitute equivalent open boundaries (for example, OpenStreetMap or Bangladesh Bureau of Statistics shapefiles).

## JRC Global Surface Water v1.4

The JRC Global Surface Water dataset is released under Creative Commons Attribution 4.0 (CC-BY 4.0). Redistribution is permitted with attribution to the Joint Research Centre and citation of the corresponding Nature paper.

## UNOSAT FL20220525BGD shapefiles

UNOSAT shapefiles distributed through HumData are released under HumData's data sharing terms with attribution to UNOSAT and UNITAR. The terms permit academic and humanitarian use; commercial redistribution requires consultation with UNITAR.

## Copernicus EMS Global Flood Monitoring rasters

The GFM rasters follow the standard Copernicus open data policy, identical in practice to the Sentinel-1 imagery licensing.

## International Charter Activation 762 products

Charter products are subject to the Charter's standard terms for the relevant product author. Many Charter deliverables permit academic use and require attribution; some impose stricter constraints. Inspect the metadata of any Charter shapefile before redistributing.

## Recommended distribution strategy

Because the underlying flood polygons (UNOSAT and Charter) carry stricter terms than the Sentinel-1 imagery, the cleanest way to share the assembled test set is to release the construction scripts publicly and let downstream users assemble the data themselves from the original sources. This sidesteps any ambiguity about whether you have the right to redistribute the underlying labels. If you do redistribute the assembled chips, retain the attribution lines from each source dataset in any accompanying README and avoid stripping metadata from the GeoTIFF files.
