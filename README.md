#### **Geospatial Patch Sampling with TorchGeo (DEM Example - Grand Canyon)**

This repository demonstrates a **reproducible, CRS-aware geospatial sampling workflow** using [TorchGeo](https://github.com/microsoft/torchgeo) and PyTorch.  
It shows how to tile a DEM into patches, validate those patches against a custom sampler, and load them into a standard PyTorch `Dataset`/`DataLoader` pipeline for spatial ML tasks such as segmentation.

The example uses a **USGS DEM subset of the Colorado River corridor (Grand Canyon)**, but the workflow generalises to any georeferenced raster dataset (DEMs, DSM/DTMs, UAV orthomosaics, rasterised point-cloud derivatives, etc.).

---

#### **Contents**

#### **`01_sampling_parity_grand_canyon.ipynb`**
Demonstrates that a **custom raster tiler** and TorchGeo’s `GridGeoSampler` can be configured for equivalent behaviour.

This notebook:

- Loads a USGS DEM tile  
- Defines a custom patch tiling function (size/stride)  
- Configures `GridGeoSampler` with matching parameters  
- Compares:
  - number of patches  
  - extents  
  - visual/structural parity  
- Produces plots confirming both samplers cover the same area with consistent patch geometry

**Key idea:** TorchGeo can serve as a clean, CRS-aware replacement for bespoke tiling logic.

---

#### **`03_patches_to_dataloader.ipynb`**  :contentReference[oaicite:0]{index=0}
Shows how patch locations from Notebook 01 connect to a minimal PyTorch `Dataset` and `DataLoader` to produce model-ready batches.

This notebook:
- Wraps DEM + slope-class target rasters into a custom `RasterPatchDataset`
- Reads patches on demand using bounding boxes from `GridGeoSampler`
- Normalises DEM values
- Produces `(X, y)` batches with shapes:
  - `X: [batch, 1, H, W]`
  - `y: [batch, H, W]`
- Includes a dummy model forward pass to confirm compatibility with U-Net-style segmentation architectures

**Key idea:**  
> **Raster → (geospatial sampler) → Dataset → DataLoader → model-ready batch**

---

#### **Data**

This repository **does not include** the DEM or label rasters, as they may be large and should be downloaded from the authoritative source.

#### **1. Download the DEM (USGS 3DEP)**

You can obtain freely available DEM data from the USGS National Map:

- USGS National Map Downloader:  
  **https://apps.nationalmap.gov/downloader/**

Search for an area of interest such as:

- *Colorado River, Grand Canyon National Park*, or  
- any region you wish to experiment with.

A 1-arcsecond (~30 m) DEM tile is sufficient for this example.

#### **2. Prepare the derived label raster (optional)**  

In the notebooks, the target layer is a **slope-class raster** derived from the DEM.  
You may create this using:

- QGIS  
- GDAL (`gdaldem slope`, classification tools)  
- WhiteboxTools  
- ArcGIS  

Alternatively, you can substitute any other categorical or continuous target raster.

#### **3. Update paths in the notebooks**

Modify:

```python
dem_path = "path/to/your/dem.tif"
slope_class_path = "path/to/your/labels.tif"
```
Note: you can prepare your slope raster using any method you prefer, or use the provided `prepare_data.py` script to generate slope and slope-class rasters from the DEM.

The notebooks are data-agnostic and will work with any georeferenced raster.

Environment & Dependencies

Requires:

- Python 3.9+
- PyTorch
- TorchGeo
- rasterio
- numpy, matplotlib, pandas, tqdm

Example install:

```python
pip install torch torchvision
pip install torchgeo rasterio numpy matplotlib pandas tqdm
```

Some systems may require setting the PROJ/GDAL data path (handled in the notebooks).

---
This repo demonstrates a clear separation between:

- Geospatial sampling (TorchGeo, CRS-aware)
- Patch extraction
- Dataset/DataLoader wiring
- Model-ready batch production

This pattern generalises across many spatial ML tasks, including:

- DEM segmentation
- Landslide susceptibility mapping
- UAV orthomosaic classification
- DSM/DTM segmentation
- Rasterised point-cloud feature learning
  
It illustrates practical, reproducible infrastructure for spatial machine learning workflows.

---
License
This project is released under the MIT License, allowing free use and adaptation with attribution.

---
Acknowledgements
DEM data courtesy of the USGS National Map (3DEP).
TorchGeo is developed by Microsoft and the PyTorch open-source community.

