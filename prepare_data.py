"""
Prepare Grand Canyon DEM data for portfolio project.

This script:
1. Reads a Grand Canyon DEM raster
2. Computes slope in degrees
3. Classifies slope into 5 categories (flat → very steep)
4. Writes output rasters for use in notebooks

Usage:
    python prepare_data.py

Requirements:
    - numpy
    - rasterio
    - Grand Canyon DEM as data/grand_canyon_dem.tif

Outputs:
    - data/grand_canyon_slope_deg.tif (slope in degrees)
    - data/grand_canyon_slope_classes.tif (categorical: 0=flat, 4=very steep)
"""
import os
import numpy as np
import rasterio
from rasterio.transform import Affine
from pathlib import Path


def compute_slope_and_classes(dem_path: Path, slope_path: Path, class_path: Path):
    """
    Compute slope (degrees) and slope classes from DEM.
    
    Args:
        dem_path: Path to input DEM raster
        slope_path: Path to output slope raster (degrees)
        class_path: Path to output slope class raster (categorical)
    """
    print(f"Reading DEM: {dem_path}")
    
    # Read DEM
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float32")
        dem_profile = src.profile
        transform: Affine = src.transform
        
        print(f"  Shape: {dem.shape}")
        print(f"  CRS: {src.crs}")
        print(f"  Resolution: {src.res}")
    
    # Handle no-data values
    nodata = dem_profile.get("nodata", None)
    if nodata is not None:
        mask = dem == nodata
        print(f"  No-data value: {nodata}")
    else:
        mask = np.isnan(dem)
    
    print(f"  No-data pixels: {mask.sum():,} ({100 * mask.sum() / mask.size:.1f}%)")
    
    # Compute slope
    print("\nComputing slope...")
    
    # Pixel size in map units
    xres = abs(transform.a)  # x resolution
    yres = abs(transform.e)  # y resolution
    
    print(f"  Pixel size: {xres:.2f} × {yres:.2f} m")
    
    # Gradients in x (east-west) and y (north-south)
    # np.gradient returns [dy, dx] for 2D arrays
    gy, gx = np.gradient(dem, yres, xres)
    
    # Slope in radians then degrees
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    slope_deg = np.degrees(slope_rad).astype("float32")
    
    # Respect mask
    slope_deg[mask] = np.nan
    
    print(f"  Slope range: {np.nanmin(slope_deg):.2f}° - {np.nanmax(slope_deg):.2f}°")
    print(f"  Mean slope: {np.nanmean(slope_deg):.2f}°")
    
    # Write slope raster
    print(f"\nWriting slope raster: {slope_path}")
    slope_profile = dem_profile.copy()
    slope_profile.update(dtype="float32", nodata=np.nan)
    
    with rasterio.open(slope_path, "w", **slope_profile) as dst:
        dst.write(slope_deg, 1)
    
    print("  ✓ Slope raster written")
    
    # Classify slope into bins
    print("\nClassifying slope into categories...")
    
    # Slope class bins (degrees)
    bins = [5, 15, 30, 45]  # Boundaries
    class_names = [
        "Flat (0-5°)",
        "Gentle (5-15°)",
        "Moderate (15-30°)",
        "Steep (30-45°)",
        "Very steep (45°+)"
    ]
    
    print("  Classes:")
    for i, name in enumerate(class_names):
        print(f"    {i}: {name}")
    
    # digitize returns: 0,1,2,3,4,5
    # 0 = < 5°, 1 = [5,15), 2 = [15,30), 3 = [30,45), 4 = 45+
    classes = np.digitize(slope_deg, bins, right=False).astype("uint8")
    
    # Set no-data pixels to 0
    classes[mask] = 0
    
    # Count pixels per class
    print("\n  Class distribution:")
    for i in range(len(class_names)):
        count = np.sum(classes == i)
        pct = 100 * count / (~mask).sum()  # Percentage of valid pixels
        print(f"    Class {i}: {count:,} pixels ({pct:.1f}%)")
    
    # Write slope class raster
    print(f"\nWriting slope class raster: {class_path}")
    class_profile = dem_profile.copy()
    class_profile.update(dtype="uint8", nodata=0)
    
    with rasterio.open(class_path, "w", **class_profile) as dst:
        dst.write(classes, 1)
    
    print("  ✓ Slope class raster written")


def main():
    """Main entry point."""
    # Paths
    data_dir = Path("data_out")
    dem_path = data_dir / "DEME_Zone3_2021_clip.tif"
    slope_path = data_dir / f'{os.path.basename(dem_path)[:-4]}_slope_deg.tif'
    class_path = data_dir / f'{os.path.basename(dem_path)[:-4]}_slope_classes.tif'
    
    # Check input exists
    if not dem_path.exists():
        print(f"ERROR: DEM not found at {dem_path}")
        print("\nPlease download a Grand Canyon DEM tile and save it as:")
        print(f"  {dem_path}")
        print("\nSuggested sources:")
        print("  - USGS 3DEP (https://apps.nationalmap.gov/downloader/)")
        print("  - USGS 2021 Colorado River DEM")
        return 1
    
    # Create output directory if needed
    data_dir.mkdir(exist_ok=True)
    
    # Process
    print("=" * 60)
    print("Grand Canyon DEM Processing")
    print("=" * 60)
    print()
    
    compute_slope_and_classes(dem_path, slope_path, class_path)
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print("\nOutput files:")
    print(f"  ✓ {slope_path}")
    print(f"  ✓ {class_path}")
    print("\nNow run the notebooks:")
    print("  - 01_sampling_parity_grand_canyon.ipynb")
    print("  - 02_torchgeo_capabilities_demo.ipynb")
    
    return 0


if __name__ == "__main__":
    exit(main())

"""
Example Output:

============================================================
Grand Canyon DEM Processing
============================================================

Reading DEM: data_out\DEME_Zone3_2021_clip.tif
  Shape: (4096, 4096)
  CRS: LOCAL_CS["NAD83(2011) / Arizona Central",UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]
  Resolution: (1.0, 1.0)
  No-data pixels: 0 (0.0%)

Computing slope...
  Pixel size: 1.00 × 1.00 m
  Slope range: 0.00° - 89.96°
  Mean slope: 8.42°

Writing slope raster: data_out\DEME_Zone3_2021_clip_slope_deg.tif      
  ✓ Slope raster written

Classifying slope into categories...
  Classes:
    0: Flat (0-5°)
    1: Gentle (5-15°)
    2: Moderate (15-30°)
    3: Steep (30-45°)
    4: Very steep (45°+)

  Class distribution:
    Class 0: 12,609,030 pixels (75.2%)
    Class 1: 859,568 pixels (5.1%)
    Class 2: 1,161,625 pixels (6.9%)
    Class 3: 1,288,147 pixels (7.7%)
    Class 4: 858,846 pixels (5.1%)

Writing slope class raster: data_out\DEME_Zone3_2021_clip_slope_classes.tif
  ✓ Slope class raster written

============================================================
COMPLETE
============================================================

Output files:
  ✓ data_out\DEME_Zone3_2021_clip_slope_deg.tif
  ✓ data_out\DEME_Zone3_2021_clip_slope_classes.tif

Now run the notebooks:
  - 01_sampling_parity_grand_canyon.ipynb
  - 02_torchgeo_capabilities_demo.ipynb
"""