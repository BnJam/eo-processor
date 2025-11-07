# Quick Start Guide

Get started with eo-processor in 5 minutes!

## Installation

### Prerequisites

- Python 3.8+
- Rust toolchain from [rustup.rs](https://rustup.rs/)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/BnJam/eo-processor.git
cd eo-processor

# Install maturin
pip install maturin

# Build and install
maturin develop --release
```

## Basic Usage

### 1. Compute NDVI (Vegetation Index)

```python
import numpy as np
from eo_processor import ndvi

# Sample satellite data
nir = np.array([0.8, 0.7, 0.6, 0.5])  # Near-infrared band
red = np.array([0.2, 0.1, 0.3, 0.2])  # Red band

# Compute NDVI
ndvi_values = ndvi(nir, red)
print(ndvi_values)
# Output: [0.6, 0.75, 0.33333333, 0.42857143]
```

### 2. Process Image Data

```python
import numpy as np
from eo_processor import ndvi

# Load satellite image (example: 1000x1000 pixels)
nir_image = np.random.rand(1000, 1000) * 0.8 + 0.2
red_image = np.random.rand(1000, 1000) * 0.4

# Compute NDVI for entire image
ndvi_image = ndvi(nir_image, red_image)

print(f"NDVI range: {ndvi_image.min():.3f} to {ndvi_image.max():.3f}")
print(f"Mean NDVI: {ndvi_image.mean():.3f}")
```

### 3. Water Detection (NDWI)

```python
import numpy as np
from eo_processor import ndwi

green = np.array([0.3, 0.4, 0.5, 0.6])
nir = np.array([0.2, 0.1, 0.3, 0.2])

water_index = ndwi(green, nir)
print(water_index)
# Output: [0.2, 0.6, 0.25, 0.5]

# Threshold for water detection
is_water = water_index > 0.3
print(f"Water pixels: {is_water.sum()}")
```

## Integration with XArray

Perfect for geospatial data analysis!

```python
import numpy as np
import xarray as xr
from eo_processor import ndvi

# Create XArray DataArrays with coordinates
nir = xr.DataArray(
    np.random.rand(100, 100),
    dims=["y", "x"],
    coords={
        "y": np.arange(100),
        "x": np.arange(100),
    },
    attrs={"band": "NIR", "units": "reflectance"}
)

red = xr.DataArray(
    np.random.rand(100, 100),
    dims=["y", "x"],
    coords={
        "y": np.arange(100),
        "x": np.arange(100),
    },
    attrs={"band": "Red", "units": "reflectance"}
)

# Compute NDVI using apply_ufunc
ndvi_result = xr.apply_ufunc(
    ndvi,
    nir,
    red,
    dask="parallelized",  # Enable parallel processing
    output_dtypes=[float],
)

# The result is an XArray DataArray
print(ndvi_result.mean().values)
```

## Parallel Processing with Dask

Scale to large datasets!

```bash
# Install Dask dependencies
pip install dask[array] xarray
```

```python
import dask.array as da
import xarray as xr
from eo_processor import ndvi

# Create large chunked arrays (10000x10000, processed in 1000x1000 chunks)
nir_dask = da.random.random((10000, 10000), chunks=(1000, 1000))
red_dask = da.random.random((10000, 10000), chunks=(1000, 1000))

# Wrap in XArray
nir_xr = xr.DataArray(nir_dask, dims=["y", "x"])
red_xr = xr.DataArray(red_dask, dims=["y", "x"])

# Compute NDVI (lazy evaluation)
ndvi_result = xr.apply_ufunc(
    ndvi,
    nir_xr,
    red_xr,
    dask="parallelized",  # True parallelism, bypasses GIL!
    output_dtypes=[float],
)

# Trigger computation
ndvi_computed = ndvi_result.compute()
print(f"Processed {ndvi_computed.size:,} pixels")
```

## Available Functions

| Function | Description | Use Case |
|----------|-------------|----------|
| `ndvi(nir, red)` | Normalized Difference Vegetation Index | Vegetation health monitoring |
| `ndwi(green, nir)` | Normalized Difference Water Index | Water body detection |
| `normalized_difference(a, b)` | Generic normalized difference | Custom spectral indices |

## Next Steps

- See `examples/basic_usage.py` for more examples
- Check `examples/xarray_dask_usage.py` for advanced workflows
- Read the [full documentation](README.md) for detailed information
- Learn about [contributing](CONTRIBUTING.md) to add new features

## Common Questions

**Q: Why use Rust instead of pure Python/NumPy?**

A: While NumPy is fast for simple operations, Rust provides:
- True parallelism (bypasses Python's GIL)
- Better performance in complex multi-step calculations
- Memory safety guarantees
- Seamless integration with Dask for distributed computing

**Q: Does it work with real satellite data?**

A: Yes! It works with any NumPy arrays, including those from:
- Rasterio (reading GeoTIFF files)
- GDAL
- Sentinel-2, Landsat data
- Any satellite imagery in NumPy format

**Q: Can I add custom spectral indices?**

A: Absolutely! See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add new functions.

## Troubleshooting

**Build errors**: Ensure Rust is installed and up to date:
```bash
rustup update
```

**Import errors**: Make sure the package is installed:
```bash
pip list | grep eo-processor
```

**Slow performance**: Use the release build:
```bash
maturin develop --release
```

## Example: Real-World Workflow

```python
import numpy as np
from eo_processor import ndvi
import rasterio

# Read satellite bands
with rasterio.open('sentinel2_nir.tif') as nir_src:
    nir = nir_src.read(1).astype(np.float64) / 10000.0

with rasterio.open('sentinel2_red.tif') as red_src:
    red = red_src.read(1).astype(np.float64) / 10000.0

# Compute NDVI
ndvi_result = ndvi(nir, red)

# Classify vegetation
vegetation_mask = ndvi_result > 0.3
print(f"Vegetation coverage: {vegetation_mask.sum() / vegetation_mask.size * 100:.1f}%")

# Save result
with rasterio.open(
    'ndvi_output.tif', 'w',
    driver='GTiff',
    height=ndvi_result.shape[0],
    width=ndvi_result.shape[1],
    count=1,
    dtype=ndvi_result.dtype,
) as dst:
    dst.write(ndvi_result, 1)
```

Happy processing! 🛰️🌍
