# eo-processor

[![Coverage](./coverage-badge.svg)](#test-coverage)

High-performance Rust UDFs for Earth Observation (EO) processing with Python bindings.

## Overview

`eo-processor` is a framework that provides Rust-based User Defined Functions (UDFs) for common Earth Observation and geospatial computations. These functions can be used within local and remote (Dask/Kubernetes) workflows, leveraging PyO3 to create highly efficient and optimized operations.

The Rust implementation bypasses Python's Global Interpreter Lock (GIL), making it ideal for:
- Long-running computations on large satellite imagery
- Parallel processing with Dask
- XArray `apply_ufunc` and `map_blocks` workflows
- CPU-intensive geospatial operations

## Features

- **High Performance**: Rust-accelerated computations that bypass Python's GIL
- **Easy Integration**: Works seamlessly with NumPy, XArray, and Dask
- **Common EO Indices**: Pre-implemented functions for NDVI, NDWI, and generic normalized differences
- **Type Safe**: Full type hints for Python IDE support
- **Flexible**: Supports both 1D and 2D arrays with automatic dimension detection

## Installation

### From Source

Requirements:
- Python 3.8+
- Rust toolchain (install from [rustup.rs](https://rustup.rs/))

```bash
# Install maturin for building
pip install maturin

# Build and install the package
maturin develop --release

# Or build wheel for distribution
maturin build --release
pip install target/wheels/*.whl
```

## Usage

### Basic Usage

```python
import numpy as np
from eo_processor import ndvi, ndwi, normalized_difference

# Compute NDVI from NIR and Red bands
nir = np.array([0.8, 0.7, 0.6])
red = np.array([0.2, 0.1, 0.3])
ndvi_result = ndvi(nir, red)
print(ndvi_result)  # [0.6, 0.75, 0.33333333]

# Note: the functions in `eo_processor` now return NumPy arrays directly.
# They no longer return a (array, dims) tuple or any dims metadata.

# Works with 2D arrays (images)
nir_image = np.random.rand(1000, 1000)
red_image = np.random.rand(1000, 1000)
ndvi_image = ndvi(nir_image, red_image)

# Compute NDWI (water index)
green = np.array([0.3, 0.4, 0.5])
ndwi_result = ndwi(green, nir)

# Generic normalized difference: (a - b) / (a + b)
custom_index = normalized_difference(band_a, band_b)
```

### XArray Integration

```python
import xarray as xr
from eo_processor import ndvi

# Create XArray DataArrays
nir = xr.DataArray(nir_data, dims=["y", "x"])
red = xr.DataArray(red_data, dims=["y", "x"])

# Apply using xr.apply_ufunc
ndvi_result = xr.apply_ufunc(
    ndvi,
    nir,
    red,
    dask="parallelized",
    output_dtypes=[float],
)
```

### Dask Integration (Parallel Processing)

```python
import dask.array as da
import xarray as xr
from eo_processor import ndvi

# Create large Dask arrays (chunked for parallel processing)
nir_dask = da.random.random((10000, 10000), chunks=(1000, 1000))
red_dask = da.random.random((10000, 10000), chunks=(1000, 1000))

# Wrap in XArray
nir_xr = xr.DataArray(nir_dask, dims=["y", "x"])
red_xr = xr.DataArray(red_dask, dims=["y", "x"])

# Compute NDVI (bypasses GIL, enables true parallelism)
ndvi_result = xr.apply_ufunc(
    ndvi,
    nir_xr,
    red_xr,
    dask="parallelized",
    output_dtypes=[float],
)

# Compute result
ndvi_computed = ndvi_result.compute()
```

### Using map_blocks with Dask

```python
import dask.array as da
from eo_processor import ndvi

nir_dask = da.random.random((5000, 5000), chunks=(500, 500))
red_dask = da.random.random((5000, 5000), chunks=(500, 500))

# Apply to blocks (each block processed independently)
ndvi_result = da.map_blocks(
    ndvi,
    nir_dask,
    red_dask,
    dtype=np.float64,
)

result = ndvi_result.compute()
```

## Available Functions

### Normalized Difference Functions

- `normalized_difference(a, b)`: Generic normalized difference `(a - b) / (a + b)`
- `normalized_difference_1d(a, b)`: 1D version
- `normalized_difference_2d(a, b)`: 2D version

### Vegetation Indices

- `ndvi(nir, red)`: Normalized Difference Vegetation Index
- `ndvi_1d(nir, red)`: 1D version
- `ndvi_2d(nir, red)`: 2D version

### Water Indices

- `ndwi(green, nir)`: Normalized Difference Water Index
- `ndwi_1d(green, nir)`: 1D version
- `ndwi_2d(green, nir)`: 2D version

## Performance

The Rust implementation provides significant performance improvements over pure Python/NumPy, especially for large arrays:

```python
import numpy as np
import time
from eo_processor import ndvi

# Large array
nir = np.random.rand(5000, 5000)
red = np.random.rand(5000, 5000)

# Rust implementation
start = time.time()
result_rust = ndvi(nir, red)
time_rust = time.time() - start

# NumPy implementation
start = time.time()
result_numpy = (nir - red) / (nir + red)
time_numpy = time.time() - start

print(f"Rust: {time_rust:.4f}s")
print(f"NumPy: {time_numpy:.4f}s")
print(f"Speedup: {time_numpy/time_rust:.2f}x")
```

## Development

### Building

```bash
# Development build
maturin develop

# Release build
maturin develop --release
```

### Testing

```bash
# Run Rust tests
cargo test

# Run Python tests (if pytest is installed)
pytest
```

### Running Examples

```bash
# Basic usage examples
python examples/basic_usage.py

# XArray/Dask examples (requires: pip install eo-processor[dask])
python examples/xarray_dask_usage.py
```

## Why Rust + PyO3?

1. **Performance**: Rust provides C-level performance with memory safety
2. **GIL-Free**: Rust code releases the Python GIL, enabling true parallelism
3. **Type Safety**: Compile-time guarantees reduce runtime errors
4. **Easy Integration**: PyO3 makes it seamless to call Rust from Python
5. **Modern Tooling**: Cargo and maturin provide excellent development experience

## Use Cases

- Processing large satellite imagery datasets (Sentinel, Landsat, etc.)
- Real-time vegetation monitoring using NDVI
- Water body detection using NDWI
- Custom spectral indices computation
- Distributed processing on Dask/Kubernetes clusters
- Time-series analysis of Earth Observation data

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{eo_processor,
  title = {eo-processor: High-performance Rust UDFs for Earth Observation},
  author = {Ben},
  year = {2025},
  url = {https://github.com/BnJam/eo-processor}
}
```
