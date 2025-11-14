# eo-processor
[![PyPI Version](https://img.shields.io/pypi/v/eo-processor.svg?color=blue)](https://pypi.org/project/eo-processor/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/eo-processor?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/eo-processor)
[![Coverage](./coverage-badge.svg)](#test-coverage)
[![Documentation Status](https://readthedocs.org/projects/eo-processor/badge/?version=latest)](https://eo-processor.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-performance Rust (PyO3) UDFs for Earth Observation (EO) processing with Python bindings.
Fast spectral indices, temporal statistics, masking utilities, and spatial distance functions.

---

## Overview

`eo-processor` accelerates common remote sensing computations using safe Rust (no `unsafe`) exposed via PyO3.
All public functions interoperate with NumPy and can be embedded in XArray / Dask pipelines.
Rust kernels release Python's GIL; multi-core parallelism (via Rayon) is leveraged for selected operations (larger temporal aggregations, pairwise distances).

Focus areas:
- Spectral & change-detection indices
- Temporal statistics & median compositing (1D–4D stacks)
- Masking & data quality filtering (value / range / SCL / invalid sentinels)
- Pairwise spatial distances (utility layer)
- Benchmark harness for reproducible performance measurements

---

## Key Features

- Rust-accelerated numerical kernels (float64 internal, stable results)
- Automatic dimensional dispatch (1D / 2D for spectral indices, 1D–4D for temporal/masking)
- Change detection support (ΔNDVI, ΔNBR)
- Flexible masking utilities (exact values, ranges, SCL codes)
- Median, mean, sample standard deviation over time axis
- Pairwise distance functions (Euclidean, Manhattan, Chebyshev, Minkowski)
- Type stubs (`__init__.pyi`) for IDE / mypy
- Benchmark script with optional NumPy baseline comparison
- Pure CPU, no external network or storage side-effects in core path

---

## Installation

### PyPI (standard)

```bash
pip install eo-processor
```

Optional extras for array ecosystem:

```bash
pip install eo-processor[dask]
```

### Using `uv`

```bash
uv venv
source .venv/bin/activate
uv pip install eo-processor
```

### From Source

Requirements:
- Python 3.8+
- Rust toolchain (`rustup` recommended)
- `maturin` for building the extension module

```bash
git clone https://github.com/BnJam/eo-processor.git
cd eo-processor

pip install maturin
maturin develop --release        # build & install in-place
# or wheel:
maturin build --release
pip install target/wheels/*.whl
```

---

## Quick Start

```python
import numpy as np
from eo_processor import ndvi, ndwi, evi, normalized_difference

nir   = np.array([0.8, 0.7, 0.6])
red   = np.array([0.2, 0.1, 0.3])
blue  = np.array([0.1, 0.05, 0.08])
green = np.array([0.35, 0.42, 0.55])

print(ndvi(nir, red))               # NDVI
print(ndwi(green, nir))             # NDWI
print(evi(nir, red, blue))          # EVI
print(normalized_difference(nir, red))
```

All inputs may be any numeric NumPy dtype (int/uint/float); internal coercion to `float64`.

---

## API Summary

| Function | Purpose |
|----------|---------|
| `normalized_difference(a, b)` | Generic normalized difference `(a - b) / (a + b)` with near-zero denominator safeguard |
| `ndvi(nir, red)` | Normalized Difference Vegetation Index |
| `ndwi(green, nir)` | Normalized Difference Water Index |
| `evi(nir, red, blue)` / `enhanced_vegetation_index(...)` | Enhanced Vegetation Index (G*(NIR - Red)/(NIR + C1*Red - C2*Blue + L)) |
| `savi(nir, red, L=0.5)` | Soil Adjusted Vegetation Index `(NIR - Red)/(NIR + Red + L) * (1 + L)` |
| `nbr(nir, swir2)` | Normalized Burn Ratio `(NIR - SWIR2)/(NIR + SWIR2)` |
| `ndmi(nir, swir1)` | Normalized Difference Moisture Index `(NIR - SWIR1)/(NIR + SWIR1)` |
| `nbr2(swir1, swir2)` | Normalized Burn Ratio 2 `(SWIR1 - SWIR2)/(SWIR1 + SWIR2)` |
| `gci(nir, green)` | Green Chlorophyll Index `(NIR / Green) - 1` (division guard) |
| `delta_ndvi(pre_nir, pre_red, post_nir, post_red)` | Change in NDVI `(NDVI_pre - NDVI_post)` |
| `delta_nbr(pre_nir, pre_swir2, post_nir, post_swir2)` | Change in NBR `(NBR_pre - NBR_post)` |
| `median(arr, skip_na=True)` | Temporal median (time axis) with NaN skipping |
| `composite(arr, method="median")` | Compositing convenience (currently median only) |
| `temporal_mean(arr, skip_na=True)` | Mean across time axis |
| `temporal_std(arr, skip_na=True)` | Sample standard deviation (n-1) across time |
| `euclidean_distance(points_a, points_b)` | Pairwise Euclidean distances |
| `manhattan_distance(points_a, points_b)` | Pairwise L1 distances |
| `chebyshev_distance(points_a, points_b)` | Pairwise L∞ distances |
| `minkowski_distance(points_a, points_b, p)` | Pairwise L^p distances (p ≥ 1) |
| `mask_vals(arr, values=None, fill_value=None, nan_to=None)` | Mask exact codes, optional fill & NaN normalization |
| `replace_nans(arr, value)` | Replace all NaNs with `value` |
| `mask_out_range(arr, min_val=None, max_val=None, fill_value=None)` | Mask values outside `[min, max]` |
| `mask_in_range(arr, min_val=None, max_val=None, fill_value=None)` | Mask values inside `[min, max]` |
| `mask_invalid(arr, invalid_values, fill_value=None)` | Mask list of sentinel values (e.g., `0, -9999`) |
| `mask_scl(scl, keep_codes=None, fill_value=None)` | Mask Sentinel‑2 SCL codes, keeping selected classes |

Temporal dimension expectations:
- 1D: `(time,)`
- 2D: `(time, band)`
- 3D: `(time, y, x)`
- 4D: `(time, band, y, x)`

Distance functions: input shape `(N, D)` and `(M, D)` → output `(N, M)` (O(N*M) memory/time).

---

## Spectral & Change Detection Indices

All indices auto-dispatch 1D vs 2D arrays (matching shapes required).

### NDVI
`(NIR - Red) / (NIR + Red)`
Interpretation (approximate):
- < 0: water / snow
- 0.0–0.2: bare soil / built surfaces
- 0.2–0.5: sparse to moderate vegetation
- > 0.5: healthy dense vegetation

### NDWI
`(Green - NIR) / (Green + NIR)`
- > 0.3: open water (often 0.4–0.6)
- 0.0–0.3: moist vegetation / wetlands
- < 0.0: dry vegetation / soil

### EVI
`G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)` (MODIS constants: G=2.5, C1=6.0, C2=7.5, L=1.0)
Improves sensitivity over high biomass & reduces soil/atmospheric noise vs NDVI.

### SAVI
`(NIR - Red) / (NIR + Red + L) * (1 + L)`
Typical `L=0.5`. Larger `L` for sparse vegetation (bright soil), smaller for dense vegetation.

### NBR
`(NIR - SWIR2) / (NIR + SWIR2)`
Used for burn severity. Compare pre/post via ΔNBR.

### NDMI
`(NIR - SWIR1) / (NIR + SWIR1)`
Moisture / canopy water content indicator.

### NBR2
`(SWIR1 - SWIR2) / (SWIR1 + SWIR2)`
Highlights moisture & thermal differences; complementary to NBR/NDMI.

### GCI
`(NIR / Green) - 1`
Chlorophyll proxy; division by near-zero guarded to avoid instability.

### Change Detection
`ΔNDVI = NDVI_pre - NDVI_post`
`ΔNBR  = NBR_pre  - NBR_post`
Positive ΔNDVI: vegetation loss. Positive ΔNBR: burn severity increase.

---

## Masking Utilities

Rust-accelerated preprocessing helpers for quality filtering.

| Function | Notes |
|----------|-------|
| `mask_vals` | Exact equality masking (codes → `fill_value` or NaN) + optional NaN normalization |
| `replace_nans` | Force all NaNs to a scalar |
| `mask_out_range` | Mask outside interval |
| `mask_in_range` | Mask inside interval |
| `mask_invalid` | Shorthand for common invalid sentinels |
| `mask_scl` | Keep only selected Sentinel‑2 SCL classes |

Example:

```python
import numpy as np
from eo_processor import mask_vals, replace_nans, mask_out_range, mask_scl

scl = np.array([4,5,6,8,9])  # vegetation, vegetation, water, cloud (med), cloud (high)
clear = mask_scl(scl, keep_codes=[4,5,6])   # -> [4., 5., 6., nan, nan]

ndvi = np.array([-0.3, 0.1, 0.8, 1.2])
valid = mask_out_range(ndvi, min_val=-0.2, max_val=1.0)  # -> [nan,0.1,0.8,nan]

arr = np.array([0, 100, -9999, 50])
clean = mask_vals(arr, values=[0, -9999])  # -> [nan,100.,nan,50.]
filled = replace_nans(clean, -9999.0)      # -> [-9999.,100.,-9999.,50.]
```

---

## Temporal Statistics & Compositing

Median, mean, and standard deviation across time axis (skip NaNs optional):

```python
import numpy as np
from eo_processor import temporal_mean, temporal_std, median

cube = np.random.rand(12, 256, 256)  # (time, y, x)
mean_img  = temporal_mean(cube)      # (256, 256)
std_img   = temporal_std(cube)       # (256, 256)
median_img = median(cube)
```

`composite(cube, method="median")` currently routes to `median`.

---

## Spatial Distances

Pairwise distance matrices:

```python
import numpy as np
from eo_processor import euclidean_distance, manhattan_distance

A = np.random.rand(100, 8)  # (N, D)
B = np.random.rand(250, 8)  # (M, D)

dist_e = euclidean_distance(A, B)    # (100, 250)
dist_l1 = manhattan_distance(A, B)
```

For large N*M consider spatial indexing or chunking (not implemented).

---

## XArray / Dask Integration

```python
import dask.array as da
import xarray as xr
from eo_processor import ndvi

nir_dask  = da.random.random((5000, 5000), chunks=(500, 500))
red_dask  = da.random.random((5000, 5000), chunks=(500, 500))

nir_xr = xr.DataArray(nir_dask, dims=["y", "x"])
red_xr = xr.DataArray(red_dask, dims=["y", "x"])

ndvi_xr = xr.apply_ufunc(
    ndvi,
    nir_xr,
    red_xr,
    dask="parallelized",
    output_dtypes=[float],
)

result = ndvi_xr.compute()
```

---

## CLI Usage

Console script exposed as `eo-processor` (installed via PyPI):

```bash
# Single index
eo-processor --index ndvi --nir nir.npy --red red.npy --out ndvi.npy

# Multiple indices (provide necessary bands)
eo-processor --index ndvi savi ndmi nbr --nir nir.npy --red red.npy --swir1 swir1.npy --swir2 swir2.npy --out-dir outputs/

# Change detection (ΔNBR)
eo-processor --index delta_nbr \
  --pre-nir pre/nir.npy --pre-swir2 pre/swir2.npy \
  --post-nir post/nir.npy --post-swir2 post/swir2.npy \
  --out outputs/delta_nbr.npy

# List supported indices
eo-processor --list

# Apply cloud mask (0=cloud, 1=clear)
eo-processor --index ndvi --nir nir.npy --red red.npy --mask cloudmask.npy --out ndvi_masked.npy

# PNG preview (requires optional Pillow)
eo-processor --index ndvi --nir nir.npy --red red.npy --out ndvi.npy --png-preview ndvi.png
```

Selected flags:
- `--savi-l` soil brightness factor for SAVI.
- `--clamp MIN MAX` output range clamping.
- `--allow-missing` skip indices lacking required bands instead of error.

---

## Performance

Example benchmark (NDVI on a large array):

```python
import numpy as np, time
from eo_processor import ndvi

nir = np.random.rand(5000, 5000)
red = np.random.rand(5000, 5000)

t0 = time.time()
rust_out = ndvi(nir, red)
t_rust = time.time() - t0

t0 = time.time()
numpy_out = (nir - red) / (nir + red)
t_numpy = time.time() - t0

print(f"Rust: {t_rust:.3f}s  NumPy: {t_numpy:.3f}s  Speedup: {t_numpy/t_rust:.2f}x")
```

Speedups depend on array shape, memory bandwidth, and CPU cores.
Use the benchmark harness for systematic comparison.

---

## Benchmark Harness

`scripts/benchmark.py` provides grouped tests:

```bash
# Spectral functions (e.g., NDVI, NDWI, EVI, SAVI, NBR, NDMI, NBR2, GCI)
python scripts/benchmark.py --group spectral --height 2048 --width 2048

# Temporal (compare Rust vs NumPy)
python scripts/benchmark.py --group temporal --time 24 --height 1024 --width 1024 --compare-numpy

# Distances
python scripts/benchmark.py --group distances --points-a 2000 --points-b 2000 --point-dim 8

# All groups; write reports
python scripts/benchmark.py --group all --compare-numpy --json-out bench.json --md-out bench.md
```

Key options:
- `--functions <list>` override group selection.
- `--compare-numpy` baseline timings (speedup > 1.0 ⇒ Rust faster).
- `--minkowski-p <p>` set order (p ≥ 1).
- `--loops`, `--warmups` repetition control.
- `--json-out`, `--md-out` artifact outputs.

---

## Test Coverage

Regenerate badge after modifying logic/tests:

```bash
tox -e coverage
python scripts/generate_coverage_badge.py coverage.xml coverage-badge.svg
```

Ensure the badge is committed if coverage changes materially.

---

## Contributing

Follow repository guidelines (`AGENTS.md`, copilot instructions). Checklist before proposing a PR:

1. Implement Rust function(s) (no `unsafe`)
2. Register via `wrap_pyfunction!` in `src/lib.rs`
3. Export in `python/eo_processor/__init__.py`
4. Add type stubs in `python/eo_processor/__init__.pyi`
5. Add tests (`tests/test_<feature>.py`) including edge cases & NaN handling
6. Update README (API Summary, examples, formulas)
7. Run:
   - `cargo fmt`
   - `cargo clippy -- -D warnings`
   - `cargo test` (if Rust tests)
   - `pytest`
   - `tox -e coverage`
   - `ruff` and `mypy` (if configured)
8. Update version if public API added (minor bump)
9. Regenerate coverage badge if changed
10. Confirm no secrets / large binaries staged

Commit message pattern:

```
<type>(scope): concise summary

Optional rationale, benchmarks, references
```

Types: feat, fix, perf, docs, test, chore, build, ci

Example:
```
feat(indices): add Green Chlorophyll Index (GCI)

Implements 1D/2D dispatch, tests, docs, benchmark entry.
```

---

## Semantic Versioning

- Patch: Internal fixes, refactors, docs only
- Minor: New functions (backward-compatible)
- Major: Breaking changes (signature changes, removals)

---

## Roadmap (Indicative)

- Additional spectral indices (future: NBR derivatives, custom moisture composites)
- Sliding window / neighborhood statistics (mean, variance)
- Optional multithread strategies for very large temporal cubes
- Expanded masking (boolean predicate composition)
- Extended change metrics (ΔNDMI, fractional vegetation cover)

(Items requiring strategic design will request human review before implementation.)

---

## Scientific Citation

```bibtex
@software{eo_processor,
  title   = {eo-processor: High-performance Rust UDFs for Earth Observation},
  author  = {Ben Smith},
  year    = {2025},
  url     = {https://github.com/BnJam/eo-processor}
}
```

---

## License

MIT License – see `LICENSE`.

---

## Disclaimer

Core library focuses on computational primitives. It does NOT perform:
- Sensor-specific radiometric calibration
- Atmospheric correction
- CRS reprojection / spatial indexing
- Cloud/shadow detection algorithms beyond simple masking
- Data acquisition / I/O orchestration

Integrate with domain tools (rasterio, xarray, dask, geopandas) for full pipelines.

---

## Support

Open issues for bugs or enhancements. Provide:
- Reproducible snippet
- Input shapes / dtypes
- Expected vs actual output
- Benchmark data (if performance-related)

---



---

## Acknowledgements

Built with PyO3, NumPy, ndarray, and Rayon.
Thanks to the scientific EO community for standardized index formulations.

---

Enjoy fast, reproducible Earth Observation processing!
