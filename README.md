# eo-processor
[![PyPI Version](https://img.shields.io/pypi/v/eo-processor.svg?color=blue)](https://pypi.org/project/eo-processor/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/eo-processor?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/eo-processor)
[![Coverage](./coverage-badge.svg)](#test-coverage)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-performance Rust UDFs for Earth Observation (EO) processing with Python bindings.
Provides fast spectral indices, temporal statistics, and (internally) spatial distance utilities.

---

## Overview

`eo-processor` accelerates common Earth Observation and geospatial computations using Rust + PyO3, exposing a Python API compatible with NumPy, XArray, and Dask. Rust execution bypasses Python's Global Interpreter Lock (GIL), enabling true parallelism in multi-core environments and large-array workflows.

Primary focus areas:
- Spectral indices (NDVI, NDWI, EVI, generic normalized differences)
- Temporal compositing/statistics (median, mean, standard deviation)
- Spatial utilities (distance computations — currently available via internal module)

---

## Key Features

- Rust-accelerated numerical kernels (safe, no `unsafe` code)
- Automatic dimensional dispatch (1D vs 2D for spectral indices)
- Temporal statistics across a leading “time” axis for 1D–4D arrays
- Optional skipping of NaN values (`skip_na=True`)
- Ready for XArray / Dask parallelized workflows
- Type hints and stubs for IDE assistance
- Deterministic, GIL-efficient performance

---

## Installation

### Using `pip` (PyPI)

```bash
pip install eo-processor
```

Optional extras (for distributed / parallel array workflows):

```bash
pip install eo-processor[dask]
```

### Using `uv` (fast dependency manager)

```bash
# Create and sync environment
uv venv
source .venv/bin/activate
uv pip install eo-processor
```

### From Source

Requirements:
- Python 3.8+
- Rust toolchain (install via https://rustup.rs/)
- `maturin` for building the extension

```bash
git clone https://github.com/BnJam/eo-processor.git
cd eo-processor

# Build & install in editable (development) mode
pip install maturin
maturin develop --release

# Or build a wheel
maturin build --release
pip install target/wheels/*.whl
```

---

## Quick Start

```python
import numpy as np
from eo_processor import ndvi, ndwi, evi, normalized_difference

nir  = np.array([0.8, 0.7, 0.6])
red  = np.array([0.2, 0.1, 0.3])
blue = np.array([0.1, 0.05, 0.08])
green = np.array([0.35, 0.42, 0.55])

ndvi_vals = ndvi(nir, red)
ndwi_vals = ndwi(green, nir)
evi_vals  = evi(nir, red, blue)
nd_generic = normalized_difference(nir, red)

print(ndvi_vals, ndwi_vals, evi_vals, nd_generic)
```

All spectral index functions return NumPy arrays directly (no tuple wrappers).

---

## API Summary

Top-level Python exports (via `eo_processor`):

| Function | Description |
|----------|-------------|
| `normalized_difference(a, b)` | Generic `(a - b) / (a + b)` with zero-denominator safeguard |
| `ndvi(nir, red)` | Normalized Difference Vegetation Index |
| `ndwi(green, nir)` | Normalized Difference Water Index |
| `enhanced_vegetation_index(nir, red, blue)` / `evi(...)` | Enhanced Vegetation Index (EVI: G*(NIR-Red)/(NIR + C1*Red - C2*Blue + L)) |
| `median(arr, skip_na=True)` | Temporal median across leading axis for 1D–4D arrays |
| `composite(arr, method="median", **kwargs)` | Convenience wrapper (currently only median) |
| `temporal_mean(arr, skip_na=True)` | Mean across time dimension |
| `temporal_std(arr, skip_na=True)` | Sample standard deviation (n-1 denominator) across time |
| `savi(nir, red, L=0.5)` | Soil Adjusted Vegetation Index: (NIR - Red)/(NIR + Red + L) * (1 + L); variable L (≥ 0) |
| `nbr(nir, swir2)` | Normalized Burn Ratio: (NIR - SWIR2)/(NIR + SWIR2) |
| `ndmi(nir, swir1)` | Normalized Difference Moisture Index: (NIR - SWIR1)/(NIR + SWIR1) |
| `nbr2(swir1, swir2)` | Normalized Burn Ratio 2: (SWIR1 - SWIR2)/(SWIR1 + SWIR2) |
| `gci(nir, green)` | Green Chlorophyll Index: (NIR / Green) - 1 (division guarded) |
| `delta_ndvi(pre_nir, pre_red, post_nir, post_red)` | Change in NDVI (pre - post); vegetation loss (positive values often indicate decrease in post-event NDVI) |
| `delta_nbr(pre_nir, pre_swir2, post_nir, post_swir2)` | Change in NBR (pre - post); burn severity (higher positive change suggests more severe burn) |

Spatial distance functions (pairwise distance matrices; now exported at the top level — note O(N*M) memory/time for large point sets). Formulas (a, b ∈ ℝ^D).
All spectral/temporal index functions accept any numeric NumPy dtype (int, uint, float32, float64, etc.); inputs are automatically coerced to float64 internally for consistency:
- Euclidean: √(∑ᵢ (aᵢ - bᵢ)²)
- Manhattan (L₁): ∑ᵢ |aᵢ - bᵢ|
- Chebyshev (L_∞): maxᵢ |aᵢ - bᵢ|
- Minkowski (L_p): (∑ᵢ |aᵢ - bᵢ|^p)^(1/p), with p ≥ 1.0 (this library enforces p ≥ 1)


| Function | Description |
|----------|-------------|
| `euclidean_distance(points_a, points_b)` | Pairwise Euclidean distances (shape (N,M)) |
| `manhattan_distance(points_a, points_b)` | Pairwise L1 distance |
| `chebyshev_distance(points_a, points_b)` | Pairwise max-abs (L∞) distance |
| `minkowski_distance(points_a, points_b, p)` | Pairwise L^p distance |
| (Median helpers for dimension dispatch) | Implementations backing `median` |

If you need spatial distance functions at the top level, add them to `python/eo_processor/__init__.py` and re-export.

---

## Spectral Indices

### NDVI
Formula: `(NIR - Red) / (NIR + Red)`
Typical interpretation:
- Water / snow: < 0 (often strongly negative for clear water)
- Bare soil / built surfaces: ~ 0.0 – 0.2
- Sparse vegetation / stressed crops: 0.2 – 0.5
- Healthy dense vegetation: > 0.5 (tropical forest can exceed 0.7)

### NDWI
Formula: `(Green - NIR) / (Green + NIR)`
Typical interpretation:
- Open water bodies: > 0.3 (often 0.4–0.6)
- Moist vegetation / wetlands: 0.0 – 0.3
- Dry vegetation / bare soil: < 0.0 (negative values)

### EVI
Formula:
`EVI = G * (NIR - Red) / (NIR + C1 * Red - C2 * Blue + L)`
Constants (MODIS): `G=2.5, C1=6.0, C2=7.5, L=1.0`
Typical interpretation:
- EVI dampens soil & atmospheric effects relative to NDVI
- Moderate vegetation: ~0.2 – 0.4
- Dense / healthy canopy: >0.4 (can reach ~0.8 in lush tropical zones)
- Very low / senescent vegetation: <0.2

### SAVI
Formula:
`SAVI = (NIR - Red) / (NIR + Red + L) * (1 + L)`
Typical soil factor `L=0.5`; recommended range 0–1. Higher L reduces soil background effects. Implementation supports variable `L` (must be ≥ 0).
Interpretation (similar to NDVI but more robust over bright soil):
- Bare / bright soil: ~0.0 – 0.2
- Moderate vegetation: 0.2 – 0.5
- Healthy dense vegetation: > 0.5
Use smaller L (e.g. 0.25) for dense vegetation, larger L (~1.0) for very sparse vegetation / bright soil conditions.

### NBR
Formula:
`NBR = (NIR - SWIR2) / (NIR + SWIR2)`
Used for burn severity and post-fire change detection.
Typical interpretation (pre-fire vs post-fire):
- Healthy vegetation (pre-fire): high positive (≈0.4 – 0.7)
- Recently burned areas: strong drop; post-fire NBR often < 0.1 or negative
Change analysis often uses ΔNBR (pre - post). Common burn severity thresholds (example ranges, refine per study):
- ΔNBR > 0.66: High severity
- 0.44 – 0.66: Moderate-high
- 0.27 – 0.44: Moderate-low
- 0.1 – 0.27: Low severity
- < 0.1: Unburned / noise

### NDMI
Formula:
`NDMI = (NIR - SWIR1) / (NIR + SWIR1)`
Moisture / canopy water content indicator.
Typical interpretation:
- High positive (>0.3): Moist / healthy canopy (leaf water content high)
- Near zero (0.0 – 0.3): Moderate moisture / possible stress onset
- Negative (<0.0): Dry vegetation / senescence / possible drought stress

### NBR2
Formula:
`NBR2 = (SWIR1 - SWIR2) / (SWIR1 + SWIR2)`
Highlights burn severity and subtle thermal / moisture differences.
Typical interpretation:
- Lower values: Increased moisture / less burn impact
- Higher values: Greater dryness / potential higher burn severity
Use in tandem with NBR or NDMI for refined burn severity or moisture discrimination.

### GCI
Formula:
`GCI = (NIR / Green) - 1`
Green Chlorophyll Index; division by near-zero Green values is guarded to return 0.
Typical interpretation:
- Values > 0 indicate chlorophyll presence
- 0 – 2: Sparse to moderate chlorophyll (grassland, early growth)
- 2 – 8: Higher chlorophyll density (crops peak growth, healthy canopy)
- > 8: Very dense chlorophyll (may indicate saturation; verify sensor & calibration)
Absolute ranges vary with sensor, atmospheric correction, and reflectance scaling—use relative comparisons or time-series trends.

All indices auto-dispatch between 1D and 2D input arrays; shapes must match.

### Change Detection Indices

Change detection indices operate on “pre” and “post” event imagery (e.g., before vs after fire, storm, harvest):

Formulae:
`ΔNDVI = NDVI(pre) - NDVI(post)`
`ΔNBR  = NBR(pre)  - NBR(post)`

Typical interpretation:
- Positive ΔNDVI: vegetation loss / canopy degradation.
- Near-zero ΔNDVI: minimal change.
- Positive ΔNBR: higher burn severity (consult study-specific threshold tables).
- Use masks (cloud, snow, shadow) to set unreliable pre/post pixels to NaN before computing deltas.

These delta indices also accept any numeric dtype; values are coerced to float64.

### CLI Usage

A command-line helper is available (`scripts/eo_cli.py`) to batch compute indices from .npy band files.
You can now also invoke the packaged CLI directly (added in 0.4.0), either as a module or via the installed console script.

Single index (script form):
```
python scripts/eo_cli.py --index ndvi --nir data/nir.npy --red data/red.npy --out outputs/ndvi.npy
```

Single index (package module form):
```
python -m eo_processor.cli --index ndvi --nir data/nir.npy --red data/red.npy --out outputs/ndvi.npy
```

Single index (console script after installation):
```
eo-processor --index ndvi --nir data/nir.npy --red data/red.npy --out outputs/ndvi.npy
```

List supported indices:
```
eo-processor --list
```

Multiple indices:
```
python scripts/eo_cli.py --index ndvi savi ndmi nbr --nir data/nir.npy --red data/red.npy --swir1 data/swir1.npy --swir2 data/swir2.npy --out-dir outputs/
```

Change detection:
```
python -m eo_processor.cli --index delta_nbr \
  --pre-nir pre/nir.npy --pre-swir2 pre/swir2.npy \
  --post-nir post/nir.npy --post-swir2 post/swir2.npy \
  --out outputs/delta_nbr.npy
```

Cloud mask (0=cloud, 1=clear):
```
eo-processor --index ndvi --nir data/nir.npy --red data/red.npy --mask data/cloudmask.npy --out outputs/ndvi_masked.npy
```

PNG preview:
```
eo-processor --index ndvi --nir data/nir.npy --red data/red.npy --out outputs/ndvi.npy --png-preview outputs/ndvi.png
```

Use `--savi-l` to adjust soil factor for SAVI; use `--clamp MIN MAX` to restrict output range before saving; `--allow-missing` skips indices lacking required bands. The module and console-script invocations accept the same arguments as the original `scripts/eo_cli.py`. Note: In 0.4.0 a circular import encountered when invoking the console script was resolved by removing the CLI import from `__init__`. Invoke the CLI via the installed `eo-processor` command or `python -m eo_processor.cli`—both now load without circular import issues.

---

## Temporal Statistics & Compositing

Temporal functions assume the first axis is “time”:

- 1D: `(time,)`
- 2D: `(time, band)`
- 3D: `(time, y, x)`
- 4D: `(time, band, y, x)`

Example (temporal mean of a stack):

```python
import numpy as np
from eo_processor import temporal_mean, temporal_std

# Simulate (time, y, x) stack: 10 timesteps of 256x256
cube = np.random.rand(10, 256, 256)
mean_image = temporal_mean(cube)       # shape (256, 256)
std_image  = temporal_std(cube)        # shape (256, 256)
```

Median compositing:

```python
from eo_processor import median, composite
median_image = median(cube)          # same as composite(cube, method="median")
```

Skip NaNs:

```python
cloudy_series = np.array([[0.2, np.nan, 0.5],
                          [0.25, 0.3,   0.45],
                          [0.22, np.nan, 0.47]])  # (time, band)
clean_mean = temporal_mean(cloudy_series, skip_na=True)   # ignores NaNs
strict_mean = temporal_mean(cloudy_series, skip_na=False) # bands with NaN → NaN
```

---

## Spatial Distances (Internal)

Currently available in the Rust core module:

```python
from eo_processor import _core

import numpy as np
points_a = np.array([[0.0, 0.0],
                     [1.0, 1.0]])
points_b = np.array([[1.0, 0.0],
                     [0.0, 1.0]])

dist_euclid = _core.euclidean_distance(points_a, points_b)
dist_manhat = _core.manhattan_distance(points_a, points_b)
dist_cheby  = _core.chebyshev_distance(points_a, points_b)
dist_mink   = _core.minkowski_distance(points_a, points_b, 3.0)
```

Each returns an `(N, M)` array of pairwise distances.
Note: These perform O(N*M) computations; for very large sets consider spatial indexing approaches (not yet implemented here).

---

## XArray / Dask Integration

```python
import dask.array as da
import xarray as xr
from eo_processor import ndvi

nir_dask = da.random.random((5000, 5000), chunks=(500, 500))
red_dask = da.random.random((5000, 5000), chunks=(500, 500))

nir_xr = xr.DataArray(nir_dask, dims=["y", "x"])
red_xr = xr.DataArray(red_dask, dims=["y", "x"])

ndvi_da = xr.apply_ufunc(
    ndvi,
    nir_xr,
    red_xr,
    dask="parallelized",
    output_dtypes=[float],
)

result = ndvi_da.compute()
```

---

## Performance

Rust implementations avoid Python-loop overhead and release the GIL. Example benchmark (single-thread baseline):

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

print(f"Rust: {t_rust:.4f}s  NumPy: {t_numpy:.4f}s  Speedup: {t_numpy/t_rust:.2f}x")
```

Observed speedups vary by platform and array size. Always benchmark in your environment.

---

## Test Coverage

The badge above is generated from `coverage.xml` via `scripts/generate_coverage_badge.py`.
To regenerate after test changes:

```bash
tox -e coverage
python scripts/generate_coverage_badge.py coverage.xml coverage-badge.svg
```

---

## Contributing

See `CONTRIBUTING.md` and `AGENTS.md` for guidelines (workflows, security posture, and pre-commit checklist).
Typical steps:

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
pytest
tox -e coverage
```

Add new Rust functions → export via `#[pyfunction]` → register in `src/lib.rs` → expose in `python/eo_processor/__init__.py` → add type stubs → add tests → update README.

---

## Roadmap (Indicative)

- Additional spectral indices (SAVI, NBR, GCI)
- Sliding window / neighborhood stats
- Direct distance exports at top-level
- Distributed temporal
 composites (chunk-aware)
- Optional GPU acceleration feasibility study

---

## Scientific Citation

```bibtex
@software{eo_processor,
  title = {eo-processor: High-performance Rust UDFs for Earth Observation},
  author = {Ben Smith},
  year = {2025},
  url = {https://github.com/BnJam/eo-processor}
}
```

---

## License

MIT License. See `LICENSE`.

---

## Disclaimer

This library focuses on computational primitives; it does not handle:
- Cloud masking
- Sensor-specific calibration
- CRS reprojection
- I/O of remote datasets

Combine with domain tools (e.g., rasterio, xarray, dask-geopandas) for complete EO pipelines.

---

## Support

Open issues for bugs or enhancements. Feature proposals with benchmarks and EO relevance are welcome.

---

## Benchmark Harness

A minimal benchmarking harness is provided at `scripts/benchmark.py` to measure performance of spectral, temporal, and spatial distance functions. The spectral group currently includes: ndvi, ndwi, evi, savi, nbr, ndmi, nbr2, gci, and normalized_difference.

Basic usage (spectral functions on a 2048x2048 image):
```bash
python scripts/benchmark.py --group spectral --height 2048 --width 2048
```

Compare Rust vs pure NumPy baselines (supported for spectral & temporal functions):
```bash
python scripts/benchmark.py --group temporal --time 24 --height 1024 --width 1024 --compare-numpy
```

Distance benchmarks (pairwise matrix; O(N*M)):
```bash
python scripts/benchmark.py --group distances --points-a 2000 --points-b 2000 --point-dim 8
```

Write JSON and Markdown reports:
```bash
python scripts/benchmark.py --group all --compare-numpy --json-out bench.json --md-out bench.md
```

Key options:
- `--group {spectral|temporal|distances|all}` selects predefined function sets.
- `--functions <names...>` overrides group selection with explicit functions.
- `--compare-numpy` enables baseline timing (speedup > 1.0 indicates Rust faster).
- `--loops / --warmups` control timing repetitions.
- `--json-out` writes structured results (including baseline metrics when enabled).
- `--md-out` writes a Markdown table suitable for PRs / reports.
- `--quiet` suppresses console table output (still writes artifacts).
- `--minkowski-p` sets the norm order for Minkowski distance (must be ≥ 1.0).

Example Markdown table excerpt (columns):
| Function | Mean (ms) | StDev (ms) | Min (ms) | Max (ms) | Elements | Throughput (M elems/s) | Speedup vs NumPy | Shape |

> Speedup vs NumPy = (NumPy mean time / Rust mean time); values > 1 indicate Rust is faster.

---

Happy processing!
