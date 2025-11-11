# Examples for eo-processor

This directory contains runnable Python scripts demonstrating usage of the high-performance Rust-backed Earth Observation functions provided by `eo_processor`.

## Files

- `basic_usage.py`: Core spectral indices (NDVI, NDWI, EVI, normalized difference) on 1D and 2D arrays plus performance snippet and temporal stats.
- `xarray_dask_usage.py`: Integration with XArray and Dask for chunked / lazy evaluation and time-series workflows.
- `map_blocks.py`: Compares `xarray.apply_ufunc`, `xarray.map_blocks`, and `dask.array.map_blocks` for NDVI computation performance.
- `temporal_operations.py`: Demonstrates `temporal_mean`, `temporal_std`, median compositing, and NaN handling.
- `spatial_distances.py`: Shows internal `_core` spatial distance functions (Euclidean, Manhattan, Chebyshev, Minkowski). These are not yet exported at the top level.

## Quick Run

Activate your environment and run any example:

`python examples/basic_usage.py`

If you cloned the repo and want optional Dask/XArray examples:

`pip install .[dask]`

or with uv:

`uv pip install .[dask]`

## Prerequisites

- Python >= 3.8
- For spectral & temporal examples: only NumPy required.
- For Dask / XArray examples: install extras: `pip install eo-processor[dask]`
- Rust toolchain only needed if rebuilding from source (not for running examples).

## Conventions

- Array dimension meanings:
  - 1D: spectral sample vector
  - 2D: spatial image (y, x)
  - 3D: (time, y, x)
  - 4D: (time, band, y, x)
- Temporal functions treat the first axis as time.
- Median, mean, and std support `skip_na=True` (default) to ignore NaNs.

## Performance Notes

- Example timings are indicative only; real speed depends on CPU cache, vectorization, and memory bandwidth.
- Rust functions release the GIL enabling parallel execution when orchestrated by Dask/XArray.
- Avoid benchmarking with very small arrays; overhead dominates at tiny sizes.

## Spatial Distance Functions

Spatial distance utilities live in the internal module and are accessed via:

`from eo_processor import _core`
`dist = _core.euclidean_distance(points_a, points_b)`

They return a dense (N, M) matrix; for large N and M consider future spatial indexing (KD-tree / ball tree) enhancements.

## Adding a New Example

1. Create a script named descriptively (e.g., `spectral_savi.py`).
2. Import only what you need (prefer explicit imports).
3. Keep run output concise (stats, shapes, timing).
4. Avoid hard dependencies beyond the project unless absolutely necessary.
5. If demonstrating a new public API function, update the main project `README.md` accordingly.
6. Do not commit large data files; generate synthetic arrays.

## Error Handling

Examples intentionally fail fast on missing optional dependencies; install extras first if you see an ImportError.

## Reproducibility

For reproducible demonstrations, you can set a NumPy random seed near the top of the script: `rng = np.random.default_rng(42)` and replace `np.random` calls with `rng`.

## Contributing Examples

- Keep scripts self-contained; no shared utility dependencies.
- Provide a brief module docstring summarizing intent.
- Prefer printing summary statistics instead of full arrays.

## License

All example code is MIT licensed under the project’s primary LICENSE.

## Disclaimer

Examples focus on computation primitives; they do not include EO-specific preprocessing (cloud masking, reprojection, radiometric corrections).

Happy experimenting!
