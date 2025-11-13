# processes_examples.py
"""
Examples demonstrating advanced temporal and pixel-wise processing utilities
provided by eo-processor:

- moving_average_temporal: Sliding window mean along leading time axis for
  1D–4D arrays with 'same' or 'valid' edge handling and NaN skipping semantics.
- pixelwise_transform: Per-element linear transform with optional clamping.

These examples illustrate usage on:
1. Simple 1D series
2. 2D (time, feature) arrays
3. 3D (time, y, x) EO data cubes
4. 4D (time, band, y, x) multi-band stacks
5. NaN handling comparisons (skip_na True vs False)
6. Chaining moving average + pixelwise transform
7. XArray/Dask integration for large EO workloads
8. Basic performance comparison vs a naïve NumPy implementation

Run this script directly to execute the demonstrations:

    python examples/processes_examples.py
"""

from __future__ import annotations

import time

import numpy as np

# Optional imports for big-data example (guarded)
try:
    import dask.array as da
    import xarray as xr
except Exception:  # pragma: no cover
    da = None
    xr = None

from eo_processor import (
    moving_average_temporal,
    pixelwise_transform,
)


# ---------------------------------------------------------------------------
# 1. Simple 1D series
# ---------------------------------------------------------------------------
def example_1d():
    series = np.array([1.0, 2.0, 3.0, 4.0])
    ma_same = moving_average_temporal(series, window=3, mode="same")
    ma_valid = moving_average_temporal(series, window=3, mode="valid")
    print("1D series:", series)
    print("Moving average (same, w=3):", ma_same)
    print("Moving average (valid, w=3):", ma_valid)


# ---------------------------------------------------------------------------
# 2. 2D (time, feature) example
# ---------------------------------------------------------------------------
def example_2d():
    arr = np.array(
        [
            [1.0, 10.0, 100.0],  # t0
            [2.0, 20.0, 200.0],  # t1
            [3.0, 30.0, 300.0],  # t2
            [4.0, 40.0, 400.0],  # t3
        ]
    )
    ma = moving_average_temporal(arr, window=3, mode="same")
    print("\n2D (time, feature) input shape:", arr.shape)
    print("Moving average (same, w=3) shape:", ma.shape)
    print("Column 0 original:", arr[:, 0], "→ MA:", ma[:, 0])


# ---------------------------------------------------------------------------
# 3. 3D (time, y, x) cube with NaNs
# ---------------------------------------------------------------------------
def example_3d_nans():
    cube = np.random.rand(8, 4, 4)
    # Inject NaNs randomly
    cube[0, 1, 2] = np.nan
    cube[3, 0, 0] = np.nan
    cube[5, 2, 1] = np.nan
    ma_skip = moving_average_temporal(cube, window=3, skip_na=True, mode="same")
    ma_noskip = moving_average_temporal(cube, window=3, skip_na=False, mode="same")
    y, x = 1, 2
    print("\n3D cube shape:", cube.shape)
    print("Pixel series (y=1,x=2):", cube[:, y, x])
    print("MA skip_na=True:", ma_skip[:, y, x])
    print("MA skip_na=False:", ma_noskip[:, y, x])


# ---------------------------------------------------------------------------
# 4. 4D (time, band, y, x) example
# ---------------------------------------------------------------------------
def example_4d():
    cube4 = np.random.rand(6, 3, 2, 2)
    out_same = moving_average_temporal(cube4, window=3, mode="same")
    out_valid = moving_average_temporal(cube4, window=3, mode="valid")
    print("\n4D cube shape:", cube4.shape)
    print("Output (same) shape:", out_same.shape)
    print("Output (valid) shape:", out_valid.shape)


# ---------------------------------------------------------------------------
# 5. Chaining moving average + pixelwise transform
# ---------------------------------------------------------------------------
def example_chain():
    cube = np.random.rand(12, 64, 64)
    ma = moving_average_temporal(cube, window=5, mode="same")
    stretched = pixelwise_transform(ma, scale=1.2, offset=-0.1, clamp_min=0.0, clamp_max=1.0)
    print("\nChaining example:")
    print("Input cube shape:", cube.shape)
    print("Moving average shape:", ma.shape)
    print("Transformed (stretched) shape:", stretched.shape)
    print("Transformed min/max:", stretched.min(), stretched.max())


# ---------------------------------------------------------------------------
# 6. Performance comparison vs naïve NumPy sliding window
#    NOTE: For large arrays, naive approach is costly; this is illustrative.
# ---------------------------------------------------------------------------
def naive_moving_average_same(series: np.ndarray, window: int) -> np.ndarray:
    """
    Simple Python/NumPy implementation (branchless, no NaN skip) – O(T * window).
    This is intentionally not optimized; used only for a rough comparison.
    """
    t = series.shape[0]
    out = np.empty(t, dtype=float)
    half_left = window // 2
    half_right = window - half_left - 1
    for i in range(t):
        start = max(0, i - half_left)
        end = min(t - 1, i + half_right)
        out[i] = series[start : end + 1].mean()
    return out


def example_perf():
    series = np.random.rand(200_000).astype(np.float64)
    window = 21
    t0 = time.perf_counter()
    rust_out = moving_average_temporal(series, window=window, mode="same")
    rust_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    naive_out = naive_moving_average_same(series, window=window)
    naive_t = time.perf_counter() - t0

    assert np.allclose(rust_out, naive_out, atol=1e-12)
    print("\nPerformance (1D series length 200k, window=21):")
    print(f"Rust moving_average_temporal: {rust_t:.4f}s")
    print(f"Naive Python version        : {naive_t:.4f}s  (speedup ~{naive_t / rust_t:.2f}x)")


# ---------------------------------------------------------------------------
# 7. XArray / Dask integration (large temporal cube)
# ---------------------------------------------------------------------------
def example_dask():
    if da is None or xr is None:
        print("\n[Dask/XArray unavailable] Skipping big-data example.")
        return
    # Large temporal cube (time, y, x)
    time_len = 48
    y, x = 2048, 2048
    chunks = (4, 256, 256)
    cube = xr.DataArray(
        da.random.random((time_len, y, x), chunks=chunks),
        dims=["time", "y", "x"],
        name="signal",
    )
    # apply_ufunc maps each chunk's numpy array to Rust UDF
    smoothed = xr.apply_ufunc(
        lambda block: moving_average_temporal(block, window=5, mode="same"),
        cube,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        dask="parallelized",
        vectorize=False,
        output_dtypes=[float],
    )
    print("\nDask/XArray example:")
    print("Original:", cube)
    print("Smoothed:", smoothed)
    # Trigger computation (may take time depending on machine)
    t0 = time.perf_counter()
    _ = smoothed.compute()
    dt = time.perf_counter() - t0
    print(f"Computed smoothed cube in {dt:.2f}s (chunks={chunks})")


# ---------------------------------------------------------------------------
# 8. Pixelwise transform examples
# ---------------------------------------------------------------------------
def example_pixelwise():
    arr = np.array([[0.05, 0.5, 1.2], [0.8, -0.3, 0.4]])
    scaled = pixelwise_transform(arr, scale=1.5, offset=-0.1, clamp_min=0.0, clamp_max=1.0)
    print("\nPixelwise transform:")
    print("Input:\n", arr)
    print("Scaled & clamped:\n", scaled)


# ---------------------------------------------------------------------------
# 9. NaN scenario for pixelwise transform
# ---------------------------------------------------------------------------
def example_pixelwise_nan():
    arr = np.array([np.nan, 1.0, 2.0, -1.0])
    transformed = pixelwise_transform(arr, scale=2.0, offset=0.5, clamp_min=0.0)
    print("\nPixelwise with NaN:")
    print("Input:", arr)
    print("Transformed:", transformed)


# ---------------------------------------------------------------------------
# main orchestrator
# ---------------------------------------------------------------------------
def main():
    example_1d()
    example_2d()
    example_3d_nans()
    example_4d()
    example_chain()
    example_perf()
    example_pixelwise()
    example_pixelwise_nan()
    example_dask()  # optional big-data integration


if __name__ == "__main__":
    main()
