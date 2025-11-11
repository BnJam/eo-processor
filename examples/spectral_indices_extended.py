#!/usr/bin/env python3
"""
Spectral Indices Extended Examples for eo-processor.

Demonstrates usage of all currently implemented spectral indices:

  - NDVI  : Normalized Difference Vegetation Index
  - NDWI  : Normalized Difference Water Index
  - EVI   : Enhanced Vegetation Index
  - SAVI  : Soil Adjusted Vegetation Index (variable L)
  - NBR   : Normalized Burn Ratio
  - NDMI  : Normalized Difference Moisture Index
  - NBR2  : Normalized Burn Ratio 2
  - GCI   : Green Chlorophyll Index
  - Generic normalized_difference

The examples cover:
  1. 1D vector usage
  2. 2D image usage
  3. Variable L for SAVI
  4. Quick performance comparison (Rust vs pure NumPy baseline)
  5. Safe handling of near-zero denominators or divisors
  6. Combined computation on a synthetic multi-band cube

Run:
  python examples/spectral_indices_extended.py

Requires:
  - eo-processor installed (Rust extension built)
  - numpy
"""

from __future__ import annotations

import time
import numpy as np

from eo_processor import (
    ndvi,
    ndwi,
    evi,
    savi,
    nbr,
    ndmi,
    nbr2,
    gci,
    normalized_difference,
)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def header(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def stats(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        print(f"{name} -> all values are NaN or non-finite")
        return
    print(
        f"{name:<8} shape={arr.shape} "
        f"min={finite.min():.4f} max={finite.max():.4f} "
        f"mean={finite.mean():.4f}"
    )


# ---------------------------------------------------------------------------
# 1. 1D Example Usage
# ---------------------------------------------------------------------------
def one_dimensional_examples():
    header("1D VECTOR EXAMPLES")

    # Simulate reflectance vectors (typical range 0–1)
    nir = np.array([0.8, 0.7, 0.6, 0.5])
    red = np.array([0.2, 0.1, 0.3, 0.25])
    blue = np.array([0.1, 0.05, 0.08, 0.07])
    green = np.array([0.35, 0.40, 0.45, 0.38])
    swir1 = np.array([0.30, 0.25, 0.28, 0.32])
    swir2 = np.array([0.18, 0.15, 0.16, 0.14])

    ndvi_vals = ndvi(nir, red)
    ndwi_vals = ndwi(green, nir)
    evi_vals = evi(nir, red, blue)
    savi_vals = savi(nir, red, L=0.5)
    nbr_vals = nbr(nir, swir2)
    ndmi_vals = ndmi(nir, swir1)
    nbr2_vals = nbr2(swir1, swir2)
    gci_vals = gci(nir, green)
    generic_vals = normalized_difference(nir, red)

    stats("NDVI", ndvi_vals)
    stats("NDWI", ndwi_vals)
    stats("EVI", evi_vals)
    stats("SAVI", savi_vals)
    stats("NBR", nbr_vals)
    stats("NDMI", ndmi_vals)
    stats("NBR2", nbr2_vals)
    stats("GCI", gci_vals)
    stats("ND(gen)", generic_vals)

    # Show equivalence of NDVI and generic normalized difference formula
    assert np.allclose(ndvi_vals, generic_vals, rtol=1e-12, atol=0.0)
    print("NDVI equals generic normalized_difference(nir, red) ✔")


# ---------------------------------------------------------------------------
# 2. 2D Image Examples
# ---------------------------------------------------------------------------
def two_dimensional_examples():
    header("2D IMAGE EXAMPLES")

    h, w = 128, 128
    rng = np.random.default_rng(42)

    nir = rng.uniform(0.3, 0.9, size=(h, w))
    red = rng.uniform(0.05, 0.4, size=(h, w))
    blue = rng.uniform(0.02, 0.25, size=(h, w))
    green = rng.uniform(0.1, 0.6, size=(h, w))
    swir1 = rng.uniform(0.2, 0.5, size=(h, w))
    swir2 = rng.uniform(0.15, 0.4, size=(h, w))

    indices = {
        "NDVI": ndvi(nir, red),
        "NDWI": ndwi(green, nir),
        "EVI": evi(nir, red, blue),
        "SAVI": savi(nir, red, L=0.5),
        "NBR": nbr(nir, swir2),
        "NDMI": ndmi(nir, swir1),
        "NBR2": nbr2(swir1, swir2),
        "GCI": gci(nir, green),
    }

    for name, arr in indices.items():
        stats(name, arr)

    # Quick sanity: ND indices should be bounded in [-1, 1]; GCI can exceed 1.
    bounded = ["NDVI", "NDWI", "NBR", "NDMI", "NBR2"]
    for name in bounded:
        arr = indices[name]
        if not (np.all(arr <= 1.0 + 1e-9) and np.all(arr >= -1.0 - 1e-9)):
            print(f"Warning: {name} out of expected range [-1,1]")
    print("Range checks completed.")


# ---------------------------------------------------------------------------
# 3. Variable L for SAVI
# ---------------------------------------------------------------------------
def savi_variable_L_demo():
    header("SAVI VARIABLE L DEMO")

    nir = np.array([0.7, 0.6, 0.5])
    red = np.array([0.2, 0.3, 0.25])
    for L in [0.0, 0.25, 0.5, 1.0]:
        out = savi(nir, red, L=L)
        expected = (nir - red) / (nir + red + L) * (1.0 + L)
        mask = np.isclose(nir + red + L, 0.0, atol=1e-10)
        expected[mask] = 0.0
        assert np.allclose(out, expected, rtol=1e-12)
        print(f"SAVI(L={L}) -> {out}")
    print("Variable L correctness verified ✔")


# ---------------------------------------------------------------------------
# 4. Performance Comparison (Rust vs NumPy baseline)
# ---------------------------------------------------------------------------
def performance_comparison():
    header("PERFORMANCE COMPARISON (Rust vs NumPy)")

    size = 1500  # moderate size for demo
    rng = np.random.default_rng(7)
    nir = rng.uniform(0.3, 0.9, size=(size, size))
    red = rng.uniform(0.05, 0.4, size=(size, size))
    swir2 = rng.uniform(0.15, 0.4, size=(size, size))

    def rust_ndvi():
        return ndvi(nir, red)

    def numpy_ndvi():
        return (nir - red) / (nir + red)

    def rust_nbr():
        return nbr(nir, swir2)

    def numpy_nbr():
        return (nir - swir2) / (nir + swir2)

    # Warmup
    rust_ndvi()
    numpy_ndvi()
    rust_nbr()
    numpy_nbr()

    def time_func(fn, loops=3):
        t = []
        for _ in range(loops):
            start = time.perf_counter()
            fn()
            t.append(time.perf_counter() - start)
        return np.mean(t), np.min(t), np.max(t)

    ndvi_rust_mean, ndvi_rust_min, ndvi_rust_max = time_func(rust_ndvi)
    ndvi_np_mean, ndvi_np_min, ndvi_np_max = time_func(numpy_ndvi)
    nbr_rust_mean, nbr_rust_min, nbr_rust_max = time_func(rust_nbr)
    nbr_np_mean, nbr_np_min, nbr_np_max = time_func(numpy_nbr)

    print(f"NDVI  Rust mean: {ndvi_rust_mean:.4f}s (min {ndvi_rust_min:.4f} max {ndvi_rust_max:.4f})")
    print(f"NDVI  NumPy mean: {ndvi_np_mean:.4f}s (min {ndvi_np_min:.4f} max {ndvi_np_max:.4f})")
    print(f"NDVI  Speedup (NumPy/Rust): {ndvi_np_mean / ndvi_rust_mean:.2f}x")

    print(f"NBR   Rust mean: {nbr_rust_mean:.4f}s (min {nbr_rust_min:.4f} max {nbr_rust_max:.4f})")
    print(f"NBR   NumPy mean: {nbr_np_mean:.4f}s (min {nbr_np_min:.4f} max {nbr_np_max:.4f})")
    print(f"NBR   Speedup (NumPy/Rust): {nbr_np_mean / nbr_rust_mean:.2f}x")


# ---------------------------------------------------------------------------
# 5. Combined Multi-Band Cube Example
# ---------------------------------------------------------------------------
def multi_band_cube_demo():
    header("MULTI-BAND CUBE DEMO")

    # Simulate a (time, band, y, x) cube for 6 timestamps and bands:
    # Order: [NIR, Red, Green, Blue, SWIR1, SWIR2]
    T, B, H, W = 6, 6, 64, 64
    rng = np.random.default_rng(101)
    cube = rng.uniform(0.05, 0.95, size=(T, B, H, W))

    NIR, RED, GREEN, BLUE, SWIR1, SWIR2 = range(6)

    # Compute indices per time step (vectorized along time axis)
    ndvi_series = np.empty((T, H, W), dtype=np.float64)
    nbr_series = np.empty((T, H, W), dtype=np.float64)
    ndmi_series = np.empty((T, H, W), dtype=np.float64)

    for t in range(T):
        nir = cube[t, NIR]
        red = cube[t, RED]
        green = cube[t, GREEN]
        blue = cube[t, BLUE]
        swir1 = cube[t, SWIR1]
        swir2 = cube[t, SWIR2]

        ndvi_series[t] = ndvi(nir, red)
        nbr_series[t] = nbr(nir, swir2)
        ndmi_series[t] = ndmi(nir, swir1)

    # Simple temporal mean for each index
    mean_ndvi = ndvi_series.mean(axis=0)
    mean_nbr = nbr_series.mean(axis=0)
    mean_ndmi = ndmi_series.mean(axis=0)

    stats("NDVI_mean", mean_ndvi)
    stats("NBR_mean", mean_nbr)
    stats("NDMI_mean", mean_ndmi)

    # Example masked area (simulate cloud or invalid pixels)
    mask = rng.random((H, W)) < 0.1
    mean_ndvi_masked = np.where(mask, np.nan, mean_ndvi)
    print(f"Applied synthetic mask (10% NaNs) to mean NDVI - finite count: {np.isfinite(mean_ndvi_masked).sum()}/{H*W}")


# ---------------------------------------------------------------------------
# 6. Edge Case Demonstrations
# ---------------------------------------------------------------------------
def edge_cases():
    header("EDGE CASES")

    # Near-zero denominators for ND-style indices
    nir = np.array([0.0, 1e-12, 0.3])
    red = np.array([0.0, -1e-12, 0.3])
    ndvi_vals = ndvi(nir, red)
    print("NDVI edge inputs:", ndvi_vals)

    # GCI division by near-zero green -> guarded to 0
    nir2 = np.array([0.5, 0.6, 0.7])
    green2 = np.array([1e-12, 0.3, 0.4])
    gci_vals = gci(nir2, green2)
    print("GCI guarded division:", gci_vals)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    one_dimensional_examples()
    two_dimensional_examples()
    savi_variable_L_demo()
    performance_comparison()
    multi_band_cube_demo()
    edge_cases()
    print("\nAll spectral index extended examples completed successfully.")


if __name__ == "__main__":
    main()
