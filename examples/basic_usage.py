"""
Basic usage examples for eo-processor.

This script demonstrates how to use the Rust-accelerated UDFs for
Earth Observation computations.
"""

import numpy as np
from eo_processor import (
    ndvi,
    ndwi,
    normalized_difference,
    enhanced_vegetation_index as evi,
    temporal_mean,
    temporal_std,
    median,
    composite,
)

# Example 1: Computing NDVI with 1D arrays
print("Example 1: NDVI with 1D arrays")
print("-" * 40)
nir_1d = np.array([0.8, 0.7, 0.6, 0.5, 0.4])
red_1d = np.array([0.2, 0.1, 0.3, 0.2, 0.1])
ndvi_result = ndvi(nir_1d, red_1d)
print(f"NIR:  {nir_1d}")
print(f"Red:  {red_1d}")
print(f"NDVI: {ndvi_result}")
print()

# Example 2: Computing NDVI with 2D arrays (image-like)
print("Example 2: NDVI with 2D arrays (100x100 image)")
print("-" * 40)
nir_2d = np.random.rand(100, 100) * 0.8 + 0.2  # NIR values between 0.2 and 1.0
red_2d = np.random.rand(100, 100) * 0.4        # Red values between 0.0 and 0.4
ndvi_2d_result = ndvi(nir_2d, red_2d)
print(f"NIR shape:  {nir_2d.shape}")
print(f"Red shape:  {red_2d.shape}")
print(f"NDVI shape: {ndvi_2d_result.shape}")
print(f"NDVI min:   {ndvi_2d_result.min():.4f}")
print(f"NDVI max:   {ndvi_2d_result.max():.4f}")
print(f"NDVI mean:  {ndvi_2d_result.mean():.4f}")
print()

# Example 3: Computing NDWI (water index)
print("Example 3: NDWI (Normalized Difference Water Index)")
print("-" * 40)
green = np.array([0.3, 0.4, 0.5, 0.6])
nir = np.array([0.2, 0.1, 0.3, 0.2])
ndwi_result = ndwi(green, nir)
print(f"Green: {green}")
print(f"NIR:   {nir}")
print(f"NDWI:  {ndwi_result}")
print()

# Example 4: NDWI with 2D arrays
print("Example 4: NDWI with 2D arrays (50x50)")
print("-" * 40)
green_2d = np.random.rand(50, 50) * 0.5 + 0.1   # Green between 0.1 and 0.6
nir_2d = np.random.rand(50, 50) * 0.4 + 0.1     # NIR between 0.1 and 0.5
ndwi_2d = ndwi(green_2d, nir_2d)
print(f"NDWI shape: {ndwi_2d.shape}")
print(f"NDWI stats -> min: {ndwi_2d.min():.4f} max: {ndwi_2d.max():.4f} mean: {ndwi_2d.mean():.4f}")
print()

# Example 5: Enhanced Vegetation Index (EVI) 1D
print("Example 5: Enhanced Vegetation Index (EVI) 1D")
print("-" * 40)
nir_evi_1d = np.array([0.6, 0.7, 0.5])
red_evi_1d = np.array([0.3, 0.2, 0.25])
blue_evi_1d = np.array([0.1, 0.05, 0.08])
evi_1d = evi(nir_evi_1d, red_evi_1d, blue_evi_1d)
print(f"NIR: {nir_evi_1d}")
print(f"Red: {red_evi_1d}")
print(f"Blue:{blue_evi_1d}")
print(f"EVI: {evi_1d}")
print()

# Example 6: Enhanced Vegetation Index (EVI) 2D
print("Example 6: Enhanced Vegetation Index (EVI) 2D (60x60)")
print("-" * 40)
nir_evi_2d = np.random.rand(60, 60) * 0.6 + 0.2   # 0.2 - 0.8
red_evi_2d = np.random.rand(60, 60) * 0.4 + 0.1   # 0.1 - 0.5
blue_evi_2d = np.random.rand(60, 60) * 0.2 + 0.05 # 0.05 - 0.25
evi_2d = evi(nir_evi_2d, red_evi_2d, blue_evi_2d)
print(f"EVI shape: {evi_2d.shape}")
print(f"EVI stats -> min: {evi_2d.min():.4f} max: {evi_2d.max():.4f} mean: {evi_2d.mean():.4f}")
print()

# Example 7: Generic normalized difference
print("Example 7: Generic normalized difference")
print("-" * 40)
a = np.array([0.9, 0.8, 0.7])
b = np.array([0.1, 0.2, 0.3])
nd_result = normalized_difference(a, b)
print(f"A:  {a}")
print(f"B:  {b}")
print(f"ND: {nd_result}")
print()

# Example 8: Performance comparison (streamlined)
print("Example 8: Performance (1000x1000 NDVI)")
print("-" * 40)
import time
size = 1000
nir_large = np.random.rand(size, size)
red_large = np.random.rand(size, size)

t0 = time.time()
ndvi_rust = ndvi(nir_large, red_large)
t_rust = time.time() - t0

t0 = time.time()
ndvi_numpy = (nir_large - red_large) / (nir_large + red_large)
t_numpy = time.time() - t0

print(f"Rust:  {t_rust*1000:.2f} ms  NumPy: {t_numpy*1000:.2f} ms  Speedup: {t_numpy/t_rust:.2f}x  Match: {np.allclose(ndvi_rust, ndvi_numpy, rtol=1e-10)}")
print()

# Example 9: Temporal statistics & median composite
print("Example 9: Temporal statistics & composite (time,y,x = 12,128,128)")
print("-" * 40)
ts = np.random.rand(12, 128, 128)
mean_img = temporal_mean(ts)
std_img = temporal_std(ts)
median_img = median(ts)
composite_img = composite(ts, method="median")

print(f"mean shape:    {mean_img.shape}, std shape: {std_img.shape}")
print(f"median shape:  {median_img.shape}, composite (median) identical: {np.allclose(median_img, composite_img)}")
print(f"mean range:    [{mean_img.min():.4f}, {mean_img.max():.4f}]")
print(f"std range:     [{std_img.min():.4f}, {std_img.max():.4f}]")
print(f"median range:  [{median_img.min():.4f}, {median_img.max():.4f}]")
