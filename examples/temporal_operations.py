"""
Temporal operations examples for eo-processor.
This script demonstrates how to use the Rust-accelerated temporal functions
and compares their performance to NumPy.
"""

import numpy as np
import time
from eo_processor import temporal_mean, temporal_std, median, composite

# Example 1: Temporal mean with a 3D array (time, y, x)
print("Example 1: Temporal mean with a 3D array")
print("-" * 40)
data_3d = np.random.rand(10, 100, 100).astype(np.float64)
mean_rust = temporal_mean(data_3d)
mean_numpy = np.mean(data_3d, axis=0)
print(f"Rust implementation shape: {mean_rust.shape}")
print(f"NumPy implementation shape: {mean_numpy.shape}")
print(f"Results match: {np.allclose(mean_rust, mean_numpy)}")
print()

# Example 2: Temporal standard deviation with a 4D array (time, band, y, x)
print("Example 2: Temporal standard deviation with a 4D array")
print("-" * 40)
data_4d = np.random.rand(10, 3, 100, 100).astype(np.float64)
std_rust = temporal_std(data_4d)
std_numpy = np.std(data_4d, axis=0, ddof=1)
print(f"Rust implementation shape: {std_rust.shape}")
print(f"NumPy implementation shape: {std_numpy.shape}")
print(f"Results match: {np.allclose(std_rust, std_numpy)}")
print()

# Example 3: Performance comparison for temporal_mean
print("Example 3: Performance comparison for temporal_mean (100x512x512 array)")
print("-" * 40)
large_array = np.random.rand(100, 512, 512).astype(np.float64)

# Rust implementation
start_rust = time.time()
mean_rust_large = temporal_mean(large_array)
time_rust = time.time() - start_rust

# NumPy implementation
start_numpy = time.time()
mean_numpy_large = np.mean(large_array, axis=0)
time_numpy = time.time() - start_numpy

print(f"Rust implementation:  {time_rust*1000:.2f} ms")
print(f"NumPy implementation: {time_numpy*1000:.2f} ms")
print(f"Speedup: {time_numpy/time_rust:.2f}x")
print()

# Example 4: Performance comparison for temporal_std
print("Example 4: Performance comparison for temporal_std (100x512x512 array)")
print("-" * 40)

# Rust implementation
start_rust = time.time()
std_rust_large = temporal_std(large_array)
time_rust = time.time() - start_rust

# NumPy implementation
start_numpy = time.time()
std_numpy_large = np.std(large_array, axis=0, ddof=1)
time_numpy = time.time() - start_numpy

print(f"Rust implementation:  {time_rust*1000:.2f} ms")
print(f"NumPy implementation: {time_numpy*1000:.2f} ms")
print(f"Speedup: {time_numpy/time_rust:.2f}x")

# Example 5: NaN handling & temporal composite (median)
print("Example 5: NaN handling & temporal composite")
print("-" * 40)
# Build a (time, y, x) cube with some NaNs
nan_cube = np.random.rand(6, 32, 32).astype(np.float64)
mask = np.random.rand(6, 32, 32) < 0.15
nan_cube[mask] = np.nan

mean_skip = temporal_mean(nan_cube, skip_na=True)
mean_strict = temporal_mean(nan_cube, skip_na=False)
std_skip = temporal_std(nan_cube, skip_na=True)
median_comp = median(nan_cube)
composite_median = composite(nan_cube, method="median")

print(f"skip_na mean NaN count: {np.isnan(mean_skip).sum()}")
print(f"strict mean NaN count:  {np.isnan(mean_strict).sum()}")
print(f"std skip_na NaN count:  {np.isnan(std_skip).sum()}")
print(f"median == composite(median): {np.allclose(median_comp, composite_median, equal_nan=True)}")
print(f"median range: [{np.nanmin(median_comp):.4f}, {np.nanmax(median_comp):.4f}]")
print()
