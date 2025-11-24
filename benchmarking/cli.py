#!/usr/bin/env python3
"""
Minimal benchmarking harness for eo-processor.

This script benchmarks selected Rust-accelerated Earth Observation functions
against representative synthetic data shapes. It reports elapsed time, throughput,
and (optionally) JSON output for downstream analysis.

Supported benchmark targets:
  - spectral: ndvi, ndwi, evi, savi, nbr, ndmi, nbr2, gci, delta_ndvi, delta_nbr, normalized_difference
  - temporal: temporal_mean, temporal_std, median
  - spatial distances: euclidean_distance, manhattan_distance,
                       chebyshev_distance, minkowski_distance

Optional baseline comparison:
  Use --compare-numpy to time an equivalent pure NumPy expression (where feasible)
  and compute a speedup ratio (Rust_mean / NumPy_mean) and include baseline
  statistics in JSON output.

Examples:
  Benchmark all spectral functions on a 4096x4096 image for 3 loops:
    python scripts/benchmark.py --group spectral --height 4096 --width 4096 --loops 3

  Benchmark temporal_mean on a time series (T=24, H=1024, W=1024):
    python scripts/benchmark.py --functions temporal_mean --time 24 --height 1024 --width 1024

  Benchmark distances for two point sets (N=5000, M=5000, D=8):
    python scripts/benchmark.py --group distances --points-a 5000 --points-b 5000 --point-dim 8

  Compare against NumPy:
    python scripts/benchmark.py --group spectral --compare-numpy

  Write JSON results:
    python scripts/benchmark.py --group spectral --json-out benchmark_results.json --compare-numpy

Notes:
  - These are synthetic benchmarks; real-world performance depends on memory bandwidth,
    CPU architecture, NUMA layout, and Dask/XArray orchestration.
  - The Rust kernels release the GIL internally, but this harness runs single-process
    sequential calls for clarity.
  - For fair comparisons, ensure a "warm" cache (initial iteration warms allocations).
  - Baseline NumPy comparison is only available for spectral and temporal functions
    where a straightforward array formula exists.

License: MIT
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

# Enforce single-threaded execution for NumPy to ensure fair comparison
# with the currently single-threaded Rust implementation.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    print("NumPy is required for benchmarking:", exc, file=sys.stderr)
    sys.exit(1)

# Attempt to import optional psutil for memory info
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


# Import eo_processor functions
try:
    import eo_processor
    from eo_processor import (
        chebyshev_distance,
        delta_nbr,
        delta_ndvi,
        euclidean_distance,
        evi,
        gci,
        manhattan_distance,
        median,
        minkowski_distance,
        moving_average_temporal,
        moving_average_temporal_stride,
        nbr,
        nbr2,
        ndmi,
        ndvi,
        ndwi,
        normalized_difference,
        pixelwise_transform,
        savi,
        temporal_mean,
        temporal_std,
    )
    from eo_processor._core import trend_analysis
    from eo_processor import zonal_stats
except ImportError as exc:  # pragma: no cover
    print("Failed to import eo_processor. Have you installed/built it?", exc, file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------------------
# Data Classes
# --------------------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    name: str
    loops: int
    warmups: int
    mean_s: float
    stdev_s: float
    min_s: float
    max_s: float
    throughput_elems: Optional[float]
    elements: Optional[int]
    shape_description: str
    memory_mb: Optional[float]
    baseline_mean_s: Optional[float] = None
    baseline_min_s: Optional[float] = None
    baseline_max_s: Optional[float] = None
    speedup_vs_numpy: Optional[float] = None
    baseline_throughput_elems: Optional[float] = None
    baseline_kind: Optional[str] = None  # e.g. 'broadcast', 'streaming', 'naive', 'prefix'

# --------------------------------------------------------------------------------------
# Argument Parsing
# --------------------------------------------------------------------------------------
def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark eo-processor Rust-accelerated functions."
    )
    parser.add_argument("--compare-numpy", action="store_true",
                        help="Time a NumPy baseline where feasible.")
    parser.add_argument("--functions", nargs="+",
                        help="Explicit list of functions to benchmark (overrides --group).")
    parser.add_argument("--group", choices=["spectral", "temporal", "distances", "processes", "zonal", "morphology", "all"],
                        default="spectral", help="Predefined function group.")
    parser.add_argument("--zones-count", type=int, default=100, help="Number of unique zones for zonal stats.")
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--time", type=int, default=12)
    parser.add_argument("--points-a", type=int, default=2000)
    parser.add_argument("--points-b", type=int, default=2000)
    parser.add_argument("--point-dim", type=int, default=4)
    parser.add_argument("--minkowski-p", type=float, default=3.0)
    parser.add_argument("--ma-window", type=int, default=5)
    parser.add_argument("--ma-stride", type=int, default=4)
    parser.add_argument("--ma-baseline", choices=["naive", "prefix"], default="naive",
                        help="Baseline style for moving averages: naive (O(T*W)) or prefix (O(T)).")
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--md-out", type=str)
    parser.add_argument("--rst-out", type=str)
    parser.add_argument("--size-sweep", nargs="+",
                        help="List of sizes: HxW or T=val:HxW for sweeps.")
    parser.add_argument("--distance-baseline", choices=["broadcast", "streaming", "both"],
                        default="broadcast")
    parser.add_argument("--stress", action="store_true",
                        help="Use larger stress-test sizes.")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------------
def human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    i = 0
    value = float(n)
    while value >= 1024 and i < len(units) - 1:
        value /= 1024
        i += 1
    return f"{value:.2f} {units[i]}"


def current_memory_mb() -> Optional[float]:
    if psutil is None:
        return None
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def time_call(fn: Callable[[], Any]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def compute_elements(func_name: str, shape_info: dict[str, int]) -> Optional[int]:
    """
    Estimate number of scalar elements processed for throughput metrics.
    For distance functions this counts pairwise component operations (N*M*D).
    For moving average stride variant we count the full input processed (T*H*W)
    rather than just output samples, reflecting total arithmetic volume.
    """
    if func_name in {
        "ndvi",
        "ndwi",
        "evi",
        "savi",
        "nbr",
        "ndmi",
        "nbr2",
        "gci",
        "delta_ndvi",
        "delta_nbr",
        "normalized_difference",
    }:
        h, w = shape_info["height"], shape_info["width"]
        return h * w
    if func_name in {"temporal_mean", "temporal_std", "median",
                     "moving_average_temporal", "moving_average_temporal_stride",
                     "pixelwise_transform"}:
        t, h, w = shape_info["time"], shape_info["height"], shape_info["width"]
        return t * h * w
    if func_name == "trend_analysis":
        # trend_analysis operates on 1D list of length T
        return shape_info["time"]
    if func_name in {
        "euclidean_distance",
        "manhattan_distance",
        "chebyshev_distance",
        "minkowski_distance",
    }:
        n, m, d = shape_info["points_a"], shape_info["points_b"], shape_info["point_dim"]
        return n * m * d
    return None


# --------------------------------------------------------------------------------------
# Synthetic Data Factories
# --------------------------------------------------------------------------------------
def make_spectral_inputs(height: int, width: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    nir = rng.uniform(0.2, 0.9, size=(height, width)).astype(np.float64)
    red = rng.uniform(0.05, 0.4, size=(height, width)).astype(np.float64)
    blue = rng.uniform(0.01, 0.25, size=(height, width)).astype(np.float64)
    return nir, red, blue


def make_temporal_stack(time_dim: int, height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(time_dim, height, width)).astype(np.float64)


def make_trend_series(length: int, seed: int) -> List[float]:
    rng = np.random.default_rng(seed)
    # Generate a sample time series with a break (similar to benchmark_trend.py)
    y = np.concatenate([
        np.linspace(0, 10, length // 2),
        np.linspace(10, 0, length // 2)
    ]) + rng.normal(0, 0.5, length)
    return y.tolist()


def make_distance_points(n: int, m: int, dim: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, size=(n, dim)).astype(np.float64)
    b = rng.normal(0.0, 1.0, size=(m, dim)).astype(np.float64)
    return a, b


# --------------------------------------------------------------------------------------
# Benchmark Executor
# --------------------------------------------------------------------------------------
def run_single_benchmark(
    func_name: str,
    loops: int,
    warmups: int,
    shape_info: dict[str, int],
    minkowski_p: float,
    seed: int,
    compare_numpy: bool = False,
    distance_baseline: str = "broadcast",
    name_override: Optional[str] = None,
    ma_window: int = 5,
    ma_stride: int = 4,
    ma_baseline_style: str = "naive",
    zones_count: int = 100,
) -> BenchmarkResult:
    # Predeclare delta arrays to satisfy static type checkers (overwritten when used).
    pre_nir: np.ndarray = np.empty((0, 0))
    pre_red: np.ndarray = np.empty((0, 0))
    post_nir: np.ndarray = np.empty((0, 0))
    post_red: np.ndarray = np.empty((0, 0))
    pre_swir2: np.ndarray = np.empty((0, 0))
    post_swir2: np.ndarray = np.empty((0, 0))
    
    # Initialize baseline variables
    baseline_kind: Optional[str] = None
    baseline_timings: List[float] = []
    supports_baseline = False
    baseline_fn: Optional[Callable[[], Any]] = None
    
    # Prepare inputs
    if func_name in {
        "ndvi",
        "ndwi",
        "evi",
        "savi",
        "nbr",
        "ndmi",
        "nbr2",
        "gci",
        "delta_ndvi",
        "delta_nbr",
        "normalized_difference",
    }:
        nir, red, blue = make_spectral_inputs(shape_info["height"], shape_info["width"], seed)
        if func_name == "ndvi":
            call = lambda: ndvi(nir, red)
        elif func_name == "ndwi":
            call = lambda: ndwi(nir, red)  # using nir as second arg as green is first logically
        elif func_name == "evi":
            call = lambda: evi(nir, red, blue)
        elif func_name == "savi":
            call = lambda: savi(nir, red, L=0.5)
        elif func_name == "nbr":
            swir2 = blue  # using blue as placeholder for swir2
            call = lambda: nbr(nir, swir2)
        elif func_name == "ndmi":
            swir1 = blue  # using blue as placeholder for swir1
            call = lambda: ndmi(nir, swir1)
        elif func_name == "nbr2":
            swir1 = red  # using red as placeholder for swir1
            swir2 = blue  # using blue as placeholder for swir2
            call = lambda: nbr2(swir1, swir2)
        elif func_name == "gci":
            call = lambda: gci(nir, red)
        elif func_name == "delta_ndvi":
            pre_nir, pre_red, _ = make_spectral_inputs(shape_info["height"], shape_info["width"], seed)
            post_nir, post_red, _ = make_spectral_inputs(shape_info["height"], shape_info["width"], seed + 1)
            call = lambda: delta_ndvi(pre_nir, pre_red, post_nir, post_red)
        elif func_name == "delta_nbr":
            pre_nir, _, pre_swir2 = make_spectral_inputs(shape_info["height"], shape_info["width"], seed)
            post_nir, _, post_swir2 = make_spectral_inputs(shape_info["height"], shape_info["width"], seed + 1)
            call = lambda: delta_nbr(pre_nir, pre_swir2, post_nir, post_swir2)
        else:  # normalized_difference
            call = lambda: normalized_difference(nir, red)
        shape_desc = f"{shape_info['height']}x{shape_info['width']}"

    elif func_name in {"temporal_mean", "temporal_std", "median"}:
        cube = make_temporal_stack(shape_info["time"], shape_info["height"], shape_info["width"], seed)
        if func_name == "temporal_mean":
            call = lambda: temporal_mean(cube)
        elif func_name == "temporal_std":
            call = lambda: temporal_std(cube)
        else:
            call = lambda: median(cube)
        shape_desc = f"{shape_info['time']}x{shape_info['height']}x{shape_info['width']}"

    elif func_name == "trend_analysis":
        series = make_trend_series(shape_info["time"], seed)
        # threshold=5.0 from benchmark_trend.py
        call = lambda: trend_analysis(series, threshold=5.0)
        shape_desc = f"T={shape_info['time']}"
    elif func_name in {"moving_average_temporal", "moving_average_temporal_stride", "pixelwise_transform"}:
        cube = make_temporal_stack(shape_info["time"], shape_info["height"], shape_info["width"], seed)
        if func_name == "moving_average_temporal":
            call = lambda: moving_average_temporal(cube, window=ma_window, skip_na=True, mode="same")
        elif func_name == "moving_average_temporal_stride":
            call = lambda: moving_average_temporal_stride(cube, window=ma_window, stride=ma_stride, skip_na=True, mode="same")
        else:  # pixelwise_transform
            call = lambda: pixelwise_transform(cube, scale=1.2, offset=-0.1, clamp_min=0.0, clamp_max=1.0)
        extra = ""
        if func_name.startswith("moving_average"):
            extra = f"(win={ma_window}"
            if func_name == "moving_average_temporal_stride":
                extra += f", stride={ma_stride}"
            extra += ")"
        shape_desc = f"{shape_info['time']}x{shape_info['height']}x{shape_info['width']}{extra}"

    elif func_name in {
        "euclidean_distance",
        "manhattan_distance",
        "chebyshev_distance",
        "minkowski_distance",
    }:
        pts_a, pts_b = make_distance_points(
            shape_info["points_a"], shape_info["points_b"], shape_info["point_dim"], seed
        )
        if func_name == "euclidean_distance":
            call = lambda: euclidean_distance(pts_a, pts_b)
        elif func_name == "manhattan_distance":
            call = lambda: manhattan_distance(pts_a, pts_b)
        elif func_name == "chebyshev_distance":
            call = lambda: chebyshev_distance(pts_a, pts_b)
        else:
            call = lambda: minkowski_distance(pts_a, pts_b, minkowski_p)
        shape_desc = f"N={shape_info['points_a']}, M={shape_info['points_b']}, D={shape_info['point_dim']}"
    elif func_name == "zonal_stats":
        # Generate random values and random zones
        values = np.random.uniform(0, 100, size=(shape_info["height"], shape_info["width"])).astype(np.float64)
        zones = np.random.randint(0, zones_count, size=(shape_info["height"], shape_info["width"]), dtype=np.int64)
        call = lambda: zonal_stats(values, zones)
        shape_desc = f"{shape_info['height']}x{shape_info['width']} (Zones={zones_count})"
        
        if compare_numpy:
            supports_baseline = True
            baseline_kind = "naive_loop"
            # Naive NumPy baseline: iterate unique zones
            def numpy_zonal():
                unique_zones = np.unique(zones)
                res = {}
                for z in unique_zones:
                    mask = (zones == z)
                    z_vals = values[mask]
                    if z_vals.size > 0:
                        res[z] = {
                            "count": z_vals.size,
                            "sum": np.sum(z_vals),
                            "mean": np.mean(z_vals),
                            "min": np.min(z_vals),
                            "max": np.max(z_vals),
                            "std": np.std(z_vals),
                        }
                return res
            baseline_fn = numpy_zonal
    elif func_name in ("binary_dilation", "binary_erosion", "binary_opening", "binary_closing"):
        # Data generation: Binary image (0 or 1)
        # Use uint8 for input as expected by Rust
        data = np.random.randint(0, 2, size=(shape_info["height"], shape_info["width"]), dtype=np.uint8)
        kernel_size = 3
        
        call = lambda: getattr(eo_processor, func_name)(data, kernel_size)
        shape_desc = f"{shape_info['height']}x{shape_info['width']} (Kernel={kernel_size})"
        
        # NumPy baseline (using slicing for vectorization, as scipy might be missing)
        if compare_numpy:
            supports_baseline = True
            baseline_kind = "numpy_slicing"
            def numpy_morph():
                # Naive vectorized implementation using slicing
                # This is O(K*K * N) where K is kernel size
                rows, cols = data.shape
                radius = kernel_size // 2
                
                dilated = None
                eroded = None

                if "dilation" in func_name or "closing" in func_name:
                    # Dilation logic
                    padded = np.pad(data, radius, mode='constant', constant_values=0)
                    out = np.zeros_like(data)
                    for kr in range(kernel_size):
                        for kc in range(kernel_size):
                            # Shift and accumulate
                            out = np.maximum(out, padded[kr:kr+rows, kc:kc+cols])
                    dilated = out
                
                if "erosion" in func_name or "opening" in func_name:
                    # Erosion logic
                    # For binary erosion, padding with 1s is typical to avoid border effects
                    # if the image is mostly 1s. If padded with 0s, erosion at border will be 0.
                    # Let's assume standard behavior for binary images where 0 is background.
                    padded = np.pad(data, radius, mode='constant', constant_values=1) 
                    out = np.ones_like(data)
                    for kr in range(kernel_size):
                        for kc in range(kernel_size):
                            out = np.minimum(out, padded[kr:kr+rows, kc:kc+cols])
                    eroded = out

                if func_name == "binary_dilation":
                    return dilated
                elif func_name == "binary_erosion":
                    return eroded
                elif func_name == "binary_opening":
                    # Erosion then Dilation
                    # Re-run dilation on 'eroded'
                    padded_d = np.pad(eroded, radius, mode='constant', constant_values=0)
                    out_d = np.zeros_like(eroded)
                    for kr in range(kernel_size):
                        for kc in range(kernel_size):
                            out_d = np.maximum(out_d, padded_d[kr:kr+rows, kc:kc+cols])
                    return out_d
                elif func_name == "binary_closing":
                    # Dilation then Erosion
                    # Re-run erosion on 'dilated'
                    padded_e = np.pad(dilated, radius, mode='constant', constant_values=1)
                    out_e = np.ones_like(dilated)
                    for kr in range(kernel_size):
                        for kc in range(kernel_size):
                            out_e = np.minimum(out_e, padded_e[kr:kr+rows, kc:kc+cols])
                    return out_e

            baseline_fn = numpy_morph
    else:  # pragma: no cover
        raise ValueError(f"Unknown function: {func_name}")

    # Warmups
    for _ in range(warmups):
        call()

    if compare_numpy:
        # Provide NumPy baseline implementations where feasible
        if func_name == "ndvi":
            supports_baseline = True
            baseline_fn = lambda: (nir - red) / (nir + red)
        elif func_name == "ndwi":
            supports_baseline = True
            baseline_fn = lambda: (nir - red) / (nir + red)
        elif func_name == "evi":
            supports_baseline = True
            G, C1, C2, L = 2.5, 6.0, 7.5, 1.0
            baseline_fn = lambda: G * (nir - red) / (nir + C1 * red - C2 * blue + L)
        elif func_name == "savi":
            supports_baseline = True
            L = 0.5
            baseline_fn = lambda: (1 + L) * (nir - red) / (nir + red + L)
        elif func_name == "nbr":
            supports_baseline = True
            swir2 = blue  # using blue as placeholder for swir2
            baseline_fn = lambda: (nir - swir2) / (nir + swir2)
        elif func_name == "ndmi":
            supports_baseline = True
            swir1 = blue  # using blue as placeholder for swir1
            baseline_fn = lambda: (nir - swir1) / (nir + swir1)
        elif func_name == "nbr2":
            supports_baseline = True
            swir1 = red  # using red as placeholder for swir1
            swir2 = blue  # using blue as placeholder for swir2
            baseline_fn = lambda: (swir1 - swir2) / (swir1 + swir2)
        elif func_name == "gci":
            supports_baseline = True
            baseline_fn = lambda: (nir / red) - 1.0
        elif func_name == "delta_ndvi":
            supports_baseline = True
            baseline_fn = lambda: ((pre_nir - pre_red) / (pre_nir + pre_red)) - ((post_nir - post_red) / (post_nir + post_red))
        elif func_name == "delta_nbr":
            supports_baseline = True
            baseline_fn = lambda: ((pre_nir - pre_swir2) / (pre_nir + pre_swir2)) - ((post_nir - post_swir2) / (post_nir + post_swir2))
        elif func_name == "normalized_difference":
            supports_baseline = True
            baseline_fn = lambda: (nir - red) / (nir + red)
        elif func_name == "temporal_mean":
            supports_baseline = True
            baseline_fn = lambda: cube.mean(axis=0)
        elif func_name == "temporal_std":
            supports_baseline = True
            baseline_fn = lambda: cube.std(axis=0, ddof=1)
        elif func_name == "median":
            supports_baseline = True
            baseline_fn = lambda: np.median(cube, axis=0)
        elif func_name == "moving_average_temporal":
            supports_baseline = True
            if ma_baseline_style == "naive":
                baseline_kind = "naive"
                # Naive same-mode baseline (variable edges) O(T*W); skip NaN logic mirrored
                def _ma_baseline():
                    arr = cube
                    T = arr.shape[0]
                    half_left = ma_window // 2
                    half_right = ma_window - half_left - 1
                    out = np.empty_like(arr)
                    for t in range(T):
                        start = max(0, t - half_left)
                        end = min(T - 1, t + half_right)
                        window = arr[start : end + 1]
                        # skip_na=True: exclude NaNs
                        valid = window[~np.isnan(window)]
                        if valid.size == 0:
                            out[t] = np.nan
                        else:
                            out[t] = valid.mean(axis=0)
                    return out
                baseline_fn = _ma_baseline
            else:
                baseline_kind = "prefix"
                # Prefix-sum baseline with NaN handling
                def _ma_prefix():
                    arr = cube
                    T = arr.shape[0]
                    # Replace NaNs with 0 for sum; build valid mask
                    valid_mask = ~np.isnan(arr)
                    arr_zero = np.nan_to_num(arr, nan=0.0)
                    csum = np.cumsum(arr_zero, axis=0)
                    ccount = np.cumsum(valid_mask.astype(np.int64), axis=0)
                    out = np.empty_like(arr)
                    half_left = ma_window // 2
                    half_right = ma_window - half_left - 1
                    for t in range(T):
                        start = max(0, t - half_left)
                        end = min(T - 1, t + half_right)
                        total_sum = csum[end] - (csum[start - 1] if start > 0 else 0)
                        total_count = ccount[end] - (ccount[start - 1] if start > 0 else 0)
                        with np.errstate(invalid="ignore", divide="ignore"):
                            out[t] = np.where(total_count > 0, total_sum / total_count, np.nan)
                    return out
                baseline_fn = _ma_prefix
        elif func_name == "moving_average_temporal_stride":
            supports_baseline = True
            def _ma_stride_baseline():
                # Compute naive moving average then stride sample
                arr = cube
                T = arr.shape[0]
                half_left = ma_window // 2
                half_right = ma_window - half_left - 1
                full = []
                for t in range(T):
                    start = max(0, t - half_left)
                    end = min(T - 1, t + half_right)
                    window = arr[start : end + 1]
                    valid = window[~np.isnan(window)]
                    if valid.size == 0:
                        full.append(np.full(arr.shape[1:], np.nan))
                    else:
                        full.append(valid.mean(axis=0))
                full_arr = np.stack(full, axis=0)
                return full_arr[::ma_stride]
            baseline_fn = _ma_stride_baseline
        elif func_name == "pixelwise_transform":
            supports_baseline = True
            baseline_fn = lambda: np.clip(cube * 1.2 - 0.1, 0.0, 1.0)
        # Distance baselines (now enabled for NumPy comparison using vectorized formulations).
        elif func_name == "euclidean_distance":
            supports_baseline = True
            baseline_kind = distance_baseline
            # Broadcast baseline (allocates N x M x D implicitly via math identity)
            broadcast_euclid = lambda: np.sqrt(
                np.clip(
                    (pts_a**2).sum(axis=1)[:, None]
                    + (pts_b**2).sum(axis=1)[None, :]
                    - 2 * (pts_a @ pts_b.T),
                    0.0,
                    None,
                )
            )
            # Streaming baseline (no large 3D temporary; pure Python loop, shows algorithmic parity)
            def streaming_euclid():
                out = np.empty((pts_a.shape[0], pts_b.shape[0]), dtype=np.float64)
                for i in range(pts_a.shape[0]):
                    diff = pts_a[i] - pts_b
                    out[i] = np.sqrt(np.sum(diff * diff, axis=1))
                return out
            baseline_fn = broadcast_euclid if distance_baseline == "broadcast" else streaming_euclid
        elif func_name == "manhattan_distance":
            supports_baseline = True
            baseline_kind = distance_baseline
            broadcast_manhattan = lambda: np.abs(pts_a[:, None, :] - pts_b[None, :, :]).sum(axis=2)
            def streaming_manhattan():
                out = np.empty((pts_a.shape[0], pts_b.shape[0]), dtype=np.float64)
                for i in range(pts_a.shape[0]):
                    diff = np.abs(pts_a[i] - pts_b)
                    out[i] = np.sum(diff, axis=1)
                return out
            baseline_fn = broadcast_manhattan if distance_baseline == "broadcast" else streaming_manhattan
        elif func_name == "chebyshev_distance":
            supports_baseline = True
            baseline_kind = distance_baseline
            broadcast_cheby = lambda: np.abs(pts_a[:, None, :] - pts_b[None, :, :]).max(axis=2)
            def streaming_cheby():
                out = np.empty((pts_a.shape[0], pts_b.shape[0]), dtype=np.float64)
                for i in range(pts_a.shape[0]):
                    diff = np.abs(pts_a[i] - pts_b)
                    out[i] = np.max(diff, axis=1)
                return out
            baseline_fn = broadcast_cheby if distance_baseline == "broadcast" else streaming_cheby
        elif func_name == "minkowski_distance":
            supports_baseline = True
            baseline_kind = distance_baseline
            broadcast_minkowski = lambda: (
                np.abs(pts_a[:, None, :] - pts_b[None, :, :]) ** minkowski_p
            ).sum(axis=2) ** (1.0 / minkowski_p)
            def streaming_minkowski():
                out = np.empty((pts_a.shape[0], pts_b.shape[0]), dtype=np.float64)
                for i in range(pts_a.shape[0]):
                    diff = np.abs(pts_a[i] - pts_b) ** minkowski_p
                    out[i] = np.sum(diff, axis=1) ** (1.0 / minkowski_p)
                return out
            baseline_fn = broadcast_minkowski if distance_baseline == "broadcast" else streaming_minkowski

    # Timed loops
    timings: List[float] = []
    gc.disable()
    try:
        for _ in range(loops):
            gc.collect()
            elapsed = time_call(call)
            timings.append(elapsed)
    finally:
        gc.enable()

    mean_s = statistics.mean(timings)
    stdev_s = statistics.pstdev(timings) if len(timings) > 1 else 0.0
    min_s = min(timings)
    max_s = max(timings)

    elements = compute_elements(func_name, shape_info)
    throughput = elements / mean_s if elements is not None and mean_s > 0 else None

    mem_mb = current_memory_mb()

    baseline_mean = baseline_min = baseline_max = speedup = None
    if supports_baseline and baseline_fn is not None:
        # Baseline warmups
        for _ in range(warmups):
            baseline_fn()
        for _ in range(loops):
            gc.collect()
            baseline_timings.append(time_call(baseline_fn))
        baseline_mean = statistics.mean(baseline_timings)
        baseline_min = min(baseline_timings)
        baseline_max = max(baseline_timings)
        if baseline_mean and mean_s > 0:
            # speedup (baseline_mean / rust_mean) > 1 means Rust faster
            speedup = baseline_mean / mean_s

    # Compute baseline throughput (elements/sec) if we have a NumPy baseline
    baseline_throughput = None
    if supports_baseline and baseline_mean and elements:
        baseline_throughput = elements / baseline_mean if baseline_mean > 0 else None

    return BenchmarkResult(
        name=name_override or func_name,
        loops=loops,
        warmups=warmups,
        mean_s=mean_s,
        stdev_s=stdev_s,
        min_s=min_s,
        max_s=max_s,
        throughput_elems=throughput,
        elements=elements,
        shape_description=shape_desc,
        memory_mb=mem_mb,
        baseline_mean_s=baseline_mean,
        baseline_min_s=baseline_min,
        baseline_max_s=baseline_max,
        speedup_vs_numpy=speedup,
        baseline_throughput_elems=baseline_throughput,
        baseline_kind=baseline_kind,
    )


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def format_result_row(r: BenchmarkResult, compare_numpy: bool = False, show_elements: bool = True, show_shape: bool = True) -> str:
    tput = (
        f"{r.throughput_elems/1e6:.2f}M elems/s"
        if r.throughput_elems is not None
        else "-"
    )
    elem_str = f"{r.elements:,}" if r.elements is not None else "-"
    mem_str = f"{r.memory_mb:.1f} MB" if r.memory_mb is not None else "-"
    row = (
        f"{r.name:22} "
        f"{r.mean_s*1000:9.2f} ms "
        f"{r.stdev_s*1000:7.2f} ms "
        f"{r.min_s*1000:7.2f} ms "
        f"{r.max_s*1000:7.2f} ms "
    )
    if show_elements:
        row += f"{elem_str:>12} "
    
    row += (
        f"{tput:>15} "
        f"{mem_str:>10} "
    )
    
    if compare_numpy:
        if r.baseline_mean_s is not None:
            base_mean = f"{r.baseline_mean_s*1000:9.2f} ms"
            base_tput = (
                f"{r.baseline_throughput_elems/1e6:.2f}M elems/s"
                if r.baseline_throughput_elems
                else "-"
            )
            if r.speedup_vs_numpy >= 1.0:
                speedup = f"{r.speedup_vs_numpy:.2f}x"
            else:
                speedup = f"-{1.0-r.speedup_vs_numpy:.2f}x"
            
            # Calculate throughput difference
            if r.throughput_elems is not None and r.baseline_throughput_elems is not None:
                diff = r.throughput_elems - r.baseline_throughput_elems
                arrow = "↑" if diff >= 0 else "↓"
                diff_str = f"{arrow} {abs(diff)/1e6:.2f}M"
            else:
                diff_str = "-"

            row += f"{base_mean:>12} {base_tput:>15} {speedup:>9} {diff_str:>12} "
        else:
            row += f"{'-':>12} {'-':>15} {'-':>9} {'-':>12} "

    if show_shape:
        row += f"{r.shape_description}"
    
    return row


def print_header(compare_numpy: bool = False, show_elements: bool = True, show_shape: bool = True):
    header = (
        f"{'Function':22} {'Mean':>9}    {'StDev':>7}    {'Min':>7}    {'Max':>7}    "
    )
    if show_elements:
        header += f"{'Elements':>12} "
    
    header += f"{'Throughput':>15} {'RSS Mem':>10} "

    if compare_numpy:
        header += f"{'NumPy Mean':>12} {'NumPy Tput':>15} {'Speedup':>9} {'Tput Diff':>12} "
    
    if show_shape:
        header += "Shape"
    
    print(header)
    print("-" * len(header))
    return len(header)


def resolve_functions(group: str, explicit: Optional[List[str]]) -> List[str]:
    if explicit:
        return explicit
    if group == "spectral":
        return [
            "ndvi",
            "ndwi",
            "evi",
            "savi",
            "nbr",
            "ndmi",
            "nbr2",
            "gci",
            "delta_ndvi",
            "delta_nbr",
            "normalized_difference",
        ]
    if group == "temporal":
        return ["temporal_mean", "temporal_std", "median", "trend_analysis"]
    if group == "distances":
        return [
            "euclidean_distance",
            "manhattan_distance",
            "chebyshev_distance",
            "minkowski_distance",
        ]
    if group == "processes":
        return [
            "moving_average_temporal",
            "moving_average_temporal_stride",
            "pixelwise_transform",
        ]
    if group == "zonal":
        return ["zonal_stats"]
    if group == "all":
        return [
            "ndvi",
            "ndwi",
            "evi",
            "savi",
            "nbr",
            "ndmi",
            "nbr2",
            "gci",
            "delta_ndvi",
            "delta_nbr",
            "normalized_difference",
            "temporal_mean",
            "temporal_std",
            "median",
            "trend_analysis",
            "euclidean_distance",
            "manhattan_distance",
            "chebyshev_distance",
            "minkowski_distance",
            "moving_average_temporal",
            "moving_average_temporal_stride",
            "pixelwise_transform",
            "zonal_stats",
            "binary_dilation",
            "binary_erosion",
            "binary_opening",
            "binary_closing",
        ]
    if group == "morphology":
        return [
            "binary_dilation",
            "binary_erosion",
            "binary_opening",
            "binary_closing",
        ]
    raise ValueError(f"Unknown group: {group}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    funcs = resolve_functions(args.group, args.functions)
    # Stress mode: override sizes to large defaults
    if args.stress:
        # Larger spatial and point set sizes for stress tests (tunable)
        if not args.size_sweep:
            args.height = max(args.height, 4096)
            args.width = max(args.width, 4096)
        args.points_a = max(args.points_a, 10000)
        args.points_b = max(args.points_b, 10000)
        args.point_dim = max(args.point_dim, 16)
        args.time = max(args.time, 48)

    shape_info = {
        "height": args.height,
        "width": args.width,
        "time": args.time,
        "points_a": args.points_a,
        "points_b": args.points_b,
        "point_dim": args.point_dim,
    }

    # Build list of shape infos for size sweep (if requested)
    sweep_specs = args.size_sweep or []
    shape_infos: List[Dict[str, int]] = []
    if sweep_specs:
        for spec in sweep_specs:
            spec = spec.strip()
            if not spec:
                continue
            t_val = args.time
            # Allow T=<time>:HxW pattern
            if "T=" in spec:
                try:
                    t_part, rest = spec.split(":", 1)
                    t_val = int(t_part.split("=", 1)[1])
                    spec = rest
                except Exception:
                    raise ValueError(f"Invalid size sweep temporal syntax: {spec}")
            if "x" not in spec.lower():
                raise ValueError(f"Invalid size sweep entry (expected HxW): {spec}")
            h_str, w_str = spec.lower().split("x", 1)
            try:
                h_val = int(h_str)
                w_val = int(w_str)
            except ValueError:
                raise ValueError(f"Non-integer size components in sweep entry: {spec}")
            shape_infos.append(
                {
                    "height": h_val,
                    "width": w_val,
                    "time": t_val,
                    "points_a": args.points_a,
                    "points_b": args.points_b,
                    "point_dim": args.point_dim,
                }
            )
    else:
        shape_infos.append(shape_info)

    results: List[BenchmarkResult] = []
    for shp in shape_infos:
        for f in funcs:
            # For distance functions with --distance-baseline=both, run twice.
            is_distance = f in {
                "euclidean_distance",
                "manhattan_distance",
                "chebyshev_distance",
                "minkowski_distance",
            }
            if is_distance and args.distance_baseline == "both":
                for mode in ("broadcast", "streaming"):
                    res = run_single_benchmark(
                        func_name=f,
                        loops=args.loops,
                        warmups=args.warmups,
                        shape_info=shp,
                        minkowski_p=args.minkowski_p,
                        seed=args.seed,
                        compare_numpy=args.compare_numpy,
                        distance_baseline=mode,
                        name_override=f"{f}[{mode}]",
                    )
                    results.append(res)
            else:
                res = run_single_benchmark(
                    func_name=f,
                    loops=args.loops,
                    warmups=args.warmups,
                    shape_info=shp,
                    minkowski_p=args.minkowski_p,
                    seed=args.seed,
                    compare_numpy=args.compare_numpy,
                    distance_baseline=args.distance_baseline,
                    name_override=None,
                    ma_window=args.ma_window,
                    ma_stride=args.ma_stride,
                )
                results.append(res)

    if not args.quiet:
        print()
        print("eo-processor Benchmark Results")
        print("=" * 34)
        print(f"Python: {platform.python_version()}  Platform: {platform.platform()}")
        print(f"Loops: {args.loops}  Warmups: {args.warmups}  Seed: {args.seed}")
        
        # Log threading configuration
        thread_vars = [
            "OMP_NUM_THREADS", 
            "MKL_NUM_THREADS", 
            "OPENBLAS_NUM_THREADS", 
            "VECLIB_MAXIMUM_THREADS", 
            "NUMEXPR_NUM_THREADS"
        ]
        env_settings = [f"{var}={os.environ.get(var, 'Not Set')}" for var in thread_vars]
        print(f"Threading: {', '.join(env_settings)}")
        
        print(f"Group: {args.group}  Functions: {', '.join(funcs)}")
        
        # Check uniformity of elements
        all_elements = [r.elements for r in results if r.elements is not None]
        unique_elements = set(all_elements)
        uniform_elements = len(unique_elements) == 1
        elements_val = all_elements[0] if uniform_elements and all_elements else None
        
        if uniform_elements and elements_val is not None:
             print(f"Elements: {elements_val:,}")

        # Check uniformity of shape
        all_shapes = [r.shape_description for r in results if r.shape_description is not None]
        unique_shapes = set(all_shapes)
        uniform_shapes = len(unique_shapes) == 1
        shape_val = all_shapes[0] if uniform_shapes and all_shapes else None

        if uniform_shapes and shape_val is not None:
             print(f"Shape: {shape_val}")

        print()
        header_len = print_header(args.compare_numpy, show_elements=not uniform_elements, show_shape=not uniform_shapes)
        for r in results:
            print(format_result_row(r, args.compare_numpy, show_elements=not uniform_elements, show_shape=not uniform_shapes))
        print("-" * header_len)
        print("Throughput reported as processed elements per second (approximation).")
        print()

    if args.json_out:
        payload = {
            "meta": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "loops": args.loops,
                "warmups": args.warmups,
                "seed": args.seed,
                "group": args.group,
                "functions": funcs,
                "shape_info": shape_info,
                "size_sweep": args.size_sweep,
                "sweep_shape_infos": shape_infos if args.size_sweep else None,
            },
            "results": [asdict(r) for r in results],
            "compare_numpy": args.compare_numpy,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        if not args.quiet:
            print(f"Wrote JSON results to: {args.json_out}")
    # Precompute meta_rows (unconditional) so both md_out and rst_out blocks can use it
    meta_rows = {
        "Python": platform.python_version(),
        "Platform": platform.platform(),
        "Group": args.group,
        "Functions": ", ".join(funcs),
        "Distance Baseline": args.distance_baseline,
        "Stress Mode": str(args.stress),
        "Loops": str(args.loops),
        "Warmups": str(args.warmups),
        "Seed": str(args.seed),
        "Compare NumPy": str(args.compare_numpy),
        "Height": str(shape_info["height"]),
        "Width": str(shape_info["width"]),
        "Time": str(shape_info["time"]),
        "Points A": str(shape_info["points_a"]),
        "Points B": str(shape_info["points_b"]),
        "Point Dim": str(shape_info["point_dim"]),
        "Size Sweep": str(args.size_sweep),
        "MA Window": str(args.ma_window),
        "MA Stride": str(args.ma_stride),
        "MA Baseline": args.ma_baseline,
        "Zones Count": str(args.zones_count),
    }

    if getattr(args, "md_out", None):
        # Build Markdown (GitHub-style) report
        lines = []
        lines.append(f"# eo-processor Benchmark Report")
        lines.append("")
        lines.append("## Meta")
        lines.append("")
        lines.append("| Key | Value |")
        lines.append("|-----|-------|")
        for k, v in meta_rows.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
        lines.append("## Results")
        lines.append("")
        lines.append("| Function | Mean (ms) | StDev (ms) | Min (ms) | Max (ms) | Elements | Rust Throughput (M elems/s) | NumPy Throughput (M elems/s) | Speedup vs NumPy | Shape |")
        lines.append("|----------|-----------|------------|----------|----------|----------|------------------------|------------------|-------|")
        for r in results:
            mean_ms = r.mean_s * 1000
            stdev_ms = r.stdev_s * 1000
            min_ms = r.min_s * 1000
            max_ms = r.max_s * 1000
            elems = f"{r.elements:,}" if r.elements is not None else "-"
            tput = f"{(r.throughput_elems/1e6):.2f}" if r.throughput_elems is not None else "-"
            speedup = f"{r.speedup_vs_numpy:.2f}x" if r.speedup_vs_numpy is not None else "-"
            btput = f"{(r.baseline_throughput_elems/1e6):.2f}" if r.baseline_throughput_elems is not None else "-"
            lines.append(f"| {r.name} | {mean_ms:.2f} | {stdev_ms:.2f} | {min_ms:.2f} | {max_ms:.2f} | {elems} | {tput} | {btput} | {speedup} | {r.shape_description} |")
        lines.append("")
        if args.compare_numpy and r.baseline_kind is not None:
            lines.append("> Speedup vs NumPy = (NumPy mean time / Rust mean time); values > 1 indicate Rust is faster.")
            if r.baseline_kind:
                lines.append(f"> NumPy baseline kind used: {r.baseline_kind}.")
        with open(args.md_out, "w", encoding="utf-8") as f_md:
            f_md.write("\n".join(lines))
        if not args.quiet:
            print(f"Wrote Markdown report to: {args.md_out}")

    # Optional Sphinx reST output (grid tables) if --rst-out was provided.
    if getattr(args, "rst_out", None):
        rst = []
        rst.append("Benchmark Report")
        rst.append("================")
        rst.append("")
        rst.append("Meta")
        rst.append("----")
        # Meta as simple definition list
        for k, v in meta_rows.items():
            rst.append(f"{k}: {v}")
        rst.append("")
        rst.append("Results")
        rst.append("-------")
        # Build grid table
        header_cols = [
            "Function",
            "Mean (ms)",
            "StDev (ms)",
            "Min (ms)",
            "Max (ms)",
            "Elements",
            "Rust Throughput (M elems/s)",
            "NumPy Throughput (M elems/s)",
            "Speedup vs NumPy",
            "Shape",
        ]
        # Determine column widths
        rows = []
        for r in results:
            mean_ms = f"{r.mean_s*1000:.2f}"
            stdev_ms = f"{r.stdev_s*1000:.2f}"
            min_ms = f"{r.min_s*1000:.2f}"
            max_ms = f"{r.max_s*1000:.2f}"
            elems = f"{r.elements:,}" if r.elements is not None else "-"
            tput = f"{(r.throughput_elems/1e6):.2f}" if r.throughput_elems is not None else "-"
            btput = f"{(r.baseline_throughput_elems/1e6):.2f}" if r.baseline_throughput_elems is not None else "-"
            speedup = f"{r.speedup_vs_numpy:.2f}x" if r.speedup_vs_numpy is not None else "-"
            rows.append([r.name, mean_ms, stdev_ms, min_ms, max_ms, elems, tput, btput, speedup, r.shape_description])

        # Compute column widths
        col_widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(header_cols)]

        def grid_sep(char="="):
            return "+" + "+".join(char * (w + 2) for w in col_widths) + "+"

        def grid_row(values):
            return "|" + "|".join(f" {v}{' ' * (w - len(v))} " for v, w in zip(values, col_widths)) + "|"

        # Header
        rst.append(grid_sep("="))
        rst.append(grid_row(header_cols))
        rst.append(grid_sep("="))
        # Data rows
        for row in rows:
            rst.append(grid_row(row))
        rst.append(grid_sep("="))
        rst.append("")
        if args.compare_numpy:
            rst.append("Speedup vs NumPy = (NumPy mean time / Rust mean time); values > 1 indicate Rust is faster.")
            rst.append("")
        with open(args.rst_out, "w", encoding="utf-8") as f_rst:
            f_rst.write("\n".join(rst))
        if not args.quiet:
            print(f"Wrote reST report to: {args.rst_out}")

        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
