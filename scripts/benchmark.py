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
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

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
    parser.add_argument("--group", choices=["spectral", "temporal", "distances", "processes", "all"],
                        default="spectral", help="Predefined function group.")
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
    parser.add_argument("--loops", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
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
    for _ in range(loops):
        elapsed = time_call(call)
        timings.append(elapsed)

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
def format_result_row(r: BenchmarkResult) -> str:
    tput = (
        f"{r.throughput_elems/1e6:.2f}M elems/s"
        if r.throughput_elems is not None
        else "-"
    )
    elem_str = f"{r.elements:,}" if r.elements is not None else "-"
    mem_str = f"{r.memory_mb:.1f} MB" if r.memory_mb is not None else "-"
    return (
        f"{r.name:22} "
        f"{r.mean_s*1000:9.2f} ms "
        f"{r.stdev_s*1000:7.2f} ms "
        f"{r.min_s*1000:7.2f} ms "
        f"{r.max_s*1000:7.2f} ms "
        f"{elem_str:>12} "
        f"{tput:>15} "
        f"{mem_str:>10} "
        f"{r.shape_description}"
    )


def print_header():
    print(
        f"{'Function':22} {'Mean':>9} {'StDev':>7} {'Min':>7} {'Max':>7} "
        f"{'Elements':>12} {'Throughput':>15} {'RSS Mem':>10} {'Shape'}"
    )
    print("-" * 115)


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
        return ["temporal_mean", "temporal_std", "median"]
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
            "euclidean_distance",
            "manhattan_distance",
            "chebyshev_distance",
            "minkowski_distance",
            "moving_average_temporal",
            "moving_average_temporal_stride",
            "pixelwise_transform",
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
        print(f"Group: {args.group}  Functions: {', '.join(funcs)}")
        print()
        print_header()
        for r in results:
            extra = ""
            if args.compare_numpy and r.baseline_mean_s is not None:
                bt = f"{r.baseline_throughput_elems/1e6:.2f}M elems/s" if r.baseline_throughput_elems else "-"
                extra = f" | NumPy mean: {r.baseline_mean_s*1000:.2f} ms NumPy throughput: {bt} speedup: {r.speedup_vs_numpy:.2f}x"
            print(f"{format_result_row(r)}{extra}")
        print("-" * 115)
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
