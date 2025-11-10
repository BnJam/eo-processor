#!/usr/bin/env python3
"""
Minimal benchmarking harness for eo-processor.

This script benchmarks selected Rust-accelerated Earth Observation functions
against representative synthetic data shapes. It reports elapsed time, throughput,
and (optionally) JSON output for downstream analysis.

Supported benchmark targets:
  - spectral: ndvi, ndwi, evi, savi, nbr, ndmi, nbr2, gci, normalized_difference
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
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
        ndvi,
        ndwi,
        evi,
        normalized_difference,
        temporal_mean,
        temporal_std,
        median,
        euclidean_distance,
        manhattan_distance,
        chebyshev_distance,
        minkowski_distance,
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
    throughput_elems: Optional[float]  # elements/sec
    elements: Optional[int]
    shape_description: str
    memory_mb: Optional[float]
    # Optional NumPy baseline metrics (present when --compare-numpy used and function supports baseline)
    baseline_mean_s: Optional[float] = None
    baseline_min_s: Optional[float] = None
    baseline_max_s: Optional[float] = None
    speedup_vs_numpy: Optional[float] = None  # baseline_mean_s / mean_s (values >1 mean Rust faster)


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


def time_call(fn: Callable[[], None]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def compute_elements(func_name: str, shape_info: Dict[str, int]) -> Optional[int]:
    """
    Estimate number of elements processed for throughput metrics.
    """
    if func_name in {"ndvi", "ndwi", "evi", "normalized_difference"}:
        h, w = shape_info["height"], shape_info["width"]
        return h * w
    if func_name in {"temporal_mean", "temporal_std", "median"}:
        t, h, w = shape_info["time"], shape_info["height"], shape_info["width"]
        return t * h * w
    if func_name in {
        "euclidean_distance",
        "manhattan_distance",
        "chebyshev_distance",
        "minkowski_distance",
    }:
        n, m = shape_info["points_a"], shape_info["points_b"]
        return n * m
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
    shape_info: Dict[str, int],
    minkowski_p: float,
    seed: int,
    compare_numpy: bool = False,
) -> BenchmarkResult:
    # Prepare inputs
    if func_name in {"ndvi", "ndwi", "evi", "normalized_difference"}:
        nir, red, blue = make_spectral_inputs(shape_info["height"], shape_info["width"], seed)
        if func_name == "ndvi":
            call = lambda: ndvi(nir, red)
        elif func_name == "ndwi":
            call = lambda: ndwi(nir, red)  # using nir as second arg as green is first logically
        elif func_name == "evi":
            call = lambda: evi(nir, red, blue)
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

    baseline_timings: List[float] = []
    supports_baseline = False
    baseline_fn: Optional[Callable[[], None]] = None

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
        # Distance baselines would require pairwise loops; skip for fairness/performance.

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

    return BenchmarkResult(
        name=func_name,
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


# --------------------------------------------------------------------------------------
# Argument Parsing
# --------------------------------------------------------------------------------------
def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark eo-processor Rust-accelerated functions."
    )
    parser.add_argument(
        "--compare-numpy",
        action="store_true",
        help="Also time a pure NumPy baseline (supported for spectral + temporal functions).",
    )
    parser.add_argument(
        "--functions",
        nargs="+",
        help="Explicit list of functions to benchmark (overrides --group).",
    )
    parser.add_argument(
        "--group",
        choices=["spectral", "temporal", "distances", "all"],
        default="spectral",
        help="Predefined function group to benchmark.",
    )
    parser.add_argument("--height", type=int, default=2048, help="Spatial height.")
    parser.add_argument("--width", type=int, default=2048, help="Spatial width.")
    parser.add_argument("--time", type=int, default=12, help="Temporal length (for temporal functions).")
    parser.add_argument("--points-a", type=int, default=2000, help="Number of points in set A for distances.")
    parser.add_argument("--points-b", type=int, default=2000, help="Number of points in set B for distances.")
    parser.add_argument("--point-dim", type=int, default=4, help="Dimensionality of point space (D).")
    parser.add_argument("--minkowski-p", type=float, default=3.0, help="Order p for Minkowski distance.")
    parser.add_argument("--loops", type=int, default=3, help="Number of timed loops.")
    parser.add_argument("--warmups", type=int, default=1, help="Number of warmup runs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--json-out",
        type=str,
        help="Write benchmark results to JSON file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress table output (still writes JSON if requested).",
    )
    parser.add_argument(
        "--md-out",
        type=str,
        help="Write benchmark results as a Markdown table to the specified file.",
    )
    return parser.parse_args(argv)


def resolve_functions(group: str, explicit: Optional[List[str]]) -> List[str]:
    if explicit:
        return explicit
    if group == "spectral":
        return ["ndvi", "ndwi", "evi", "savi", "nbr", "ndmi", "nbr2", "gci", "normalized_difference"]
    if group == "temporal":
        return ["temporal_mean", "temporal_std", "median"]
    if group == "distances":
        return [
            "euclidean_distance",
            "manhattan_distance",
            "chebyshev_distance",
            "minkowski_distance",
        ]
    if group == "all":
        return [
            "ndvi",
            "ndwi",
            "evi",
            "normalized_difference",
            "temporal_mean",
            "temporal_std",
            "median",
            "euclidean_distance",
            "manhattan_distance",
            "chebyshev_distance",
            "minkowski_distance",
        ]
    raise ValueError(f"Unknown group: {group}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    funcs = resolve_functions(args.group, args.functions)

    shape_info = {
        "height": args.height,
        "width": args.width,
        "time": args.time,
        "points_a": args.points_a,
        "points_b": args.points_b,
        "point_dim": args.point_dim,
    }

    results: List[BenchmarkResult] = []
    for f in funcs:
        res = run_single_benchmark(
            func_name=f,
            loops=args.loops,
            warmups=args.warmups,
            shape_info=shape_info,
            minkowski_p=args.minkowski_p,
            seed=args.seed,
            compare_numpy=args.compare_numpy,
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
                extra = f" | NumPy mean: {r.baseline_mean_s*1000:.2f} ms speedup: {r.speedup_vs_numpy:.2f}x"
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
            },
            "results": [asdict(r) for r in results],
            "compare_numpy": args.compare_numpy,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        if not args.quiet:
            print(f"Wrote JSON results to: {args.json_out}")
    if getattr(args, "md_out", None):
        # Build Markdown report
        lines = []
        lines.append(f"# eo-processor Benchmark Report")
        lines.append("")
        lines.append("## Meta")
        lines.append("")
        lines.append("| Key | Value |")
        lines.append("|-----|-------|")
        meta_rows = {
            "Python": platform.python_version(),
            "Platform": platform.platform(),
            "Group": args.group,
            "Functions": ", ".join(funcs),
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
        }
        for k, v in meta_rows.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
        lines.append("## Results")
        lines.append("")
        lines.append("| Function | Mean (ms) | StDev (ms) | Min (ms) | Max (ms) | Elements | Throughput (M elems/s) | Speedup vs NumPy | Shape |")
        lines.append("|----------|-----------|------------|----------|----------|----------|------------------------|------------------|-------|")
        for r in results:
            mean_ms = r.mean_s * 1000
            stdev_ms = r.stdev_s * 1000
            min_ms = r.min_s * 1000
            max_ms = r.max_s * 1000
            elems = f"{r.elements:,}" if r.elements is not None else "-"
            tput = f"{(r.throughput_elems/1e6):.2f}" if r.throughput_elems is not None else "-"
            speedup = f"{r.speedup_vs_numpy:.2f}x" if r.speedup_vs_numpy is not None else "-"
            lines.append(f"| {r.name} | {mean_ms:.2f} | {stdev_ms:.2f} | {min_ms:.2f} | {max_ms:.2f} | {elems} | {tput} | {speedup} | {r.shape_description} |")
        lines.append("")
        if args.compare_numpy:
            lines.append("> Speedup vs NumPy = (NumPy mean time / Rust mean time); values > 1 indicate Rust is faster.")
        with open(args.md_out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        if not args.quiet:
            print(f"Wrote Markdown report to: {args.md_out}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
