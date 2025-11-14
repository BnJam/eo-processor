========================================
Architecture & Performance Model
========================================

Overview
--------
`eo-processor` accelerates Earth Observation (EO) numeric primitives by implementing
user‑defined functions (UDFs) in Rust and exposing them to Python via `PyO3`.
This page details the architectural choices that yield consistent speedups over
pure Python / NumPy workflows while preserving safety, determinism, and API clarity.

Design Objectives
-----------------
1. Eliminate Python GIL contention for CPU-bound loops.
2. Provide dimensional dispatch (1D–4D) without Python-level branching overhead.
3. Preserve numerical stability (uniform float64 internal type + epsilon guards).
4. Minimize intermediate allocations (single-pass or fused operations where feasible).
5. Ensure compile-time memory safety (no `unsafe` blocks).
6. Fit seamlessly into XArray / Dask pipelines as ufunc-like callables.
7. Keep error messages explicit (shape mismatches, invalid parameter values).
8. Offer predictable scaling thresholds for parallel execution (avoid micro-task overhead).

Rust Extension Boundary
-----------------------
The compiled module (`eo_processor._core`) is a native extension created by `maturin`.
Key steps at build/import time:

1. Rust crates (`ndarray`, `numpy`, `rayon`, `pyo3`) compile into a shared object.
2. Functions annotated with `#[pyfunction]` are registered inside the `#[pymodule] fn _core(...)`.
3. Python entrypoints in `python/eo_processor/__init__.py` provide thin wrappers (and aliases like `evi`).
4. Autodoc and type stub (`__init__.pyi`) expose signatures to editors and static type checkers.

Dimensional Dispatch
--------------------
Each computational family (indices, temporal, spatial, masking) inspects the input array rank
inside Rust:

- Spectral indices: support 1D or 2D (and some 3D/4D variants for batch/time-first layouts).
- Temporal statistics: 1D–4D; interpret leading axis as time.
- Median compositing: specialized 1D–4D implementations to avoid generalized recursion.
- Masking utilities: unified loops over arbitrary rank (explicit 1D–4D branches).

This avoids repeated Python `if arr.ndim == ...` checks and reduces interpreter overhead.

Numerical Stability & Coercion
------------------------------
All incoming numeric dtypes (int, uint, float32, float64) are coerced to `f64` once.
Benefits:

- Single arithmetic path (no branching for dtype).
- Consistent floating precision for downstream calculations.
- Simplifies NaN handling (uses IEEE-754 semantics uniformly).

Small denominators in normalized difference style formulas are guarded with an `EPSILON`
constant to prevent division blow-ups; near-zero denominators default to 0 or neutral outcomes
as documented in docstrings.

Parallel Execution Strategy
---------------------------
Selective Rayon usage:

- Temporal loops: parallel over spatial indices (pixels) after slicing time series.
- Pairwise distances: parallel over outer dimension only when `N * M` exceeds a threshold.
- Median/Std/Mean: parallel chunking for 3D / 4D arrays.

Why thresholds? Aggressive parallelism on small arrays causes overhead that erodes gains.
Threshold tuning (e.g., `10_000` distance pair cutoff) can be adjusted with empirical benchmarking.

Memory Efficiency
-----------------
Common Python/NumPy pattern:
    (nir - red) / (nir + red)

Creates two temporaries unless optimized; Rust versions combine operations in a single loop.
Similarly, `ΔNDVI` and `ΔNBR` compute pre+post indices in fused passes (one set of ephemeral values).
Median computations limit series extraction to exact pixel slices, sorting compact vectors.

NaN Handling
------------
Temporal and median functions optionally `skip_na=True`:

- Filter NaNs before statistical aggregation.
- If all samples are NaN, result is NaN (propagate emptiness semantics).
- Avoids NumPy masked array overhead and branch-heavy Python loops.

Error Handling
--------------
Rust raises Python exceptions with explicit context:

- Shape mismatch (e.g., spectral band arrays differ in shape).
- Invalid parameter (e.g., Minkowski `p < 1.0`).
- Unsupported dimensionality (non 1D–4D arrays trigger `TypeError`).
- SCL masking receives invalid codes gracefully; missing code sets output to NaN.

Performance Comparative Summary
-------------------------------
Category           | Rust UDF Advantage | Fairness Caveat
------------------ | ------------------ | ---------------
Spectral Indices   | Fused arithmetic + GIL release reduce temporaries. | NumPy already vectorized; gains typically modest (≤1.3×) unless very large tiles.
Temporal Stats     | Parallel spatial reduction (Rayon) amortizes time axis iteration. | Small grids may see overhead; benchmark your T,Y,X before adopting.
Median Composite   | Native per‑pixel nth‑selection (planned) + parallel iteration. | Current implementation still sorts entire series; optimization ongoing.
Pairwise Distances | Streaming algorithm avoids allocating N×M×D broadcasts. | A “broadcast” NumPy baseline does extra work; compare both broadcast & streaming baselines.
Masking Utilities  | Single pass with minimal Python overhead. | If mask condition is simple, NumPy boolean indexing can match performance.

Important: A headline speedup >2× usually indicates an algorithmic difference (e.g., avoiding large broadcasts) rather than lower constant factors alone. Always verify with size sweeps.

Integration with XArray / Dask
------------------------------
Use `xarray.apply_ufunc` with `dask="parallelized"`:

.. code-block:: python

    import xarray as xr
    import dask.array as da
    from eo_processor import ndvi

    nir = xr.DataArray(da.random.random((6000, 6000), chunks=(750, 750)), dims=["y", "x"])
    red = xr.DataArray(da.random.random((6000, 6000), chunks=(750, 750)), dims=["y", "x"])

    ndvi_xr = xr.apply_ufunc(
        ndvi,
        nir,
        red,
        dask="parallelized",
        output_dtypes=[float],
    )
    out = ndvi_xr.compute()

The Rust function executes inside each Dask worker on chunk data without holding the GIL.
For large temporal stacks (time-first), chunk on spatial dimensions to maximize per-task parallelism.

Extensibility Guidelines
------------------------
When adding a new index or transform:

1. Implement in a domain-specific module (e.g., `indices.rs`).
2. Provide dimensional dispatch for relevant rank(s).
3. Add input shape assertions early.
4. Add parallelization only if O(N) or O(N*M) complexity is high enough.
5. Update Python exports + stubs + README + Sphinx autosummary list.
6. Add tests (shape mismatch, NaN scenario, typical value range).
7. Benchmark if claiming performance improvement (>20% threshold preferred).

Testing & Validation
--------------------
Rust unit tests validate core arithmetic and shape behavior.
Python tests (in `tests/`) confirm wrapper semantics and dtype coercion.
Coverage badge should be regenerated after logic additions.

Representative Benchmark Template & Empirical Results
-----------------------------------------------------

Template
~~~~~~~~
.. code-block:: python

    import numpy as np, time
    from eo_processor import temporal_mean

    cube = np.random.rand(24, 1024, 1024)  # (time, y, x)
    t0 = time.perf_counter()
    rust_mean = temporal_mean(cube)
    rust_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    numpy_mean = np.nanmean(cube, axis=0)
    numpy_t = time.perf_counter() - t0

    print(f"Rust {rust_t:.3f}s vs NumPy {numpy_t:.3f}s (speedup {numpy_t / rust_t:.2f}x)")
    assert np.allclose(rust_mean, numpy_mean, atol=1e-12)

Methodology
~~~~~~~~~~~
- Platform: macOS (ARM64), CPython 3.10, release build of the Rust extension.
- Timings: single run wall-clock using time.perf_counter(), warm cache.
- Dtypes: float64 inputs (internal coercion matches typical use).
- Validation: np.allclose(...) with tight tolerances (1e-12–1e-9).
- Disclaimer: Results vary with CPU topology, memory bandwidth, and array shapes.
  Always benchmark on your target workload before making pipeline decisions.

Representative Results
~~~~~~~~~~~~~~~~~~~~~~
Large-array benchmarks (single run; no warm repeats):

+---------------------+----------------------------+-----------+-----------+-----------+
| Function            | Input Shape                | Rust (s)  | NumPy (s) | Speedup   |
+=====================+============================+===========+===========+===========+
| ndvi                | (5000, 5000)               | 0.080     | 0.112     | 1.40x     |
| median (temporal)   | (15, 2000, 2000)           | 1.313     | 1.947     | 1.48x     |
| temporal_mean       | (24, 2000, 2000)           | 0.891     | 0.474     | 0.53x     |
| euclidean_distance  | (3000, 64) × (3000, 64)    | 0.063     | 0.041     | 0.65x     |
+---------------------+----------------------------+-----------+-----------+-----------+

Interpretation
~~~~~~~~~~~~~~
- Fused Index Arithmetic: ndvi shows a clear gain from single-pass fused math + GIL release.
- Temporal Median: Sorting per pixel in native code + parallel spatial iteration yields
  >1.4× improvement versus NumPy’s nanmedian for this shape.
- Temporal Mean / Std: For medium-sized grids Rust may be slower if parallel thresholds
  trigger overhead. Larger time dimensions or spatial extents generally improve relative
  performance; tune chunking (e.g., via Dask) for best results.
- Pairwise Distances: For moderately sized matrices NumPy’s optimized BLAS-backed
  matrix multiply strategy can outperform the current Rust implementation. Rust gains
  appear for larger N×M where outer-loop parallelization amortizes overhead.

Guidance
~~~~~~~~
- Use Rust spectral indices (ndvi, nbr, etc.) for production-scale tile processing.
- Prefer the Rust median for large temporal stacks when robustness to outliers matters.
- Benchmark temporal_mean / temporal_std on your exact sizes; consider increasing
  spatial dimensions or enabling chunked execution for better scaling.
- For very large pairwise distance workloads (> millions of pairs) the Rust implementation
  can scale with additional cores; otherwise NumPy's vectorized BLAS approach may suffice.

Performance Claim Template
~~~~~~~~~~~~~~~~~~~~~~~~~~
When adding new optimized functions, follow the required documentation pattern:

.. code-block:: text

    Benchmark:
    Array size: 5000 x 5000
    Old: 1.42s
    New: 1.05s
    Speedup: 1.35x

    Methodology:
    Single run, warm cache discarded
    Timing via time.perf_counter()
    Validation: np.allclose(new, old, atol=1e-12)

Keep empirical claims conservative and reproducible. Attach code snippets or
reference a test harness (noting environment) for transparency.

Key Differences from Pure NumPy
-------------------------------
- Fewer temporaries in multi-step formulas.
- Parallelism not limited by GIL.
- Consistent float64 internal representation.
- Explicit shape checks produce clearer early errors than silent broadcasting misalignments.
- Integrated NaN filtering in statistical functions (no separate boolean masks required).

Future Optimization Opportunities
---------------------------------
(Subject to human review per repository guidelines)

- Adaptive parallel thresholds informed by runtime profiling.
- Optional streaming median (reducing per-pixel vector allocation for long time series).
- SIMD acceleration (requires careful evaluation; staying within safe Rust & avoiding `unsafe`).
- Advanced mask composition DSL (predicate fusion to reduce data passes).

Security & Safety Notes
-----------------------
- No `unsafe` code blocks in kernel implementations.
- No network or file I/O in the performance-critical path.
- All external inputs constrained to NumPy array interfaces; memory safety enforced by borrow semantics.

Reference of Core Modules
-------------------------
Module        | Purpose
------------- | --------------------------------------------
`indices`     | Spectral & change detection indices (NDVI, NDWI, SAVI, EVI, NBR, NDMI, NBR2, GCI, ΔNDVI, ΔNBR)
`temporal`    | Time-axis statistics (mean, std) with NaN skipping
`spatial`     | Median compositing + pairwise distance calculations
`masking`     | Value/range/sentinel/SCL masking utilities
`lib.rs`      | PyO3 module registration (bridge to Python)

Rust Acceleration Note
----------------------
The high-level rationale for Rust UDF acceleration (GIL release, dimensional dispatch, float64 coercion, parallel strategy) is also summarized on the main index page. For convenience:

.. admonition:: Summary
   :class: tip

   - GIL-free native loops enable true multi-core throughput.
   - Float64 coercion yields stable, predictable numerics.
   - Selective Rayon parallelism avoids overhead on small workloads.
   - Fused loop bodies minimize intermediate allocations.
   - Dimension-aware implementations remove Python branching overhead.
   - No `unsafe`; memory safety guaranteed by compiler.

Cross-References
----------------
- Function Index: :doc:`functions`
- README Overview: :doc:`../README`
- Quick Start: :doc:`../QUICKSTART`
- Contribution Guide: External link in navigation.

License
-------
MIT. See repository root for full text.

Benchmark Report (Generated)
----------------------------
The following benchmark summary (generated via ``scripts/benchmark.py``) is included for up-to-date performance measurements across spectral indices and related functions:

.. include:: benchmarks.rst
.. include:: dist_bench.rst

Python Version Benchmarks
-------------------------
The following per-interpreter benchmark reports are generated via version-specific tox
benchmark environments (py38-bench … py313-bench). Each file captures synthetic performance
snapshots for that Python version using identical data shapes and loop counts.

.. include:: benchmark-py38.rst
.. include:: benchmark-py39.rst
.. include:: benchmark-py310.rst
.. include:: benchmark-py311.rst
.. include:: benchmark-py312.rst
.. include:: benchmark-py313.rst

Benchmark Fairness & Methodology
--------------------------------
To ensure transparent comparisons we distinguish two baseline styles:

1. Broadcast Baseline (NumPy):
   - Uses broadcasting to form (N, M, D) intermediary (explicitly or implicitly via algebra).
   - High memory footprint; stresses allocator & cache.
2. Streaming Baseline (NumPy):
   - Pure Python loop over outer dimension; each iteration performs a vectorized reduction over D.
   - Low memory; more Python overhead.

Rust distance functions currently implement a streaming pattern with parallel outer-loop execution (when size threshold exceeded). When reporting speedups:
speedup_vs_numpy = baseline_mean / rust_mean
Values > 1 mean Rust faster. A large speedup versus the broadcast baseline may disappear or shrink versus the streaming baseline.

Reporting Guidelines:
- Always state which baseline was used (broadcast / streaming / both).
- Include array or point set shape and element count.
- Use ≥5 timing loops + ≥1 warmup; prefer median over mean if distribution skewed.
- Disclose algorithmic differences (e.g., fewer temporaries) rather than attributing all gains to “Rust”.
- For spectral indices, expect near bandwidth-limited behavior; huge (>1.5×) gains are unlikely on modern NumPy unless parallelism or algorithmic fusion changes.

Stress Benchmarks
-----------------
Stress benchmarks highlight scalability for Earth Observation “big tile” or large point set workloads (e.g., 4096×4096 spatial grids, time depth ≥48, distance matrices with N,M ≥ 10,000). Enable via:
  scripts/benchmark.py --group all --stress --compare-numpy --distance-baseline both --loops 7 --warmups 2 --size-sweep 2048x2048 4096x4096

Outputs should be included as generated reST fragments:
.. include:: stress-benchmarks.rst

Interpreting Stress Results:
- Spectral indices: near-linear scaling with element count until memory bandwidth saturates.
- Temporal reducers: benefit from more spatial pixels (parallel work units) and longer time axis (amortized slice overhead).
- Distances: Rust streaming avoids allocating (N,M,D) memory; broadcast NumPy baseline may degrade sharply with very large N,M,D.

Planned Improvements (Benchmarking):
- Add nth‑selection median to reduce O(T log T) sorts to O(T).
- Adaptive parallel threshold auto-tuned from calibration run.
- Optional SIMD path (still safe Rust) after empirical validation.

Documentation Update Protocol:
Whenever a new optimization lands:
1. Re-run size sweep (small, medium, large shapes).
2. Update fairness table (above) if algorithmic class changes.
3. Regenerate stress-benchmarks.rst.
4. Provide before/after code snippet and numerical equivalence check (np.allclose with tolerance).
5. Bump minor version if public API extended; patch if internal only.

End of Architecture Overview.
