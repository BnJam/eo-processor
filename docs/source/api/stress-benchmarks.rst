Stress Benchmark Report
=======================

This document is an auto-included placeholder for high-scale ("stress") benchmark
results demonstrating how `eo-processor` behaves on large Earth Observation (EO)
workloads (multi-thousand pixel tiles, deep temporal stacks, and large pairwise
distance matrices). Replace the placeholder sections after running the
benchmark harness with the `--stress` flag and appropriate size sweep.

Generation Command (Example)
----------------------------
Run on a suitably provisioned host (≥16 cores, high memory bandwidth):

.. code-block:: bash

    # Spectral + temporal + distance functions, large sizes, both distance baselines
    python scripts/benchmark.py \
        --group all \
        --stress \
        --compare-numpy \
        --distance-baseline both \
        --loops 7 \
        --warmups 2 \
        --size-sweep 2048x2048 4096x4096 T=48:4096x4096 \
        --rst-out docs/source/api/stress-benchmarks.rst

Replace or extend `--size-sweep` for your target data volume. For distance tests,
`--stress` internally scales point counts (e.g. sets N,M ≥ 10,000). Adjust if memory
constraints require smaller sets.

Methodology Summary
-------------------
- Warmups: 2 (cache warming + JIT / allocation priming)
- Timed loops: 7 (mean + median recommended for skewed distributions)
- Speedup Formula: NumPy mean time / Rust mean time (values > 1 ⇒ Rust faster)
- Baselines (distance):
  - broadcast: allocates or implicitly materializes an (N, M, D) intermediate
  - streaming: Python loop over rows, vectorized per-row differences (no giant temp)
- Validation: `np.allclose(rust, numpy, atol=1e-12)` (or tighter where numerically stable)
- Environment variables: record CPU model, core count, memory size, Python version.

Interpretation Guidelines
-------------------------
- Spectral indices: Expect near memory-bandwidth limits; improvements mainly from fused arithmetic.
- Temporal reducers (mean/std): Parallelism helps as spatial size increases; small grids may underutilize threads.
- Median: Large time axis + large spatial grid magnifies native advantages (pending nth-selection optimization).
- Distances: Streaming Rust avoids enormous temporaries; fairness reported with both baseline strategies.
- Extremely large N×M distance benchmarks may exceed RAM in broadcast mode—those failures should be documented, not hidden.

Placeholder Results
-------------------
(Replace the following list-table with actual generated data.)

.. list-table::
   :header-rows: 1
   :widths: 18 14 10 10 10 12 14 16

   * - Function
     - Shape / Params
     - Rust Mean (ms)
     - NumPy Mean (ms)
     - Speedup (x)
     - Elements (≈)
     - Baseline Type
     - Notes
   * - ndvi
     - 4096x4096
     - TBD
     - TBD
     - TBD
     - 16,777,216
     - vectorized
     - Fused arithmetic
   * - temporal_mean
     - 48x4096x4096
     - TBD
     - TBD
     - TBD
     - 805,306,368
     - axis=0 reduce
     - Parallel spatial iteration
   * - median
     - 36x4096x4096
     - TBD
     - TBD
     - TBD
     - 604,241,920
     - axis=0 median
     - Full sort (opt planned)
   * - euclidean_distance[broadcast]
     - N=10000,M=10000,D=16
     - TBD
     - TBD
     - TBD
     - 100,000,000
     - broadcast
     - High memory pressure
   * - euclidean_distance[streaming]
     - N=10000,M=10000,D=16
     - TBD
     - TBD
     - TBD
     - 100,000,000
     - streaming
     - Low memory; parallelizable
   * - minkowski_distance[broadcast]
     - N=8000,M=8000,D=16,p=3
     - TBD
     - TBD
     - TBD
     - 64,000,000
     - broadcast
     - Alloc temp; exponent p
   * - minkowski_distance[streaming]
     - N=8000,M=8000,D=16,p=3
     - TBD
     - TBD
     - TBD
     - 64,000,000
     - streaming
     - Reduced memory footprint

Stress Benchmark Caveats
------------------------
- If any broadcast distance benchmark triggers MemoryError / OOM, document the failure and emphasize the streaming alternative.
- For massive temporal stacks, per-pixel allocations (median) can dominate; future optimization will reduce sort complexity.
- Speedups <1 (Rust slower) for smaller shapes are not regressions—overhead dominates when work units are tiny.

How to Update This File
-----------------------
1. Run the stress benchmark command.
2. Replace the "Placeholder Results" table with generated data (ensure Sphinx-compatible formatting).
3. Add a short narrative below interpreting any extreme speedups (e.g. >3×) and explicitly noting algorithmic differences.
4. Commit with message: `docs(benchmarks): update stress benchmark results` (or `perf` if tied to new code).
5. Regenerate coverage badge if tests changed.

Next Optimization Targets (Tracking)
------------------------------------
- Streaming median with select_nth_unstable to avoid full sort.
- Tiled + SIMD distance kernels (still safe Rust).
- Adaptive parallel threshold for small/medium arrays.
- Optional multi-dtype path with specialized kernels (evaluate benefit vs complexity).

Validation Snippet (Template)
-----------------------------
.. code-block:: python

    import numpy as np
    from eo_processor import ndvi
    nir = np.random.rand(4096, 4096)
    red = np.random.rand(4096, 4096)
    rust_out = ndvi(nir, red)
    np_out = (nir - red) / (nir + red)
    assert np.allclose(rust_out, np_out, atol=1e-12)

End of Stress Benchmark Placeholder
-----------------------------------
Replace TBD entries after executing the benchmark harness. Ensure reproducibility notes
(environment, CPU model, core count) accompany final numbers.
