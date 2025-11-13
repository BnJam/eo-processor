temporal_std
============

.. currentmodule:: eo_processor

.. autofunction:: temporal_std

Overview
--------
`temporal_std` computes the sample standard deviation (n - 1 divisor) across the leading *time* axis
of arrays with 1–4 dimensions:

Supported input shapes (time-first):
- 1D: (time,)
- 2D: (time, band)
- 3D: (time, y, x)
- 4D: (time, band, y, x)

NaN Handling
------------
By default `skip_na=True`, which excludes NaNs from each per‑pixel time series before computing
the mean and variance. If fewer than 2 valid (non-NaN) samples remain, the output for that element
is set to NaN. Setting `skip_na=False` propagates NaNs: if any NaN occurs in the series the result
becomes NaN.

Computation Details
-------------------
For each spatial (and optional band) position the function:
1. Extracts the 1D time series vector
2. Optionally filters NaNs
3. Computes mean
4. Computes unbiased variance: sum((x - mean)^2) / (n - 1)
5. Takes square root for the standard deviation

All arithmetic is performed in `float64` for numerical stability regardless of the original dtype.

Performance Notes
-----------------
- Implemented in Rust (PyO3) with optional parallel iteration (Rayon) for 3D / 4D arrays across spatial pixels.
- Fused per‑pixel time series traversal (extract → mean → variance accumulation → sqrt) minimizes temporaries.
- Single dtype coercion (float64) ensures stable arithmetic and consistent NaN handling.
- Parallel thresholds avoid spawning threads for small workloads (reduces overhead).

Performance (Representative Benchmarks)
---------------------------------------
Single-run measurements (macOS ARM64, CPython 3.10, release build, `time.perf_counter()`, float64 data, warm cache):

.. list-table:: Temporal Std Benchmark (Single Run)
   :header-rows: 1

   * - Shape (time, y, x)
     - Description
     - Rust (s)
     - NumPy (s)
     - Rust Throughput (M elems/s)
     - NumPy Throughput (M elems/s)
     - Speedup
   * - (24, 1024, 1024)
     - Medium cube
     - 0.319
     - 0.114
     - 77.34
     - 216.65
     - 0.36x
   * - (24, 2000, 2000)
     - Large cube
     - (benchmark)
     - (benchmark)
     - (benchmark)
     - (benchmark)
     - (pending)

(Example shows a case where NumPy is faster for the specific medium size; larger spatial domains or different time lengths may shift relative performance. Always benchmark on your target workload.)

Reproduction Snippet:
.. code-block:: python

    import numpy as np, time
    from eo_processor import temporal_std

    cube = np.random.rand(24, 1024, 1024)
    t0 = time.perf_counter(); rust_out = temporal_std(cube); rust_t = time.perf_counter() - t0
    t0 = time.perf_counter(); numpy_out = np.nanstd(cube, axis=0, ddof=1); numpy_t = time.perf_counter() - t0
    print(f"Rust {rust_t:.3f}s vs NumPy {numpy_t:.3f}s speedup {numpy_t/rust_t:.2f}x")
    assert np.allclose(rust_out, numpy_out, atol=1e-9)

Performance Claim Template:
.. code-block:: text

    Benchmark:
    Shape: (24, 1024, 1024)
    NumPy nanstd(ddof=1): 0.114s
    Rust temporal_std: 0.319s
    Speedup: 0.36x (NumPy faster for this shape)
    Methodology: single run, time.perf_counter(), float64 arrays
    Validation: np.allclose(..., atol=1e-9)

Guidance:
- If your pipeline already uses other Rust-accelerated reducers (median, indices), keeping std in Rust can simplify multi-core scheduling and reduce Python dispatch overhead in aggregate.
- For very large spatial grids or longer time axes, parallel scaling may reduce Rust runtime relative to NumPy; confirm with local benchmarking.
- Consider external chunking (Dask/XArray) to distribute work across processes; each worker invokes the Rust kernel GIL-free.

Returns
-------
The time dimension is removed:
- Input (T,) → scalar `float64`
- Input (T, B) → shape (B,)
- Input (T, Y, X) → shape (Y, X)
- Input (T, B, Y, X) → shape (B, Y, X)

Examples
--------
Basic usage with a 3D time series cube:

.. code-block:: python

    import numpy as np
    from eo_processor import temporal_std

    # (time=5, y=2, x=2)
    cube = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, 3.0], [4.0, 5.0]],
        [[3.0, 4.0], [5.0, 6.0]],
        [[4.0, 5.0], [6.0, 7.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ])
    std_img = temporal_std(cube)
    print(std_img.shape)  # (2, 2)

Handling NaNs:

.. code-block:: python

    import numpy as np
    from eo_processor import temporal_std

    series = np.array([1.0, np.nan, 3.0, 4.0])
    out_skip = temporal_std(series, skip_na=True)   # Uses [1.0, 3.0, 4.0]
    out_prop = temporal_std(series, skip_na=False)  # NaN propagates → NaN

Edge Cases
----------
- All-NaN time series → NaN
- Single valid sample → NaN (variance undefined)
- Mixed integer/float input dtypes → coerced to float64 internally

See Also
--------
- `temporal_mean` for mean across time axis
- `median` for robust median compositing
- `composite` wrapper (currently median only)

End of `temporal_std` reference.
