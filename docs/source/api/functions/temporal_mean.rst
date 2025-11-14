temporal_mean
==============

.. currentmodule:: eo_processor
.. autofunction:: temporal_mean

Overview
--------
`temporal_mean` computes the arithmetic mean along the leading (time) axis of an array.
It supports 1D, 2D, 3D, and 4D inputs with shapes interpreted as:

- 1D: ``(time,)`` → result is a scalar (``float64``)
- 2D: ``(time, feature)`` → result shape ``(feature,)``
- 3D: ``(time, y, x)`` → result shape ``(y, x)``
- 4D: ``(time, band, y, x)`` → result shape ``(band, y, x)``

NaN Handling
------------
The parameter ``skip_na`` controls how NaNs are treated:

- ``skip_na=True`` (default): NaNs are excluded from the mean calculation for each pixel location.
  If all values at that location are NaN, the output is set to NaN.
- ``skip_na=False``: Any NaN in the series forces the output at that location to NaN.

Internal Behavior
-----------------
- Inputs of any numeric dtype (int/uint/float) are coerced to ``float64`` once in the Rust layer.
- Parallelization (Rayon) is applied for 3D/4D arrays over spatial indices for improved throughput on large grids.
- For 1D and 2D arrays, simple sequential Rust loops avoid unnecessary parallel overhead.

Parameters
----------
See the generated signature above; primary arguments:

``arr`` : ``numpy.ndarray``
    Input time-first array (1D–4D).
``skip_na`` : ``bool``, default ``True``
    Whether to ignore NaNs when computing the mean.

Returns
-------
``numpy.ndarray`` or ``float``
    Mean with time axis removed. Scalar for 1D input; array otherwise. Output dtype is always ``float64``.

Edge Cases
----------
- All-NaN time series at a location → output NaN (when ``skip_na=True``).
- Series length < 1 (empty input) is not supported and will raise an error upstream.
- Single valid value among NaNs → output that value (with ``skip_na=True``).

Example (1D)
------------
.. code-block:: python

    import numpy as np
    from eo_processor import temporal_mean

    ts = np.array([1.0, 2.0, np.nan, 5.0])
    m = temporal_mean(ts)              # skip_na=True by default
    print(m)  # (1 + 2 + 5) / 3 = 8/3 ≈ 2.6666667

Example (3D)
------------
.. code-block:: python

    import numpy as np
    from eo_processor import temporal_mean

    cube = np.random.rand(12, 256, 256)         # (time, y, x)
    mean_img = temporal_mean(cube)              # shape (256, 256)

Example (NaN Propagation)
-------------------------
.. code-block:: python

    import numpy as np
    from eo_processor import temporal_mean

    arr = np.array([[1.0, np.nan],
                    [3.0, 4.0],
                    [np.nan, 2.0]])  # shape (time, feature)
    # Column 0: [1.0, 3.0, NaN] -> mean = 2.0
    # Column 1: [NaN, 4.0, 2.0] -> mean = 3.0
    out_skip = temporal_mean(arr, skip_na=True)   # array([2.0, 3.0])
    out_noprop = temporal_mean(arr, skip_na=False) # NaNs present -> array([nan, nan])

Performance Notes
-----------------
- For large 3D/4D arrays, per-pixel series extraction and mean accumulation run in native Rust without Python GIL contention.
- Parallel iteration threshold heuristics avoid excessive overhead on small arrays.

Performance (Representative Benchmarks)
---------------------------------------
Unlike some other reducers (e.g., median), `temporal_mean` does not always outperform NumPy on every shape. For moderate array sizes NumPy’s highly optimized C loops plus contiguous memory access can match or exceed the Rust implementation, especially when parallel thresholds intentionally avoid spawning threads to reduce overhead.

Single-run measurements (macOS ARM64, CPython 3.10, release build, `time.perf_counter()`, float64 data, warm cache):

.. list-table:: Temporal Mean Benchmark (Single Run)
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
     - 0.276
     - 0.078
     - 91.17
     - 322.63
     - 0.28x
   * - (24, 2000, 2000)
     - Large cube
     - 0.891
     - 0.474
     - 107.77
     - 202.53
     - 0.53x

Interpretation:
- Current parallel heuristics favor avoiding overhead on medium-sized grids; NumPy can be faster.
- For substantially larger spatial domains or when integrating with other Rust kernels in a pipeline, total end-to-end throughput may still benefit from uniform Rust execution.
- Further tuning (e.g., adjusting parallel thresholds or adding adaptive chunking) can improve relative performance; such changes will be documented with updated benchmarks.

Reproduction Snippet:
.. code-block:: python

    import numpy as np, time
    from eo_processor import temporal_mean

    cube = np.random.rand(24, 2000, 2000)
    t0 = time.perf_counter()
    rust_out = temporal_mean(cube)
    rust_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    numpy_out = np.nanmean(cube, axis=0)
    numpy_t = time.perf_counter() - t0

    print(f"Rust {rust_t:.3f}s vs NumPy {numpy_t:.3f}s speedup {numpy_t/rust_t:.2f}x")
    assert np.allclose(rust_out, numpy_out, atol=1e-12)

Performance Claim Template:
.. code-block:: text

    Benchmark:
    Shape: (24, 2000, 2000)
    NumPy nanmean: 0.474s
    Rust temporal_mean: 0.891s
    Speedup: 0.53x (NumPy faster for this shape)
    Methodology: single run, time.perf_counter(), float64 arrays
    Validation: np.allclose(..., atol=1e-12)

Guidance:
- Benchmark on your target workload (time axis length, spatial size) before selecting an implementation.
- If most of your pipeline already uses Rust-accelerated functions, keeping the mean in Rust can simplify threading and reduce Python call overhead in aggregate.
- Consider external chunking (Dask/XArray) to exploit multi-worker parallelism; each worker invokes the Rust kernel without holding the GIL.


Numerical Stability
-------------------
- All intermediate sums use ``float64``.
- No special epsilon handling is required for mean (unlike normalized difference denominators).

Related Functions
-----------------
- :func:`temporal_std` sample standard deviation (ddof=1) along the time axis.
- :func:`median` temporal median with optional NaN skipping.
- :func:`composite` convenience wrapper (currently median only).

When to Use
-----------
Use `temporal_mean` to reduce a temporal stack to a representative average surface or per-band summary, especially prior to:
- Change detection baselining
- Feature engineering for machine learning
- Seasonal or annual aggregation steps

Limitations
-----------
- Does not provide weighted mean; for weighting logic you would need a custom wrapper.
- Does not internally chunk extremely large time dimensions; consider external chunking (e.g., Dask) for memory management.

End of temporal_mean reference.
