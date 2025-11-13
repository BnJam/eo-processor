.. _median:

median
======

.. currentmodule:: eo_processor

.. autofunction:: median

Overview
--------
`median` computes the per‑pixel (or per element) median along the leading *time* axis
of arrays with 1–4 dimensions:

Supported input layouts (time-first):
- 1D: (time,)
- 2D: (time, band|feature)
- 3D: (time, y, x)
- 4D: (time, band, y, x)

The output removes the leading time axis and preserves the remaining spatial / band axes:
- 1D input → scalar (float64)
- 2D input → (band,)
- 3D input → (y, x)
- 4D input → (band, y, x)

NaN Handling
------------
By default (`skip_na=True`) NaNs are excluded from the median calculation for each pixel / band.
If all values for a pixel along time are NaN (after skipping), the result is NaN.
If `skip_na=False`, any NaN in the time series forces the output at that position to NaN.

Performance Characteristics
---------------------------
The function is implemented in Rust with specialized dispatch for each supported rank:
- 1D & 2D: sequential loops (small overhead; no parallelism needed).
- 3D & 4D: parallel iteration over spatial indices (rows/columns or band/rows/columns) using rayon.
Each pixel’s time series is copied into a temporary vector, optionally filtered for NaNs, sorted,
and the median selected (middle element or average of the two middle elements for even-length series).

Memory Notes:
- Sorting per pixel allocates a vector of length = time dimension.
- There are no large intermediate arrays created; processing is per pixel.
- Internal working dtype is float64 (inputs are coerced once for stability).

Parameters
----------
arr : numpy.ndarray
    Time-first array of rank 1–4 with any numeric dtype (int/uint/float). Coerced to float64 internally.
skip_na : bool, default True
    Whether to ignore NaN values in each time series. If False, any NaN in the time series forces the output at that position to NaN.

Returns
-------
numpy.ndarray
    Array with time axis removed; dtype float64.

Raises
------
TypeError
    If `arr` is not a 1D–4D numeric NumPy array after coercion.
ValueError
    (Indirectly) for internal shape inconsistencies (rare; indicates malformed input).

Examples
--------
1D (simple series):
    >>> import numpy as np
    >>> from eo_processor import median
    >>> series = np.array([1.0, 5.0, 3.0, 2.0])
    >>> median(series)
    2.5  # average of (2.0,3.0) after sorting → [1,2,3,5]

2D (time, band):
    >>> cube = np.array([[1, 10],
    ...                  [4,  2],
    ...                  [3,  8]], dtype=float)
    >>> median(cube)
    array([3., 8.])  # medians per band

3D (time, y, x) with NaNs:
    >>> stack = np.array([
    ...     [[1.0,  np.nan],
    ...      [5.0,  4.0]],
    ...     [[2.0,  7.0],
    ...      [np.nan, 6.0]],
    ...     [[3.0,  8.0],
    ...      [2.0,  5.0]],
    ... ])
    >>> median(stack)  # skip_na=True (default)
    array([[2., 7.],
           [3., 5.]])
    >>> median(stack, skip_na=False)
    array([[2., nan],
           [nan, 5.]])

4D (time, band, y, x):
    >>> arr4d = np.random.rand(6, 2, 4, 4)  # 6 time steps
    >>> med = median(arr4d)
    >>> med.shape
    (2, 4, 4)

Comparison to NumPy
-------------------
NumPy’s `np.nanmedian` does not directly support a “time-first multi-rank specialized parallel loop”.
This implementation:
- Avoids Python overhead for per-pixel extraction loops.
- Selectively parallelizes only spatial axes (not the time axis) for better cache behavior.
Use `np.nanmedian(arr, axis=0)` for correctness comparison when `skip_na=True`.

Integration With Compositing
----------------------------
`composite(arr, method="median")` is a thin wrapper around `median`. Future compositing
methods (e.g., percentiles) will follow the same shape semantics.

Performance
-----------
Benchmark (single run; macOS ARM64, CPython 3.10, release build – time.perf_counter):

.. list-table:: Median Benchmark (Single Run)
   :header-rows: 1

   * - Time Series Shape
     - Rust (s)
     - NumPy (s)
     - Rust Throughput (M elems/s)
     - NumPy Throughput (M elems/s)
     - Speedup
   * - (15, 1024, 1024)
     - 0.336
     - 0.507
     - 46.81
     - 31.01
     - 1.51x
   * - (15, 2000, 2000)
     - 1.313
     - 1.947
     - 45.72
     - 30.82
     - 1.48x

Reproduction snippet:
.. code-block:: python

    import numpy as np, time
    from eo_processor import median

    cube = np.random.rand(15, 2000, 2000)
    t0 = time.perf_counter(); rust_out = median(cube); rust_t = time.perf_counter() - t0
    t0 = time.perf_counter(); np_out = np.nanmedian(cube, axis=0); np_t = time.perf_counter() - t0
    print(f"Rust {rust_t:.3f}s vs NumPy {np_t:.3f}s speedup {np_t/rust_t:.2f}x")
    assert np.allclose(rust_out, np_out, atol=1e-12)

Interpretation:
- Gains come from per-pixel time series extraction, native sort, and parallel spatial iteration.
- Very small time dimensions may show smaller or no gains; benchmark your workload.
- Sorting cost scales with time axis length; consider temporal downsampling if performance is critical.

Performance Claim Template:
.. code-block:: text

    Benchmark:
    Shape: (15, 2000, 2000)
    NumPy nanmedian: 1.947s
    Rust median: 1.313s
    Speedup: 1.48x
    Methodology: single run, time.perf_counter(), float64, np.allclose(..., atol=1e-12)

Related Functions
-----------------
- :func:`temporal_mean` – Mean along time axis
- :func:`temporal_std` – Sample standard deviation (ddof=1)
- :func:`composite` – Convenience dispatcher (currently median only)

Best Practices
--------------
- Ensure the first axis truly represents time; transpose beforehand if needed.
- For very large time dimensions, consider preprocessing (e.g., masking or downsampling) to reduce per-pixel sort cost.
- Keep `skip_na=True` unless specific sentinel NaN propagation semantics are needed.

Edge Cases
----------
- All NaNs in a pixel (with `skip_na=True`) → output NaN.
- Empty time dimension (e.g., shape (0, y, x)) is not supported (will raise).
- Extremely large time axis sizes may increase sort overhead; a streaming median is not implemented yet.

Version Notes
-------------
Document reflects public API as of version 0.6.0. Any change to behavior or added parameters will require a minor version bump per project guidelines.

License
-------
MIT (see repository root).

End of median reference.
