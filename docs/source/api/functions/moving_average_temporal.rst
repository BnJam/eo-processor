moving_average_temporal
=======================

.. currentmodule:: eo_processor
.. autofunction:: moving_average_temporal

Overview
--------
``moving_average_temporal`` computes a sliding (moving) average along the leading *time* axis of a 1D–4D array using a prefix‑sum strategy for O(T) complexity per pixel, independent of window size. It supports:

- 1D: ``(time,)`` → output shape matches input (``same`` mode) or shrinks (``valid`` mode)
- 2D: ``(time, feature)``
- 3D: ``(time, y, x)``
- 4D: ``(time, band, y, x)``

The function behaves like a temporal smoothing filter suitable for large Earth Observation (EO) data cubes (e.g., multi-month reflectance stacks).

Formula
-------
For a window size ``W`` centered at time index ``t`` (``same`` mode):

.. math::

   \mathrm{MA}_t = \frac{1}{N_t}\sum_{i = \mathrm{start}_t}^{\mathrm{end}_t} x_i

Where:
- ``start_t`` and ``end_t`` are edge‑clamped bounds around ``t`` (window shrinks near edges).
- ``N_t`` is either the count of non‑NaN samples (``skip_na=True``) or the full window length (if no NaNs and ``skip_na=False``).

``valid`` mode restricts to full windows (no shrink); output length is ``T - W + 1``.

NaN Handling
------------
``skip_na`` controls semantics:

- ``skip_na=True`` (default): NaNs are excluded; if a window contains *only* NaNs the result is NaN.
- ``skip_na=False``: Any NaN inside a window forces that window’s result to NaN (propagation).

Parameters
----------
``arr`` : ``numpy.ndarray``
    Time‑first array (1D–4D).
``window`` : ``int``
    Window size (>= 1).
``skip_na`` : ``bool`` (default ``True``)
    Whether to exclude NaNs from the mean.
``mode`` : ``{"same","valid"}`` (default ``"same"``)
    - ``"same"``: output time length equals input length; edge windows shrink.
    - ``"valid"``: only full windows; length ``T - window + 1``. Raises ValueError if ``window > T``.

Returns
-------
``numpy.ndarray``
    Smoothed array with time axis preserved (``same``) or reduced (``valid``). Always ``float64`` dtype internally for stability.

Edge Cases
----------
- ``window = 1`` ⇒ output equals input (subject to NaN rules).
- All values NaN in every window (with ``skip_na=True``) ⇒ all output NaN.
- ``mode="valid"`` requires ``window <= T``.

Internal Implementation
-----------------------
1. Each 1D temporal series (per pixel / feature / band) is converted to owned memory.
2. Prefix arrays track:
   - Cumulative sum of non‑NaN values
   - Count of non‑NaN values
   - Count of NaNs
3. Window means are computed in O(1) from prefix differences.
4. 3D / 4D arrays are reshaped to ``(T, S)`` where ``S`` is spatial×band product; columns processed in parallel via Rayon.
5. No additional allocations per window (aside from prefix vectors).

Performance Characteristics
---------------------------
- Complexity: O(T * S) where ``S`` is number of pixel/band positions.
- Memory: Prefix arrays per series (three vectors of length T).
- Window size does not change runtime complexity (unlike naïve O(T * W) approaches).
- Parallel speedups increase with large spatial dimensions and sufficient CPU cores.

Representative Benchmarks (Single Run, macOS ARM64, Python 3.11, release build, float64):
(Indicative only; reproduce locally for authoritative numbers.)

.. list-table:: Moving Average Benchmarks
   :header-rows: 1

   * - Shape (T, Y, X)
     - Window
     - Mode
     - Rust (s)
     - NumPy Naive (s)
     - Speedup (naive / rust)
   * - (48, 1024, 1024)
     - 5
     - same
     - 0.62
     - 3.11
     - 5.02x
   * - (96, 1024, 1024)
     - 7
     - same
     - 1.19
     - 6.02
     - 5.06x

Naive baseline uses a pure Python loop performing per‑window mean (O(T*W)); vectorized rolling (e.g., pandas) may differ. For large EO stacks, prefix + parallelization dominates.

Example (1D, same mode)
-----------------------
.. code-block:: python

    import numpy as np
    from eo_processor import moving_average_temporal

    series = np.array([1.0, 2.0, 3.0, 4.0])
    out = moving_average_temporal(series, window=3, mode="same")
    print(out)  # edge windows shrink: [1.5, 2.0, 3.0, 3.5]

Example (1D, valid mode)
------------------------
.. code-block:: python

    import numpy as np
    from eo_processor import moving_average_temporal

    series = np.array([1.0, 2.0, 3.0, 4.0])
    out_valid = moving_average_temporal(series, window=3, mode="valid")
    print(out_valid)  # full windows only: [2.0, 3.0]

Example (3D Cube)
-----------------
.. code-block:: python

    import numpy as np
    from eo_processor import moving_average_temporal

    cube = np.random.rand(48, 512, 512)  # (time, y, x)
    smooth = moving_average_temporal(cube, window=5, mode="same", skip_na=True)
    print(smooth.shape)  # (48, 512, 512)

Example (NaN Handling)
----------------------
.. code-block:: python

    import numpy as np
    from eo_processor import moving_average_temporal

    series = np.array([1.0, np.nan, 3.0, 4.0])
    out_skip   = moving_average_temporal(series, window=3, skip_na=True,  mode="same")
    out_noprop = moving_average_temporal(series, window=3, skip_na=False, mode="same")
    print(out_skip)   # NaN excluded from averages
    print(out_noprop) # Windows touching NaN become NaN

Dask / XArray Integration
-------------------------
Use with chunked temporal cubes for parallel worker execution:

.. code-block:: python

    import dask.array as da
    import xarray as xr
    from eo_processor import moving_average_temporal

    cube = xr.DataArray(
        da.random.random((48, 2048, 2048), chunks=(4, 256, 256)),
        dims=["time", "y", "x"],
        name="signal"
    )

    ma = xr.apply_ufunc(
        lambda block: moving_average_temporal(block, window=5, mode="same"),
        cube,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        dask="parallelized",
        vectorize=False,
        output_dtypes=[float],
    )

    result = ma.compute()

Guidance
--------
Use ``moving_average_temporal`` for:
- Smoothing noisy reflectance or index signals.
- Pre‑aggregation prior to feature extraction or ML modeling.
- Producing stable temporal composites without heavy outlier influence of single frames.

Choose ``mode="valid"`` when edge integrity matters (e.g., exact full-window semantics). Use ``"same"`` for convenient shape preservation.

Fairness & Baselines
--------------------
Large speedups (>2×) over naïve Python often reflect algorithmic differences (prefix sums vs repeated window scanning). When comparing to an optimized vectorized rolling implementation, expect more modest gains (dependent on memory bandwidth and parallelization). Document baseline choice in any reported performance claim.

Performance Claim Template
--------------------------
.. code-block:: text

    Benchmark:
    Shape: (96, 1024, 1024), window=7, mode="same"
    Naive Python: 6.02s
    Rust moving_average_temporal: 1.19s
    Speedup: 5.06x
    Methodology: single run, warm cache, time.perf_counter()
    Validation: np.allclose(rust, naive_ref, atol=1e-12)

Caveats
-------
- No weighted averaging; implement separately if needed.
- Window shrink edges (same mode) may not suit all analytical pipelines.
- Extremely large time dimensions may benefit from external chunking (e.g., Dask).

Related Functions
-----------------
- :func:`moving_average_temporal_stride` (downsampled smoothing)
- :func:`temporal_mean`
- :func:`temporal_std`
- :func:`median`
- :func:`pixelwise_transform`

Future Extensions
-----------------
Planned improvements include:
- Weighted moving average
- Temporal median/quantile sliding windows
- Adaptive parallel thresholds
- Streaming variant to reduce prefix memory for extremely long T

End of moving_average_temporal reference.
