pixelwise_transform
===================

.. currentmodule:: eo_processor
.. autofunction:: pixelwise_transform

Overview
--------
``pixelwise_transform`` applies a per-element linear transform to 1D–4D numeric (time‑first) arrays:

.. math::
   y = \mathrm{clamp}(\text{scale} \cdot x + \text{offset})

Clamping is optional—if both ``clamp_min`` and ``clamp_max`` are ``None`` the result is simply ``scale * x + offset``. NaNs propagate unchanged (no special handling or implicit replacement).

Supported ranks (any numeric dtype accepted; internally coerced to ``float64``):
- 1D: ``(N,)``
- 2D: ``(T, F)`` or generic ``(rows, cols)``
- 3D: ``(T, Y, X)``
- 4D: ``(T, B, Y, X)``

Purpose
-------
A lightweight, fused transform for:
- Contrast / dynamic range adjustments
- Reflectance normalization
- Simple standardization (e.g., scale & shift to [0,1])
- Post-smoothing enhancement (chaining with temporal moving averages)

Parameters
----------
``arr`` : ``numpy.ndarray`` (1D–4D)
    Input numeric array.
``scale`` : ``float``, default ``1.0``
    Multiplicative factor applied to each element.
``offset`` : ``float``, default ``0.0``
    Additive offset after scaling.
``clamp_min`` : ``float`` or ``None``, default ``None``
    Lower bound; values below are set to this boundary.
``clamp_max`` : ``float`` or ``None``, default ``None``
    Upper bound; values above are set to this boundary.

Returns
-------
``numpy.ndarray`` of same shape and float64 dtype.

NaN Behavior
------------
- Input NaNs remain NaN regardless of scaling or clamping.
- No implicit replacement; chain with ``replace_nans`` or masking functions if needed.

Formula Details
---------------
Unclamped:
.. math:: y = s \cdot x + o

Clamped:
.. math::
   y = \min\left(\max\left(s \cdot x + o,\ \text{clamp\_min}\right),\ \text{clamp\_max}\right)

If only one clamp is defined, only that bound is enforced.

Complexity
----------
- Time: O(N) where N = total elements.
- Memory: One output allocation (no intermediate temporaries).
- Parallelism: Current implementation is a single pass; potential vectorization handled by Rust compiler. (No Rayon parallelization—operation typically memory bound.)

Examples
--------
Basic scaling:
.. code-block:: python

    import numpy as np
    from eo_processor import pixelwise_transform

    arr = np.array([0.0, 0.5, 1.0])
    out = pixelwise_transform(arr, scale=2.0, offset=-0.5)
    print(out)  # [-0.5, 0.5, 1.5]

With clamping:
.. code-block:: python

    stretched = pixelwise_transform(arr, scale=2.0, offset=-0.5,
                                    clamp_min=0.0, clamp_max=1.0)
    print(stretched)  # [0.0, 0.5, 1.0]

2D transform:
.. code-block:: python

    img = np.random.rand(512, 512)
    norm = pixelwise_transform(img, scale=1.2, offset=-0.1, clamp_min=0.0, clamp_max=1.0)

4D stack (time, band, y, x):
.. code-block:: python

    cube4 = np.random.rand(12, 4, 256, 256)
    adjusted = pixelwise_transform(cube4, scale=0.8, offset=0.02)

Chaining with moving average:
.. code-block:: python

    from eo_processor import moving_average_temporal, pixelwise_transform
    cube = np.random.rand(48, 512, 512)
    smooth = moving_average_temporal(cube, window=5)
    enhanced = pixelwise_transform(smooth, scale=1.1, offset=0.05,
                                   clamp_min=0.0, clamp_max=1.0)

Use Cases
---------
- Producing normalized inputs for ML models after temporal smoothing.
- Reflectance stretching for visualization.
- Intensity compression before encoding (clamping to [0, 1]).
- Quick brightness/contrast adjustments in preprocessing pipelines.

Performance Notes
-----------------
A single memory-bound pass typically dominated by system bandwidth. Rust avoids Python loop overhead, benefiting large arrays (e.g., multi-band tiles). Gains vs pure NumPy may be modest for very simple transforms because NumPy also performs vectorized arithmetic efficiently; primary advantage is when chaining inside a Rust-heavy pipeline (reduced Python call overhead).

Representative Micro Benchmark (Indicative)
-------------------------------------------
Shape (Y, X) = (4096, 4096), scale=1.2, offset=-0.1, clamp [0,1]:

+----------------------+----------+----------+----------+
| Implementation       | Time (s) | Elements | Throughput (ME/s) |
+======================+==========+==========+===================+
| Rust pixelwise       | 0.052    | 16,777,216 | 322.64 |
| NumPy (scale+offset) | 0.060    | 16,777,216 | 279.62 |
+----------------------+----------+----------+----------+
Speedup ≈ 1.15x (environment dependent; rerun locally).

Fairness & Baseline
-------------------
Speedups >1× emerge primarily from reduced Python dispatch when chaining many operations; raw arithmetic often similar. Report which baseline is used (single combined vectorized expression in NumPy vs separate steps).

Performance Claim Template
--------------------------
.. code-block:: text

    Benchmark:
    Shape: (4096, 4096)
    Operation: scale=1.2, offset=-0.1, clamp_min=0.0, clamp_max=1.0
    NumPy: 0.060s
    Rust pixelwise_transform: 0.052s
    Speedup: 1.15x
    Methodology: single run, time.perf_counter(), float64 data, warm cache
    Validation: np.allclose(rust_out, np.clip(img*1.2 - 0.1, 0, 1), atol=1e-12)

Guidance
--------
Use ``pixelwise_transform`` when:
- Operating inside a Rust-accelerated workflow.
- You need uniform NaN propagation without mask arrays.
- Chaining scale/offset/clamp logic across large multidimensional arrays.

Limitations
-----------
- No gamma or nonlinear curves (implement externally).
- No per-band distinct scale/offset (pre-apply broadcasting or extend function).
- No vectorized SIMD annotations yet (opportunity for future optimization).
- No automatic handling of integer overflow (inputs coerced to float64 first).

Related Functions
-----------------
- :func:`moving_average_temporal`
- :func:`moving_average_temporal_stride`
- :func:`temporal_mean`
- :func:`median`
- :func:`replace_nans`
- Masking utilities (value/range based)

Potential Future Extensions
---------------------------
- Per-band scaling arrays (e.g. ``scale[b]``).
- Nonlinear transforms (gamma, log, sigmoid).
- Fused multi-step pipeline (e.g. smoothing + transform + masking in one pass).
- Optional parallelization for extremely large arrays (evaluate overhead benefits).

Validation Snippet
------------------
.. code-block:: python

    import numpy as np
    from eo_processor import pixelwise_transform
    img = np.random.rand(1024, 1024)
    rust_out = pixelwise_transform(img, scale=1.2, offset=-0.1, clamp_min=0, clamp_max=1)
    np_out = np.clip(img * 1.2 - 0.1, 0, 1)
    assert np.allclose(rust_out, np_out, atol=1e-12)

End of pixelwise_transform reference.
