ndvi
====

.. currentmodule:: eo_processor

Normalized Difference Vegetation Index (NDVI).

.. autofunction:: ndvi

Formula
-------

.. math::

   \mathrm{NDVI} = \frac{NIR - Red}{NIR + Red}

Where both bands are reflectance (typically scaled to 0–1). An internal epsilon guard ensures near‑zero denominators yield 0 rather than unstable large magnitudes.

Usage
-----

.. code-block:: python

   import numpy as np
   from eo_processor import ndvi

   nir = np.array([0.8, 0.7, 0.6])
   red = np.array([0.2, 0.1, 0.3])
   out = ndvi(nir, red)
   print(out)  # array of NDVI values

Supported Shapes
----------------
- 1D arrays: (pixels,)
- 2D arrays: (rows, cols)

If you pass higher dimensional arrays they are currently dispatched internally via the generic normalized difference primitive, but only 1D/2D are part of the documented public contract (future versions may formalize 3D/4D spectral support).

Input Dtypes
------------
Any numeric dtype (int, uint, float32, float64) is accepted; values are coerced to float64 internally for stable arithmetic.

Output Range (Typical)
----------------------
NDVI values are bounded in [-1, 1]. Common interpretation guidelines:

.. list-table:: NDVI Interpretation (Typical)
   :header-rows: 1

   * - NDVI
     - Interpretation
   * - < 0
     - Water / snow / clouds
   * - 0.0–0.2
     - Bare soil / built surfaces
   * - 0.2–0.5
     - Sparse to moderate vegetation
   * - > 0.5
     - Dense, healthy vegetation

Numerical Stability
-------------------
A small epsilon (~1e-10) is applied to detect near-zero denominators. If the absolute value of (NIR + Red) is below epsilon the output is set to 0.0 for that element to avoid division artifacts.

Performance
-----------
Rust implementation:
- Fused arithmetic (single pass, minimal temporaries)
- Releases Python's GIL enabling multi-core execution for larger arrays
- Coerces dtype once, reducing branching
- Single dtype coercion (float64) for stable arithmetic

Benchmark (Representative Single Run)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Platform: macOS ARM64, CPython 3.10, release build
Timing: time.perf_counter(), warm cache, float64 arrays

.. list-table:: NDVI Benchmark (Single Run)
   :header-rows: 1

   * - Size
     - Rust (s)
     - NumPy (s)
     - Rust Throughput (M/s)
     - NumPy Throughput (M/s)
     - Speedup
     - Notes
   * - 3000x3000
     - 0.039
     - 0.030
     - 230.77
     - 300.00
     - 0.78x
     - Smaller tile; NumPy temporaries cheap
   * - 5000x5000
     - 0.080
     - 0.112
     - 312.50
     - 223.21
     - 1.40x
     - Larger tile benefits from fused loop

Interpretation:
- Small/medium arrays may show parity or slight regression (Python overhead vs parallel threshold).
- Larger arrays benefit from fused arithmetic and GIL release (reduced temporaries, better cache locality).

Reproduction Snippet:
.. code-block:: python

    import numpy as np, time
    from eo_processor import ndvi

    nir = np.random.rand(5000, 5000)
    red = np.random.rand(5000, 5000)

    t0 = time.perf_counter()
    rust_out = ndvi(nir, red)
    rust_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    numpy_out = (nir - red) / (nir + red + 1e-10)
    numpy_t = time.perf_counter() - t0

    print(f"Rust {rust_t:.3f}s vs NumPy {numpy_t:.3f}s speedup {numpy_t/rust_t:.2f}x")
    assert np.allclose(rust_out, numpy_out, atol=1e-12)

Performance Claim Template:
.. code-block:: text

    Benchmark:
    Array size: 5000 x 5000
    Old (NumPy): 0.112s
    New (Rust ndvi): 0.080s
    Speedup: 1.40x
    Methodology: single run, time.perf_counter(), float64 arrays, validation np.allclose(..., atol=1e-12)


Example With XArray/Dask
------------------------

.. code-block:: python

   import dask.array as da
   import xarray as xr
   from eo_processor import ndvi

   nir = da.random.random((6000, 6000), chunks=(750, 750))
   red = da.random.random((6000, 6000), chunks=(750, 750))
   nir_xr = xr.DataArray(nir, dims=["y", "x"])
   red_xr = xr.DataArray(red, dims=["y", "x"])

   ndvi_xr = xr.apply_ufunc(
       ndvi, nir_xr, red_xr,
       dask="parallelized",
       output_dtypes=[float],
   )
   result = ndvi_xr.compute()

Error Handling
--------------
- Shape mismatch raises ``ValueError`` with context
- Non-numeric or unsupported ndim raises ``TypeError``

Testing
-------
See ``tests/test_indices.py`` for:
- Range property tests (values remain within [-1, 1] ± floating tolerance)
- Shape mismatch negative tests
- Dtype coercion verification

See Also
--------
- :func:`normalized_difference` (generic base primitive)
- :func:`delta_ndvi` (change detection pre vs post event)
- :func:`savi` (soil-adjusted variant)
- :func:`enhanced_vegetation_index` (EVI; improved sensitivity over high biomass)

Notes
-----
For change analysis prefer :func:`delta_ndvi` which computes NDVI at two epochs and returns the difference (pre - post).

End of NDVI documentation.
