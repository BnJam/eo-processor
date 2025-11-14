savi
====

.. currentmodule:: eo_processor

Soil Adjusted Vegetation Index (SAVI)
-------------------------------------

SAVI introduces a soil brightness correction factor `L` to reduce the influence of
exposed soil in areas of sparse vegetation. It is especially useful where bare soil
background would otherwise bias NDVI.

**Formula**

.. math::

   \mathrm{SAVI} = \frac{(NIR - Red)}{(NIR + Red + L)} \times (1 + L)

Typical values:
- L = 0.0 → behaves more like standard NDVI for dense vegetation
- L = 0.5 → commonly used default (moderate vegetation)
- L = 1.0 → stronger soil brightness correction for very sparse vegetation

Parameters
----------
- `nir`: Near-infrared reflectance array (numeric dtype; coerced to float64 internally)
- `red`: Red reflectance array (same shape as `nir`)
- `L` (keyword `L` or alternative lowercase `l`): Soil brightness factor (≥ 0.0)

Dimensional Dispatch
--------------------
Accepts matching-shaped arrays with rank 1–4:
- 1D: `(n_pixels,)`
- 2D: `(rows, cols)` or `(samples, features)`
- 3D: `(time, rows, cols)`
- 4D: `(time, bands, rows, cols)`

The implementation validates shape equality and applies an element-wise fused loop.

Stability
---------
A small epsilon (1e-10) guards near‑zero denominators; if `(NIR + Red + L)` is effectively zero,
the output defaults to 0.0 for that element.

Keyword Precedence
------------------
If both `L` and `l` are supplied in the Python wrapper, the lowercase `l` value takes precedence:

.. code-block:: python

   out = savi(nir, red, L=0.0, l=0.25)  # uses 0.25

Usage Examples
--------------
Basic:

.. code-block:: python

   import numpy as np
   from eo_processor import savi

   nir = np.array([0.7, 0.6, 0.5])
   red = np.array([0.2, 0.3, 0.1])
   out = savi(nir, red)  # L defaults to 0.5

Custom L:

.. code-block:: python

   out_sparse = savi(nir, red, L=1.0)
   out_dense  = savi(nir, red, L=0.0)

3D (time series cube):

.. code-block:: python

   cube_nir = np.random.rand(12, 256, 256)
   cube_red = np.random.rand(12, 256, 256)
   savi_med = savi(cube_nir, cube_red, L=0.5)

Interpretation
--------------
Higher SAVI values indicate healthier or denser vegetation adjusted for soil brightness.
Values generally range roughly from -1 to +1, similar to NDVI, though scale and distribution
can shift with different `L` choices.

Returns
-------
Float64 array with the same shape as inputs (time axis retained for higher-rank inputs).

Raises
------
- `ValueError` on shape mismatch
- `ValueError` if `L < 0`
- `TypeError` if input arrays are not 1D–4D numeric arrays

See Also
--------
- :doc:`ndvi` for classic vegetation index without soil brightness factor
- :doc:`enhanced_vegetation_index` for improved sensitivity over high biomass
- :doc:`normalized_difference` base primitive used by several indices

Autodoc
-------
.. autofunction:: savi
