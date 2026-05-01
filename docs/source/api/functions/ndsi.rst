ndsi
====

.. currentmodule:: eo_processor

.. autofunction:: ndsi

Overview
--------
`ndsi` computes the Normalized Difference Snow Index:

.. math::

   NDSI = \frac{Green - SWIR1}{Green + SWIR1}

It is commonly used to separate snow/ice from most non-snow surfaces.
Values typically range from -1 to 1:
- Higher positive values (often > 0.3): likely snow/ice
- Near 0: mixed or uncertain surfaces
- Negative values: vegetation, soil, water, or built surfaces

Usage
-----
.. code-block:: python

   import numpy as np
   from eo_processor import ndsi

   green = np.array([0.52, 0.58, 0.44])
   swir1 = np.array([0.18, 0.22, 0.35])

   out = ndsi(green, swir1)
   print(out)  # element-wise (green - swir1)/(green + swir1)

Shapes & Dtypes
---------------
- Supports 1D and 2D arrays in the public Python API.
- Inputs may be any numeric dtype (int/uint/float); coerced to `float64` internally.
- Shapes of `green` and `swir1` must match exactly; mismatch raises `ValueError`.

Numerical Stability
-------------------
Very small denominators are guarded with an EPSILON (1e-10). When `green + swir1` is ~0
the output is set to 0.0 to avoid instability.

Interpretation Notes
--------------------
Use NDSI with cloud masks, temperature, or elevation constraints in production snow workflows.
Snow detection quality can vary with illumination, terrain shadow, and sensor band responses.

See Also
--------
- :func:`ndwi` for water-focused normalized difference
- :func:`ndmi` for moisture-focused normalized difference
- :func:`normalized_difference` generic primitive used by multiple indices

End of NDSI documentation.
