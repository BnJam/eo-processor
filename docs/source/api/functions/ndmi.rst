ndmi
====

.. currentmodule:: eo_processor

.. autofunction:: ndmi

Overview
--------
`ndmi` computes the Normalized Difference Moisture Index:

.. math::

   NDMI = \frac{NIR - SWIR1}{NIR + SWIR1}

It highlights canopy / vegetation moisture content. Values typically range from -1 to 1:
- Near -1: very dry / bare areas
- Around 0: moderate moisture or mixed surfaces
- Higher positive values (> 0.2–0.4): moist, healthy vegetation / canopy water content

Usage
-----
.. code-block:: python

   import numpy as np
   from eo_processor import ndmi

   nir   = np.array([0.52, 0.61, 0.44])
   swir1 = np.array([0.28, 0.33, 0.30])

   out = ndmi(nir, swir1)
   print(out)  # element-wise (nir - swir1)/(nir + swir1)

Shapes & Dtypes
---------------
- Supports 1D and 2D arrays in the public Python API (internally higher ranks may be dispatched via the normalized difference primitive but are not guaranteed).
- Inputs may be any numeric dtype (int/uint/float); coerced to `float64` internally.
- Shapes of `nir` and `swir1` must match exactly; mismatch raises `ValueError`.

Numerical Stability
-------------------
Very small denominators are guarded with an EPSILON (1e-10). When `nir + swir1` is ~0 the output is set to 0.0 to avoid instability.

Interpretation Notes
--------------------
For time-series analysis, increases in NDMI can indicate post-rainfall recovery or phenological changes related to canopy moisture. Combine with NDVI/NBR to contextualize disturbance or recovery signals.

See Also
--------
- :func:`ndvi` for vegetation vigor
- :func:`nbr` / :func:`delta_nbr` for burn severity and change detection
- :func:`normalized_difference` generic primitive used by multiple indices

End of NDMI documentation.
