nbr
===

.. currentmodule:: eo_processor

.. autofunction:: nbr

Overview
--------
`nbr` computes the Normalized Burn Ratio (NBR):

.. math::

   \mathrm{NBR} = \frac{NIR - SWIR2}{NIR + SWIR2}

This index is widely used for burn severity assessment. Comparing pre‑ and post‑event NBR via
``delta_nbr`` highlights fire impacts (larger positive Δ often indicates higher severity).

Parameters
----------
nir : numpy.ndarray
    Near‑infrared reflectance band (any numeric dtype; internally coerced to float64).
swir2 : numpy.ndarray
    Short‑wave infrared 2 reflectance band (same shape & dtype category as `nir`).

Returns
-------
numpy.ndarray
    NBR values with the same shape as inputs (float64). Values typically range from -1 to 1.

Numeric Stability
-----------------
For pixels where ``nir + swir2`` is ~0, an epsilon guard returns 0.0 to avoid division instability.
This prevents spurious large magnitude values when both bands are near zero.

Shape Support
-------------
- 1D: (n_pixels,)
- 2D: (rows, cols)
- 3D / 4D: Supported internally via the generic normalized difference primitive if higher‑rank arrays
  are passed (e.g., (time, y, x) or (time, band, y, x)). Public documentation emphasizes 1D/2D usage;
  ensure matching shapes between `nir` and `swir2` for all dimensions.

Typical Interpretation (Approximate)
------------------------------------
- High positive values: Recently burned regions often show reduced NIR and increased SWIR reflectance.
- Moderate: Mixed or partially recovered vegetation.
- Low / negative: Water, snow, or non‑vegetated surfaces.

Related Functions
-----------------
- ``delta_nbr``: Change detection between pre‑ and post‑event NBR.
- ``ndvi`` / ``delta_ndvi``: Vegetation condition and change.
- ``ndmi``: Moisture-related index for complementary analysis.

Example
-------
.. code-block:: python

   import numpy as np
   from eo_processor import nbr

   nir   = np.array([0.6, 0.5, 0.4], dtype=np.float32)
   swir2 = np.array([0.3, 0.2, 0.1], dtype=np.float32)
   out = nbr(nir, swir2)
   print(out)

End of NBR reference.
