delta_nbr
=========

.. currentmodule:: eo_processor
.. autofunction:: delta_nbr

Overview
--------
``delta_nbr`` computes the change in the Normalized Burn Ratio (NBR) between two epochs
(e.g., pre‑fire and post‑fire). It is defined as:

.. math::

   \Delta \mathrm{NBR} = \mathrm{NBR}_{pre} - \mathrm{NBR}_{post}

where each NBR is:

.. math::

   \mathrm{NBR} = \frac{NIR - SWIR2}{NIR + SWIR2}

Positive ΔNBR values typically indicate burn severity (larger reduction in NIR / increase in SWIR2),
while near‑zero or negative values can suggest little change or recovery.

Formula
-------
1. Compute NBR for the pre‑event bands: :math:`(NIR_{pre} - SWIR2_{pre}) / (NIR_{pre} + SWIR2_{pre})`
2. Compute NBR for the post‑event bands: :math:`(NIR_{post} - SWIR2_{post}) / (NIR_{post} + SWIR2_{post})`
3. Subtract: :math:`\Delta \mathrm{NBR} = \mathrm{NBR}_{pre} - \mathrm{NBR}_{post}`

Very small denominators (|NIR + SWIR2| < 1e-10) in either epoch are guarded internally
and yield a 0.0 contribution for that pixel’s NBR.

Parameters (see signature)
--------------------------
- ``nir_pre``: Near‑infrared reflectance array (pre‑event)
- ``swir2_pre``: SWIR2 reflectance array (pre‑event)
- ``nir_post``: Near‑infrared reflectance array (post‑event)
- ``swir2_post``: SWIR2 reflectance array (post‑event)

All arrays must have identical shape (1D, 2D; higher ranks may dispatch through internal
normalized difference logic but only 1D/2D are formally documented).

Returns
-------
``numpy.ndarray`` (float64) with the same shape as the input arrays, containing ΔNBR values.
Typical dynamic range is roughly [-2, 2], though interpretability focuses on relative
magnitude rather than absolute bounds.

Usage
-----
.. code-block:: python

   import numpy as np
   from eo_processor import delta_nbr

   nir_pre   = np.array([0.62, 0.58, 0.55])
   swir2_pre = np.array([0.25, 0.22, 0.20])
   nir_post  = np.array([0.40, 0.37, 0.35])
   swir2_post= np.array([0.30, 0.28, 0.27])

   d = delta_nbr(nir_pre, swir2_pre, nir_post, swir2_post)
   print(d)  # Positive values -> burn impact

Interpretation (Guideline – scene & sensor dependent)
-----------------------------------------------------
Higher positive ΔNBR: stronger burn / disturbance.
Moderate positive: partial burn or mixed recovery.
Near 0: little change.
Negative: potential regrowth, moisture increase, or noise (validate with ancillary data).

Numerical Stability
-------------------
Each epoch’s denominator uses an epsilon (~1e-10). If ``|NIR + SWIR2|`` < epsilon for that epoch,
its NBR contribution is set to 0.0, limiting ΔNBR magnitude inflation.

Error Handling
--------------
- Shape mismatch → ``ValueError``.
- Non-numeric or unsupported ndim → ``TypeError``.

See Also
--------
- :func:`nbr` – single-epoch Normalized Burn Ratio
- :func:`delta_ndvi` – vegetation change detection
- :func:`ndmi` – moisture index (complementary for recovery assessment)
- :func:`normalized_difference` – underlying primitive pattern

Notes
-----
For classification of burn severity, apply published threshold schemes (e.g., USGS ΔNBR classes)
on atmospherically corrected surface reflectance. Thresholds vary by sensor and preprocessing.

End of delta_nbr documentation.
