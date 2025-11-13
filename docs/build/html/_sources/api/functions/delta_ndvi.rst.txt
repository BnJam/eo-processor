delta_ndvi
==========

.. currentmodule:: eo_processor

Delta NDVI (change in Normalized Difference Vegetation Index between two epochs).

.. autofunction:: delta_ndvi

Description
-----------
``delta_ndvi(pre_nir, pre_red, post_nir, post_red)`` computes the difference between the NDVI
value at a pre-event epoch and a post-event epoch:

.. math::

   \Delta \mathrm{NDVI} = \mathrm{NDVI}_{\text{pre}} - \mathrm{NDVI}_{\text{post}}

with:

.. math::

   \mathrm{NDVI} = \frac{NIR - Red}{NIR + Red + \varepsilon}

An internal epsilon (\\( \varepsilon \approx 10^{-10} \\)) guards near‑zero denominators; when
``|NIR + Red| < ε`` the NDVI for that element is treated as 0.0 to avoid instability.

Interpretation
--------------
- Positive ΔNDVI: vegetation loss or degradation (pre higher than post; e.g. burn, drought, harvest)
- Negative ΔNDVI: vegetation gain or recovery (post higher than pre)
- Near zero: minimal change

Parameters
----------
pre_nir, pre_red : numpy.ndarray
    Near‑infrared and red reflectance arrays for the pre‑event date.
post_nir, post_red : numpy.ndarray
    Near‑infrared and red reflectance arrays for the post‑event date.

All four arrays must:
- Have identical shape (1D or 2D documented; higher ranks dispatch internally but are not part of the formal contract yet)
- Be numeric (int/uint/float); values are coerced to ``float64`` internally

Returns
-------
numpy.ndarray (float64)
Array of the same shape representing ``NDVI_pre - NDVI_post``.

Usage
-----
Basic 1D example:

.. code-block:: python

   import numpy as np
   from eo_processor import delta_ndvi

  pre_nir  = np.array([0.62, 0.58, 0.40])
  pre_red  = np.array([0.10, 0.12, 0.15])
  post_nir = np.array([0.55, 0.50, 0.38])
  post_red = np.array([0.14, 0.16, 0.18])

  change = delta_ndvi(pre_nir, pre_red, post_nir, post_red)
  print(change)  # positive values -> decline in NDVI

2D (image) example:

.. code-block:: python

   from eo_processor import delta_ndvi
   pre_nir  = np.random.rand(512, 512)
   pre_red  = np.random.rand(512, 512)
   post_nir = np.random.rand(512, 512)
   post_red = np.random.rand(512, 512)
   delta = delta_ndvi(pre_nir, pre_red, post_nir, post_red)
   assert delta.shape == pre_nir.shape

Numerical Stability
-------------------
Each NDVI epoch uses epsilon safeguarding. This limits extreme outputs from near‑zero denominators
so that ΔNDVI reflects genuine spectral changes rather than numeric artifacts.

Error Handling
--------------
- Shape mismatch → ``ValueError`` with context
- Non-numeric or unsupported dimensionality (>4D) → ``TypeError``

Performance Notes
-----------------
- Executes two fused NDVI passes in Rust and subtracts results in a single loop.
- Releases the GIL; large arrays benefit from internal parallelism thresholds.
- Single dtype coercion per input reduces overhead.

Related Functions
-----------------
- :func:`ndvi` (single-epoch vegetation index)
- :func:`delta_nbr` (burn ratio change)
- :func:`normalized_difference` (generic primitive underlying NDVI)
- :func:`savi`, :func:`enhanced_vegetation_index` (alternative vegetation metrics)

Best Practices
--------------
- Ensure radiometric consistency (same atmospheric correction / scaling) between pre and post inputs.
- Combine with masking (e.g., ``mask_scl`` to remove clouds / shadows) before computing change.
- For multi-temporal sequences beyond two dates, compute per-step deltas or trend metrics separately.

End of delta_ndvi reference.
