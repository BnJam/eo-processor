gci
===

.. currentmodule:: eo_processor
.. autofunction:: gci

Overview
--------
The Green Chlorophyll Index (GCI) is a simple reflectance‑based proxy for chlorophyll content and
plant vigor. It uses the ratio between near‑infrared (NIR) and green bands:

Formula
-------
.. math::

    \mathrm{GCI} = \frac{\mathrm{NIR}}{\mathrm{Green}} - 1

Where:
- NIR   = near‑infrared reflectance (typically 0–1 reflectance units)
- Green = green band reflectance

A near‑zero green reflectance is guarded internally; the implementation returns 0.0 when the
denominator is below an epsilon threshold to avoid numerical instability.

Typical Interpretation
----------------------
- GCI > 0 generally indicates presence of vegetation.
- Larger positive values can correlate with higher chlorophyll concentration (exact biophysical
  mapping requires calibration against field measurements or radiative transfer modeling).

Usage Example
-------------
.. code-block:: python

    import numpy as np
    from eo_processor import gci

    nir   = np.array([0.80, 0.90, 0.70], dtype=np.float64)
    green = np.array([0.40, 0.45, 0.35], dtype=np.float64)

    out = gci(nir, green)
    print(out)          # (nir/green) - 1 element-wise

Dimensions & Dtypes
-------------------
- Supports 1D and 2D arrays (the function internally dispatches to matching loops; higher ranks
  may be supported indirectly via shared normalized difference logic but are not part of the
  documented contract here).
- Inputs may be any numeric dtype; values are coerced to float64 for stable arithmetic.
- Output array is float64.

Edge Cases
----------
- Green ≈ 0: Result forced to 0.0 (instead of exploding toward ±∞).
- Shape mismatch between `nir` and `green` raises a `ValueError`.
- Non 1D/2D input raises a `TypeError`.

Performance Notes
-----------------
Loop executed in Rust (PyO3) with:
- Early shape validation (clear error rather than silent broadcast)
- Single fused pass (no temporaries)
- Float64 coercion once per input

See Also
--------
- :func:`ndvi` — vegetation vigor index using NIR and Red.
- :func:`enhanced_vegetation_index` (EVI) — atmosphere/soil-resistant vegetation metric.
- :func:`normalized_difference` — generic normalized difference primitive.

References
----------
- Concept popularized in remote sensing literature as a simple reflectance ratio derivative.
  For rigorous chlorophyll retrieval consider PROSPECT + SAIL model inversions or empirical
  regression against leaf samples.

End of GCI documentation.
