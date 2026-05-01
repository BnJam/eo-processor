evi2
====

.. currentmodule:: eo_processor

.. autofunction:: evi2

Overview
--------
`evi2` computes the 2-band Enhanced Vegetation Index variant (EVI2):

.. math::

   EVI2 = G \cdot \frac{NIR - Red}{NIR + C1 \cdot Red + L}

Using common constants:
- :math:`G = 2.5`
- :math:`C1 = 2.4`
- :math:`L = 1.0`

EVI2 is often used when the blue band is unavailable, while retaining some of EVI's
improvements over NDVI.

Usage
-----
.. code-block:: python

   import numpy as np
   from eo_processor import evi2

   nir = np.array([0.60, 0.70, 0.35])
   red = np.array([0.25, 0.20, 0.15])

   out = evi2(nir, red)
   print(out)

Shapes & Dtypes
---------------
- Supports 1D and 2D arrays in the public Python API.
- Inputs may be any numeric dtype (int/uint/float); coerced to `float64` internally.
- Shapes of `nir` and `red` must match exactly; mismatch raises `ValueError`.

Numerical Stability
-------------------
Very small denominators are guarded with an EPSILON (1e-10). When `nir + 2.4*red + 1`
is ~0 the output is set to 0.0 to avoid instability.

See Also
--------
- :func:`enhanced_vegetation_index` / :func:`evi` (3-band EVI)
- :func:`ndvi` (simple 2-band normalized difference)

End of EVI2 documentation.

