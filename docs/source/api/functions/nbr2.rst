nbr2
====


.. currentmodule:: eo_processor

Normalized Burn Ratio 2 (``nbr2``) highlights moisture and thermal contrasts using two
short‑wave infrared bands (SWIR1, SWIR2). It is complementary to the standard
Normalized Burn Ratio (``nbr``) and NDMI for post‑fire and moisture analysis.

Formula
-------
.. math::

   \mathrm{NBR2} = \frac{\mathrm{SWIR1} - \mathrm{SWIR2}}{\mathrm{SWIR1} + \mathrm{SWIR2}}

Near‑zero denominators are guarded internally; results default to ``0.0`` for
values where ``|SWIR1 + SWIR2| < 1e-10``.

Typical Interpretation (context dependent)
------------------------------------------
- Higher positive values: areas with increased moisture / certain unburned conditions
- Strong negative shifts (pre/post comparison): potential burn severity indicators
- Always validate ranges against sensor calibration & atmospheric correction steps

Parameters
----------
.. autofunction:: nbr2

Returns
-------
A NumPy ``float64`` array of the same shape as the input arrays (1D or 2D
“pixel grid”). For higher dimensional stacks, see the generalized
``normalized_difference`` primitive; ``nbr2`` is documented for the common 1D/2D use.

Notes
-----
- Inputs may be any numeric dtype; internally coerced to ``float64`` for stable arithmetic.
- Shapes must match exactly; otherwise a ``ValueError`` is raised.
- For time‑series or multi‑band cubes (3D/4D), call ``normalized_difference(swir1, swir2)``
  directly if you need broader dimensional dispatch.
- Use ``delta_nbr`` for change detection between pre/post epochs: it computes
  :math:`\mathrm{NBR}_{pre} - \mathrm{NBR}_{post}`.

Examples
--------
1D arrays:

.. code-block:: python

   import numpy as np
   from eo_processor import nbr2

   swir1 = np.array([0.40, 0.55, 0.47])
   swir2 = np.array([0.30, 0.25, 0.22])
   out = nbr2(swir1, swir2)
   # (swir1 - swir2) / (swir1 + swir2)
   print(out)

2D arrays:

.. code-block:: python

   swir1 = np.array([[0.40, 0.50],
                     [0.45, 0.55]])
   swir2 = np.array([[0.30, 0.28],
                     [0.20, 0.15]])
   out = nbr2(swir1, swir2)
   print(out.shape)  # (2, 2)

Cross‑Reference
---------------
- :doc:`delta_nbr` (change detection)
- :doc:`nbr` (standard burn ratio)
- :doc:`normalized_difference` (generic building block)
