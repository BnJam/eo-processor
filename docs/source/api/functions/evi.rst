evi
===

.. currentmodule:: eo_processor

Enhanced Vegetation Index (alias)
---------------------------------

The function ``evi`` is an alias of :func:`enhanced_vegetation_index` provided for user convenience
and parity with common remote sensing naming conventions.

Formula (MODIS standard constants):

.. math::

   \mathrm{EVI} = G \cdot \frac{NIR - Red}{NIR + C_1 \cdot Red - C_2 \cdot Blue + L}

Where:

- :math:`G = 2.5`
- :math:`C_1 = 6.0`
- :math:`C_2 = 7.5`
- :math:`L = 1.0`

Inputs
------
All three band arrays (``nir``, ``red``, ``blue``) must have identical shape.
Supported dtypes: any numeric NumPy dtype (coerced to ``float64`` internally).

Dimensional Dispatch
--------------------
The alias exposes the same dimensional behavior as ``enhanced_vegetation_index`` (1D–2D
documented; internal kernel supports higher ranks but these are not guaranteed as part of the
public API contract unless explicitly stated in release notes).

Numerical Stability
-------------------
A near-zero safeguard (epsilon) is applied to the denominator; pixels where the denominator
magnitude is below the threshold return ``0.0``.

Usage Example
-------------
.. code-block:: python

    import numpy as np
    from eo_processor import evi

    nir  = np.array([0.6, 0.7])
    red  = np.array([0.3, 0.2])
    blue = np.array([0.1, 0.05])
    out = evi(nir, red, blue)
    print(out)

See Also
--------
- :func:`enhanced_vegetation_index` (primary implementation)

API Reference
-------------
.. autofunction:: evi
