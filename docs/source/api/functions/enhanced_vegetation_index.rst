enhanced_vegetation_index
=========================

.. currentmodule:: eo_processor

Enhanced Vegetation Index (EVI)
-------------------------------
``enhanced_vegetation_index`` computes the MODIS-style Enhanced Vegetation Index providing improved
sensitivity in high biomass regions and partial resistance to atmospheric and soil background effects.

Formula
-------
.. math::

   \mathrm{EVI} = G \cdot \frac{NIR - Red}{NIR + C_1 \cdot Red - C_2 \cdot Blue + L}

Default constants (MODIS convention):
- :math:`G = 2.5`
- :math:`C_1 = 6.0`
- :math:`C_2 = 7.5`
- :math:`L = 1.0`

Parameters
----------
See autodoc for the full signature. Core arguments:

- ``nir`` : Near-infrared reflectance array
- ``red`` : Red reflectance array
- ``blue`` : Blue reflectance array
- ``G`` : Gain factor (default 2.5)
- ``C1`` : Atmospheric resistance coefficient for red band (default 6.0)
- ``C2`` : Atmospheric resistance coefficient for blue band (default 7.5)
- ``L`` : Canopy background adjustment (default 1.0)

All three band arrays must have identical shape. Any numeric dtype is accepted; values are
coerced to ``float64`` internally for stable arithmetic.

Dimensional Support
-------------------
Documented public contract: 1D and 2D arrays
- 1D: ``(pixels,)``
- 2D: ``(rows, cols)``

Higher ranks (e.g. ``(time, y, x)`` or ``(time, band, y, x)``) may work via internal generalized
kernels but are not formally guaranteed; future versions may expand documented support.

Numerical Stability
-------------------
A near-zero safeguard (epsilon ~1e-10) is applied to the denominator
``(NIR + C1*Red - C2*Blue + L)``. Elements where ``|denominator| < epsilon`` yield ``0.0`` to
avoid extreme magnitudes.

Usage
-----
.. code-block:: python

    import numpy as np
    from eo_processor import enhanced_vegetation_index

    nir  = np.array([0.60, 0.70, 0.65])
    red  = np.array([0.30, 0.25, 0.28])
    blue = np.array([0.10, 0.05, 0.07])

    evi_vals = enhanced_vegetation_index(nir, red, blue)
    print(evi_vals)

Custom constants (e.g., tuning for sensor-specific calibration):

.. code-block:: python

    evi_custom = enhanced_vegetation_index(nir, red, blue, G=2.4, C1=5.5, C2=7.0, L=1.0)

Alias
-----
``evi`` is provided as a direct alias of ``enhanced_vegetation_index`` for convenience and parity
with common remote sensing nomenclature.

Performance Notes
-----------------
Rust implementation:
- Single fused loop (no intermediate temporary arrays)
- GIL released during computation
- Conditional parallelism on large arrays (multi-core speedup)
- Early shape validation produces clear ``ValueError`` on mismatch

Interpretation (Typical)
------------------------
Higher EVI values indicate vigorous, healthy vegetation. Compared to NDVI, EVI reduces saturation
in dense canopy areas and mitigates some atmospheric / soil background effects through the added
coefficients and blue band term.

Error Handling
--------------
- Shape mismatch raises ``ValueError``
- Unsupported dimensionality raises ``TypeError``
- Negative or non-physical constant values (if validated in future) will raise appropriate errors

See Also
--------
- :func:`evi` (alias)
- :func:`ndvi` (classic normalized difference)
- :func:`savi` (soil-adjusted variant)
- :func:`normalized_difference` (generic primitive)

Autodoc
-------
.. autofunction:: enhanced_vegetation_index

End of enhanced_vegetation_index documentation.
