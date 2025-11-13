ndwi
====

.. currentmodule:: eo_processor

Normalized Difference Water Index (NDWI).

.. autofunction:: ndwi

Overview
--------
`ndwi` computes the Normalized Difference Water Index:

.. math::

   NDWI = \frac{Green - NIR}{Green + NIR}

Typical interpretation:
- > 0.3: open water (often 0.4–0.6)
- 0.0–0.3: moist vegetation / wetlands
- < 0.0: dry vegetation / soil

Inputs must have matching shape (1D–2D); any numeric dtype is accepted (coerced to float64 internally).
