normalized_difference
=====================

.. currentmodule:: eo_processor

.. autofunction:: normalized_difference

Overview
--------
`normalized_difference(a, b)` computes the element‑wise ratio:

.. math::

   \frac{a - b}{a + b}

with an internal near‑zero (EPSILON = 1e-10) safeguard. When the denominator `a + b` has absolute value less than EPSILON the output for that element is set to `0.0` to avoid unstable large magnitudes or division by zero.

Dimensional Support
-------------------
This function supports 1D, 2D, 3D, and 4D numeric NumPy arrays (all input dtypes are coerced to `float64` internally). Shapes for `a` and `b` must match exactly.

Typical axis interpretations (not enforced; purely numeric):
- 1D: spectral samples or time series
- 2D: image or (samples, features)
- 3D: (time, y, x) or (band, y, x) depending on upstream layout
- 4D: (time, band, y, x) or other structured stacks

Input Rules
-----------
- Both inputs must be NumPy arrays with identical shape.
- Any numeric dtype is accepted; coercion to float64 occurs before computation.
- NaNs propagate: if either element is NaN, the corresponding result element becomes NaN unless later post‑processing masks or replaces them.

Output
------
A `float64` NumPy array of the same shape as `a` and `b`.

Numerical Stability
-------------------
Small denominators: if `|a + b| < 1e-10`, the output element is set to `0.0`.
This avoids extreme values for near‑cancelling inputs.

Performance Notes
-----------------
- Implemented in Rust using the `ndarray` crate with fused iteration (no intermediate arrays).
- GIL is released during execution.
- For larger arrays (particularly 3D/4D) memory access is sequential, improving cache locality compared to chained pure NumPy expressions `(a - b) / (a + b)` which create 2 temporaries.

Examples
--------
Basic 1D usage:

.. code-block:: python

   import numpy as np
   from eo_processor import normalized_difference

   nir = np.array([0.8, 0.7, 0.6])
   red = np.array([0.2, 0.1, 0.3])
   ndvi_like = normalized_difference(nir, red)
   print(ndvi_like)

2D image:

.. code-block:: python

   import numpy as np
   from eo_processor import normalized_difference

   a = np.random.rand(512, 512)
   b = np.random.rand(512, 512)
   nd = normalized_difference(a, b)
   assert nd.shape == a.shape

4D stack (e.g., (time, band, y, x)):

.. code-block:: python

   import numpy as np
   from eo_processor import normalized_difference

   a = np.random.rand(4, 3, 128, 128)
   b = np.random.rand(4, 3, 128, 128)
   out = normalized_difference(a, b)
   assert out.shape == (4, 3, 128, 128)

Edge case with near‑zero denominator:

.. code-block:: python

   import numpy as np
   from eo_processor import normalized_difference
   a = np.array([1e-12, 0.5])
   b = np.array([-1e-12, 0.5])
   # First pair sums to ~0 → safeguarded to 0.0
   out = normalized_difference(a, b)
   print(out)  # [0.0, 0.0]

Related Functions
-----------------
- `ndvi(nir, red)`: vegetation index built atop normalized difference.
- `ndwi(green, nir)`, `nbr(nir, swir2)`, `ndmi(nir, swir1)`, `nbr2(swir1, swir2)`: all reuse the normalized difference pattern.
- `delta_ndvi`, `delta_nbr`: compute change between two epochs (internally call normalized difference twice).
- `savi`, `enhanced_vegetation_index (evi)`: specialized ratios with soil/atmospheric adjustments.

Error Handling
--------------
Raises `ValueError` if shapes differ.
Raises `TypeError` if inputs cannot be interpreted as 1D–4D numeric arrays.

Testing Guarantees
------------------
The test suite validates:
- Correct equality with manual `(a - b) / (a + b)` for standard arrays.
- Antisymmetry: `normalized_difference(a, b) == -normalized_difference(b, a)`.
- Zero safeguarding behavior.
- Range adherence for index derivatives (e.g., NDVI bounds in dedicated tests).

Notes
-----
For spectral index semantics (NDVI, NDWI, etc.) prefer the dedicated named helpers; `normalized_difference` is ideal for generic band math or custom ratio experimentation.

End of normalized_difference reference.
