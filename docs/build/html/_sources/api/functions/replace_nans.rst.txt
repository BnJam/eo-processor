replace_nans
============

.. currentmodule:: eo_processor

Overview
--------
`replace_nans` replaces every NaN (IEEE-754 floating point Not-a-Number) value in an input array
with a user-provided scalar. The function supports 1D–4D numeric NumPy arrays of any dtype; all
inputs are coerced to `float64` internally for consistent numerical behavior in the Rust layer.

Signature
---------
.. autofunction:: replace_nans

Parameters (Docstring Summary)
------------------------------
- ``arr`` : numpy.ndarray
  Input array (1D, 2D, 3D, or 4D). Any numeric dtype is accepted.
- ``value`` : float
  Scalar value to substitute for every NaN encountered.

Returns
-------
- numpy.ndarray (float64) with identical shape where all NaNs have been replaced by ``value``.

Behavior & Notes
----------------
- If the input contains no NaNs, the output is still a float64 copy (dtype coercion step).
- Replacement is unconditional—original finite values are left untouched.
- The operation is applied element-by-element in Rust (no Python loops).
- Time or spatial semantics are not interpreted; dimensions are passed through unchanged.
- Useful as a final cleanup step after masking operations that produced NaNs.

Example
-------
.. code-block:: python

   import numpy as np
   from eo_processor import replace_nans

   arr = np.array([[np.nan, 1.2], [3.4, np.nan]], dtype=np.float32)
   filled = replace_nans(arr, -9999.0)
   # filled -> array([[-9999.,    1.2],
   #                  [   3.4, -9999.]])

Interaction With Masking
------------------------
When combining with masking utilities:
1. Apply a mask (e.g., ``mask_vals`` or ``mask_out_range``).
2. Then call ``replace_nans`` to assign a sentinel value if downstream tools
   cannot handle NaNs directly.

See Also
--------
- :doc:`mask_vals <mask_vals>`
- :doc:`mask_out_range <mask_out_range>`
- :doc:`mask_in_range <mask_in_range>`
- :doc:`mask_invalid <mask_invalid>`
- :doc:`mask_scl <mask_scl>`

End of `replace_nans` reference.
