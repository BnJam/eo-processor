mask_in_range
=============

.. currentmodule:: eo_processor

.. autofunction:: mask_in_range

Detailed Description
--------------------
`mask_in_range` masks (replaces with NaN or a provided fill value) all elements whose numeric
value lies inside the closed interval `[min_val, max_val]`. Values outside that interval are
retained unchanged. Any combination of bounds may be omitted:

- If both `min_val` and `max_val` are provided: mask values `v` where `min_val <= v <= max_val`.
- If only `min_val` is provided: mask values `v` where `v >= min_val`.
- If only `max_val` is provided: mask values `v` where `v <= max_val`.
- If neither bound is provided: returns a float64 copy of the input (no masking).

All numeric input dtypes (int/uint/float) are coerced to `float64` internally to support NaN
semantics. Output always has dtype `float64` and the same shape as the input array (1D–4D).

Parameters
----------
arr : numpy.ndarray
    Input array (1D, 2D, 3D, or 4D). Any numeric dtype accepted.
min_val : float, optional
    Lower bound of interval to mask (inclusive). If None, no lower constraint.
max_val : float, optional
    Upper bound of interval to mask (inclusive). If None, no upper constraint.
fill_value : float, optional
    Replacement value for masked positions. Defaults to NaN when None.

Returns
-------
numpy.ndarray
    Float64 array with masked values replaced by `fill_value` or NaN.

Notes
-----
- Use `mask_out_range` to perform the complementary operation (mask values *outside* an interval).
- When neither bound is specified, the function performs a dtype normalization (int → float64) but
  no values are changed.
- Masking is performed in a single pass over the data; no intermediate allocations proportional
  to array size beyond the output buffer.
- For very large arrays you can chain with other masking utilities (`mask_vals`, `mask_invalid`)
  to build composite filters without constructing Boolean masks in Python.

Example
-------
.. code-block:: python

    import numpy as np
    from eo_processor import mask_in_range

    arr = np.array([-1.0, 0.5, 1.0, 1.5])
    # Mask values inside [0.0, 1.0]
    masked = mask_in_range(arr, min_val=0.0, max_val=1.0)
    # Result: [-1.0, nan, nan, 1.5]

    # Mask everything >= 2.0 (upper bound omitted)
    arr2 = np.array([1.0, 2.0, 3.0])
    masked2 = mask_in_range(arr2, min_val=2.0, fill_value=-9999.0)
    # Result: [1.0, -9999.0, -9999.0]

See Also
--------
- `mask_out_range` – inverse logic (mask outside an interval)
- `mask_vals` – exact value masking
- `mask_invalid` – masking of a list of sentinel codes
