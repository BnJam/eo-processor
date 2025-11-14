mask_vals
=========

.. currentmodule:: eo_processor

.. autofunction:: mask_vals

Overview
--------
`mask_vals` applies exact-value masking to an input NumPy array (1D–4D). It replaces any values
found in the supplied `values` sequence with either `fill_value` (if provided) or `NaN`. After masking,
it can optionally normalize all NaNs (both original and newly created) to a single scalar using `nan_to`.

Key Behaviors
-------------
- Input may be any numeric dtype; internal Rust kernel coerces to `float64`.
- Returned array is always `float64` to preserve NaN semantics.
- Works for 1D, 2D, 3D, and 4D arrays without shape-dependent logic at the Python layer.
- If `values` is `None` or an empty list/tuple, the function performs only dtype coercion (and optional `nan_to` handling).
- If both `fill_value` and `nan_to` are specified, masking is applied first (masked values → `fill_value` or NaN), then all NaNs are replaced by `nan_to`.

Parameters (Recap)
------------------
- `arr`: Input array (numeric, 1–4 dimensions).
- `values`: Sequence of numeric codes to mask (e.g., `[0, -9999]`). `None` or empty sequence means no masking.
- `fill_value`: Scalar value written where codes match. Defaults to NaN when omitted.
- `nan_to`: Scalar used to replace every NaN after masking (including original NaNs).

Return
------
- A `numpy.ndarray` (`float64`) of the same shape as `arr`, with requested values masked.

Examples
--------
Basic masking:
.. code-block:: python

    import numpy as np
    from eo_processor import mask_vals

    arr = np.array([0, 1, 2, 0, 3], dtype=np.int16)
    out = mask_vals(arr, values=[0])
    # Positions with 0 become NaN
    assert np.isnan(out[[0, 3]]).all()
    assert np.array_equal(out[[1, 2, 4]], np.array([1.0, 2.0, 3.0]))

Fill value instead of NaN:
.. code-block:: python

    arr = np.array([0, 10, 0], dtype=np.int32)
    out = mask_vals(arr, values=[0], fill_value=-1.0)
    # -> [-1.0, 10.0, -1.0]

Normalize NaNs after masking:
.. code-block:: python

    arr = np.array([0, np.nan, 5], dtype=np.float64)
    out = mask_vals(arr, values=[0], nan_to=-9999.0)
    # 0 -> NaN; all NaNs -> -9999.0
    # -> [-9999.0, -9999.0, 5.0]

Combined fill and nan_to:
.. code-block:: python

    arr = np.array([0, 1, np.nan])
    out = mask_vals(arr, values=[0], fill_value=np.nan, nan_to=0.0)
    # 0 -> NaN then all NaNs -> 0.0
    # -> [0.0, 1.0, 0.0]

3D / 4D support:
.. code-block:: python

    cube = np.array([
        [[0, 1], [2, 0]],
        [[3, 4], [0, 5]],
    ], dtype=np.int16)  # shape (time, y, x)
    masked = mask_vals(cube, values=[0])
    # All zeros become NaN; dtype coerced to float64.

Notes
-----
1. Idempotent when `values` does not appear in `arr` (only dtype coercion occurs).
2. Providing `nan_to` without `values` acts as a uniform NaN replacement step.
3. Performance: implemented in Rust with a single pass; no intermediate Python loops.
4. Memory: returns a new array; original input is never modified in place.
5. Numeric stability: NaNs are standard IEEE-754; comparisons against `values` occur after coercion to `float64`.

Edge Cases
----------
- Empty `values` sequence: treated the same as `values=None` (no masking).
- Large `values` list: linear membership checks; prefer a small set of sentinel codes.
- Mixed integer/float sentinel list: all converted to `float64` for comparison.

See Also
--------
- `replace_nans` – Unified NaN replacement when masking is not needed.
- `mask_out_range` / `mask_in_range` – Range-based masking.
- `mask_invalid` – Convenience wrapper for a list of invalid sentinel values.
- `mask_scl` – Sentinel-2 Scene Classification Layer masking.

Testing Guidance
----------------
Recommended assertions:
- Shape preserved: `out.shape == arr.shape`
- Dtype: `out.dtype == np.float64`
- Correct NaN placement: `np.isnan(out[idx])` for each masked index
- `nan_to` effects: no remaining NaNs if `nan_to` provided

End of `mask_vals` reference.
