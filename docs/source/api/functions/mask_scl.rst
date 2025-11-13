mask_scl
========

.. currentmodule:: eo_processor
.. autofunction:: mask_scl

Overview
--------
`mask_scl` applies a semantic filter to a Sentinel‑2 Scene Classification Layer (SCL) array,
keeping only the classes you specify and masking all others to either `NaN` (default) or a
custom `fill_value`. The output is always `float64` to preserve `NaN` semantics even if the
input array is an integer dtype.

Default Keep Codes
------------------
If `keep_codes` is not provided, the function preserves the following SCL classes:

- 4: Vegetation
- 5: Not Vegetated (bare soil / built surfaces)
- 6: Water
- 7: Unclassified (often usable surface)
- 11: Snow / Ice

All other codes are masked.

Typical Sentinel‑2 SCL Codes
----------------------------
For reference (not all are kept by default):

+------+-------------------------------------------+
| Code | Meaning                                   |
+------+-------------------------------------------+
| 0    | No Data                                   |
| 1    | Saturated / Defective                     |
| 2    | Dark Area Pixels                          |
| 3    | Cloud Shadows                             |
| 4    | Vegetation                                |
| 5    | Not Vegetated                             |
| 6    | Water                                     |
| 7    | Unclassified                              |
| 8    | Cloud (Medium Probability)                |
| 9    | Cloud (High Probability)                  |
| 10   | Thin Cirrus                               |
| 11   | Snow / Ice                                |
+------+-------------------------------------------+

Parameters
----------
- `scl` (`numpy.ndarray`): Input SCL codes (any numeric dtype). Supported ranks: 1D–4D.
- `keep_codes` (`Sequence[int] | None`): List of codes to keep. Others are masked.
  Defaults to `[4, 5, 6, 7, 11]`.
- `fill_value` (`float | None`): Value written to masked positions. If `None`, masked
  positions become `NaN`.

Returns
-------
`numpy.ndarray` (float64) with the same shape as `scl`; retained codes converted to float,
masked codes set to `NaN` or `fill_value`.

Behavior & Notes
----------------
- Input is coerced to float64 in the Rust layer for consistent numeric behavior.
- If `keep_codes` is an empty sequence, all values are masked.
- If a code appears multiple times in `keep_codes`, duplicates are ignored (set semantics).
- Passing a `fill_value` preserves non-kept code positions with that scalar instead of `NaN`.
- The function does not perform validation of whether codes are valid Sentinel‑2 SCL values; any integers can be treated as categorical codes.

Examples
--------
Basic usage with defaults (keeps vegetation, water, etc.):

.. code-block:: python

   import numpy as np
   from eo_processor import mask_scl

   scl = np.array([0,4,5,6,7,8,9,10,11])
   masked = mask_scl(scl)
   # masked -> [nan,4.,5.,6.,7.,nan,nan,nan,11.]

Custom selection (only vegetation + water):

.. code-block:: python

   scl = np.array([4,5,6,8,9])
   keep = [4,6]
   masked = mask_scl(scl, keep_codes=keep)
   # -> [4., nan, 6., nan, nan]

Using a fill value instead of NaN:

.. code-block:: python

   scl = np.array([3,4,9,11])
   masked = mask_scl(scl, keep_codes=[4,11], fill_value=-1.0)
   # -> [-1.0, 4.0, -1.0, 11.0]

Edge Case (empty keep list):

.. code-block:: python

   scl = np.array([4,5,6])
   masked = mask_scl(scl, keep_codes=[], fill_value=np.nan)
   # All values masked -> [nan, nan, nan]

Performance Considerations
--------------------------
- The implementation performs a single pass over the array.
- Parallel dispatch may be enabled internally for higher-dimensional arrays when large enough.
- Memory allocation is limited to the output buffer; no auxiliary large temporaries are created.

Error Handling
--------------
- Shape is preserved exactly; no broadcasting.
- If `keep_codes` contains values not present in `scl`, they are simply ignored (no warning).

See Also
--------
- `mask_vals`: Mask arbitrary exact value codes (general-purpose).
- `mask_out_range`: Mask values strictly outside a numeric interval.
- `mask_in_range`: Mask values strictly inside a numeric interval.
- `replace_nans`: Replace all `NaN` occurrences with a scalar.

End of `mask_scl` reference.
