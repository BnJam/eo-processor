mask_out_range
==============

.. currentmodule:: eo_processor

.. autofunction:: mask_out_range

Overview
--------
``mask_out_range`` masks (replaces) all values lying strictly outside an inclusive interval
defined by ``[min_val, max_val]``. The function supports 1D–4D numeric NumPy arrays
(any integer / unsigned / float dtype); inputs are coerced to ``float64`` internally.

If a bound is omitted (``None``), only the other bound is applied. By default masked
values are set to ``NaN``; a custom scalar can be supplied via ``fill_value``.

Typical Use Cases
-----------------
- Filtering physically implausible reflectance values (e.g., outside ``[0.0, 1.0]``)
- Removing extreme outliers before temporal aggregation
- Constraining intermediate computation results to expected ranges

Parameters
----------
- **arr** : numpy.ndarray (1D–4D)
  Input numeric array (time-first if >2D, but dimensional meaning does not affect masking).
- **min_val** : float | None, optional
  Minimum permitted value (inclusive). Values lower than this are masked. If ``None``, no lower bound is applied.
- **max_val** : float | None, optional
  Maximum permitted value (inclusive). Values greater than this are masked. If ``None``, no upper bound is applied.
- **fill_value** : float | None, optional
  Replacement for masked positions. Defaults to ``NaN`` when ``None``.

Returns
-------
numpy.ndarray (float64)
Same shape as ``arr`` with masked positions replaced by ``NaN`` or ``fill_value``.

Behavior Details
----------------
- Bounds are inclusive: values exactly equal to ``min_val`` or ``max_val`` are retained.
- Only values strictly outside the specified interval are masked.
- If both bounds are ``None``, the function returns a float64 copy of the input (no masking).
- Result dtype is always ``float64`` to preserve NaN semantics even if input was integer.

Numeric Considerations
----------------------
- There is no epsilon comparison; comparisons are direct (exact for integers, IEEE for floats).
- For floating point edge cases (e.g., extremely small negatives due to numerical noise),
  consider passing a slightly widened interval if needed.

Examples
--------
Basic range masking:

.. code-block:: python

    import numpy as np
    from eo_processor import mask_out_range

    arr = np.array([-1.0, 0.2, 0.8, 1.3], dtype=np.float32)
    out = mask_out_range(arr, min_val=0.0, max_val=1.0)
    # out -> [nan, 0.2, 0.8, nan]

Upper bound only:

.. code-block:: python

    arr = np.array([10, 15, 20], dtype=np.int16)
    out = mask_out_range(arr, max_val=15)
    # out -> [10.0, 15.0, nan]

Custom fill value:

.. code-block:: python

    arr = np.array([0.4, 0.5, 5.5, -2.0])
    out = mask_out_range(arr, min_val=0.0, max_val=1.0, fill_value=-9999.0)
    # out -> [0.4, 0.5, -9999.0, -9999.0]

No bounds (passthrough conversion to float64):

.. code-block:: python

    arr = np.array([1, 2, 3], dtype=np.uint8)
    out = mask_out_range(arr)
    # out.dtype == float64; values unchanged

Edge Cases
----------
- Passing ``min_val > max_val`` is not meaningful; current implementation does not reorder.
  (If needed, validate externally or submit an enhancement request.)
- Empty arrays return an empty float64 array.
- NaNs present in the input remain NaN (they are not considered “outside” the range).

Performance Notes
-----------------
Operation is executed in Rust with a single pass over the array:
- O(N) time where N is total number of elements.
- No temporary allocation besides the output buffer.
- Multi-dimensional arrays are traversed in memory order for good cache locality.

See Also
--------
- :func:`mask_in_range` – masks values *inside* an interval.
- :func:`mask_vals` – masks exact value codes.
- :func:`mask_invalid` – masks a list of sentinel values.
- :func:`replace_nans` – replaces all NaNs with a scalar.

Version
-------
Introduced prior to 0.6.0; documentation clarified in 0.6.0.

End of ``mask_out_range`` reference.
