mask_invalid
============

.. currentmodule:: eo_processor

.. autofunction:: mask_invalid

Overview
--------
`mask_invalid` masks a list of sentinel (invalid) numeric codes in an input array
(1D–4D). Masked positions are replaced by `NaN` (default) or a user‑provided
`fill_value`. The function returns a `float64` NumPy array (even if the input
dtype was integer) to preserve NaN semantics for downstream processing.

Typical Use Cases
-----------------
- Removing sensor error flags (e.g., 0 or -9999).
- Normalizing heterogeneous invalid code conventions before temporal statistics.
- Preparing arrays for compositing by stripping placeholder values.

Parameters
----------
- ``arr`` : numpy.ndarray
  Input array of shape (time, ...), accepting 1D, 2D, 3D, or 4D layouts. Any numeric dtype.
- ``invalid_values`` : Sequence[float]
  Iterable of numeric codes to treat as invalid (exact equality).
- ``fill_value`` : float, optional
  Replacement for invalid positions. Defaults to ``NaN`` when omitted.

Returns
-------
numpy.ndarray (float64)
Array matching input shape with invalid codes replaced by NaN or the specified
``fill_value``.

Behavior Notes
--------------
- Equality test is exact; supply the precise codes used in your dataset.
- Output is always ``float64`` (internal coercion) to guarantee NaN support.
- No in-place modification of the original input array occurs.

Examples
--------
Basic masking (default NaN):
::
    import numpy as np
    from eo_processor import mask_invalid

    arr = np.array([0, 1, -9999, 2], dtype=np.int32)
    out = mask_invalid(arr, invalid_values=[0, -9999])
    # out -> [nan, 1., nan, 2.]

Using a custom fill value:
::
    out = mask_invalid(arr, invalid_values=[0], fill_value=-1.0)
    # out -> [-1., 1., -9999., 2.]  (only 0 masked)

3D array (time, y, x):
::
    cube = np.array([
        [[0, 5], [6, -9999]],
        [[1, 0], [3, 4]],
    ])
    cleaned = mask_invalid(cube, invalid_values=[0, -9999])
    # All 0 and -9999 entries become NaN; shape preserved.

Edge Cases
----------
- If ``invalid_values`` is empty, the function returns a float64 copy of the input.
- Very large lists of invalid codes are handled by iterating per element; consider
  pre-filtering if performance becomes critical.
- Supplying values not present in the array yields a simple float64 copy.

Integration Tips
----------------
- Chain with `replace_nans(arr, value)` if you need a consistent sentinel
  after masking rather than NaN.
- Use prior to temporal aggregations (`temporal_mean`, `median`, etc.) to prevent
  invalid values from influencing statistics when `skip_na=True`.
- Combine with `mask_vals` when you need both equality masking and NaN normalization
  logic (e.g., replacing NaNs to a sentinel after masking).

Related Functions
-----------------
- ``mask_vals``: Masks exact codes with optional additional NaN normalization step.
- ``mask_out_range`` / ``mask_in_range``: Range-based masking.
- ``replace_nans``: Post-processing for NaN replacement.
- ``mask_scl``: Specialized Sentinel‑2 SCL masking.

Performance
-----------
For typical EO arrays (millions of pixels), masking runs in a single pass and allocates
one output array. There is no parallel overhead for small arrays; performance scales primarily
with total element count.

Version & Stability
-------------------
- Numeric coercion and masking semantics are stable across patch releases.
- Adding new masking functions will trigger a minor version bump per repository governance.

End of `mask_invalid` reference.
