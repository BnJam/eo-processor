composite
=========

.. currentmodule:: eo_processor

.. autofunction:: composite

Overview
--------
`composite(arr, method="median", **kwargs)` reduces a time‑first stack (1D–4D) along its leading
axis using a named statistical reducer. It is a convenience wrapper around individually
documented reducers (currently the temporal median). Additional methods (e.g., mean, std,
percentiles) may be added in future minor versions.

Supported Input Ranks (time-first)
----------------------------------
- 1D: (time,)
- 2D: (time, feature|band)
- 3D: (time, y, x)
- 4D: (time, band, y, x)

The output removes the leading time axis and preserves remaining axes (same behavior
as the underlying reducer function).

Current Methods
---------------
- "median" → Calls :func:`median` (temporal median with optional NaN skipping)

Parameters (summary)
--------------------
- `arr`: NumPy array (numeric, 1–4 dimensions). Leading axis interpreted as time.
- `method`: String name of reducer. Currently `"median"` (default).
- `skip_na` (passed through when supported): For `"median"`, whether to ignore NaNs (defaults True).

Returns
-------
NumPy float64 array (or scalar for 1D input) with time axis removed.

NaN Handling
------------
Delegated to the underlying reducer. For `"median"`:
- `skip_na=True`: Excludes NaNs per pixel/band; all-NaN series become NaN.
- `skip_na=False`: Any NaN in the series propagates NaN.

Usage Examples
--------------
Basic median composite of a 3D cube:
.. code-block:: python

    import numpy as np
    from eo_processor import composite

    cube = np.random.rand(12, 256, 256)  # (time, y, x)
    med = composite(cube, method="median")
    print(med.shape)  # (256, 256)

Median composite with NaN skipping disabled:
.. code-block:: python

    noisy = cube.copy()
    noisy[3, 100:120, 140:160] = np.nan
    med_prop = composite(noisy, method="median", skip_na=False)

1D series:
.. code-block:: python

    ts = np.array([1.0, 5.0, 3.0, 2.0])
    med_scalar = composite(ts)  # method="median"
    print(med_scalar)

Extension Guidelines
--------------------
When new methods are added they must:
1. Match the time‑first axis removal semantics.
2. Validate rank (1–4) and dtype coercion to float64.
3. Respect NaN handling flags (`skip_na` or equivalent).
4. Be documented in this page under "Current Methods".
5. Include tests mirroring existing median coverage.

Performance Notes
-----------------
- Underlying reducers (e.g., :func:`median`) are implemented in Rust with parallel
  dispatch for higher ranks where beneficial.
- Wrapper adds negligible overhead (simple method dispatch).

Error Handling
--------------
- Unsupported `method` value → `ValueError` with available methods listed.
- Non 1D–4D numeric input → `TypeError` (via underlying reducer).
- Missing keyword parameters for new methods will raise errors once they are introduced.

See Also
--------
- :func:`median` – Core temporal median reducer.
- :func:`temporal_mean` – Direct mean (not yet exposed via `composite` if absent).
- :func:`temporal_std`
- Masking utilities (`mask_vals`, `mask_invalid`) for pre‑composite cleanup.

Version Notes
-------------
This documentation reflects the API as of version 0.6.0 with only `"median"` available.
Adding another reducer will require a minor version bump per governance rules.

End of composite reference.
