chebyshev_distance
==================

.. currentmodule:: eo_processor

.. autofunction:: chebyshev_distance

Overview
--------
`chebyshev_distance(points_a, points_b)` computes the full pairwise Chebyshev (L∞) distance
matrix between two 2D numeric arrays of shapes `(N, D)` and `(M, D)`, returning an `(N, M)`
float64 matrix. The Chebyshev distance between vectors x and y is:

.. math::

   d_{\infty}(x, y) = \max_{i=1..D} |x_i - y_i|

It measures the maximum absolute coordinate difference and is useful when the largest
deviation across dimensions dominates similarity (e.g., certain quality thresholds or
grid-based neighborhood analyses).

Shape Requirements
------------------
- `points_a.ndim == 2`
- `points_b.ndim == 2`
- Feature dimension matches: `points_a.shape[1] == points_b.shape[1]`

Parameters (see signature above)
--------------------------------
- `points_a`: NumPy array `(N, D)` (any numeric dtype; coerced to float64)
- `points_b`: NumPy array `(M, D)` (same feature dimension as `points_a`)

Returns
-------
`numpy.ndarray` of shape `(N, M)` with `out[i, j] = max(abs(points_a[i, k] - points_b[j, k]))`.

Numerical & Dtype Notes
-----------------------
- All inputs are coerced to `float64` internally for uniform arithmetic.
- No special epsilon handling is required (pure absolute differences).
- Extremely large values propagate normally; consider prior clipping if needed.

Performance
-----------
- Implemented in Rust; releases the Python GIL.
- Outer loops may parallelize for large `N * M` workloads.
- Memory usage is `O(N*M)` for the output; for very large matrices consider blockwise strategies
  (not implemented here).

Examples
--------
Basic usage:

.. code-block:: python

    import numpy as np
    from eo_processor import chebyshev_distance

    A = np.array([[0.0, 1.0],
                  [2.0, 3.5]], dtype=np.float32)   # (2, 2)
    B = np.array([[1.0, 2.0],
                  [0.0, -1.0],
                  [2.0, 3.0]], dtype=np.float64)   # (3, 2)

    dist = chebyshev_distance(A, B)
    # dist[0, 0] = max(|0-1|, |1-2|) = 1
    # dist[1, 2] = max(|2-2|, |3.5-3|) = 0.5
    print(dist.shape)  # (2, 3)

Edge Case (single feature dimension):

.. code-block:: python

    A = np.array([[0.0], [5.0]])
    B = np.array([[2.5], [10.0]])
    d = chebyshev_distance(A, B)  # Equivalent to absolute difference

Error Handling
--------------
Raises `ValueError` if:
- Inputs are not 2D
- Feature dimensions differ

Raises `TypeError` if inputs are non-numeric.

Use Cases
---------
- Fast max-difference screening
- Grid / tile similarity where worst-case deviation matters
- Alternative to L2/L1 norms for bounding box or threshold checks

See Also
--------
- :func:`euclidean_distance` (L2 norm)
- :func:`manhattan_distance` (L1 norm)
- :func:`minkowski_distance` (general L^p; Chebyshev as p→∞)
- :func:`normalized_difference` (unrelated ratio primitive)

End of chebyshev_distance documentation.
