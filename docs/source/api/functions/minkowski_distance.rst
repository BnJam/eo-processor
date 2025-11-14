minkowski_distance
==================

.. currentmodule:: eo_processor

.. autofunction:: minkowski_distance

Overview
--------
Computes the full pairwise Minkowski distance matrix between two 2D point sets
`points_a` (N, D) and `points_b` (M, D) for a given order `p` (p ≥ 1).

The Minkowski distance of order `p` between vectors `x` and `y` is:

.. math::

   d_p(x, y) = \left( \sum_{i=1}^{D} |x_i - y_i|^p \right)^{1/p}

Special cases:
- p = 1 → Manhattan (L1) distance
- p = 2 → Euclidean (L2) distance
- p → ∞ (large p) approximates Chebyshev (L∞) distance

Input Requirements
------------------
- `points_a` and `points_b` must be 2-dimensional NumPy arrays with the same
  feature dimension D.
- Any numeric dtype is accepted; values are coerced to float64 internally.
- Raises `ValueError` if `p < 1.0`.

Output
------
Returns a float64 NumPy array of shape (N, M) where element (i, j) is the
Minkowski distance between `points_a[i]` and `points_b[j]`.

Performance Notes
-----------------
- For large N*M products, consider memory implications of a full matrix.
- Internally, parallelization may occur for sufficiently large workloads;
  small matrices avoid parallel overhead.
- If you need only nearest neighbors, building a spatial index (not provided
  here) can be more efficient than computing the full matrix.

Examples
--------
.. code-block:: python

    import numpy as np
    from eo_processor import minkowski_distance

    A = np.random.rand(5, 3)   # (N=5, D=3)
    B = np.random.rand(7, 3)   # (M=7, D=3)

    dist_p2 = minkowski_distance(A, B, p=2.0)  # Euclidean
    dist_p3 = minkowski_distance(A, B, p=3.0)

    print(dist_p2.shape)  # (5, 7)

Error Handling
--------------
- `ValueError` if shapes are incompatible (different D).
- `ValueError` if `p < 1.0`.

See Also
--------
- `euclidean_distance` (special case p=2)
- `manhattan_distance` (special case p=1)
- `chebyshev_distance` (L∞ variant)

End of minkowski_distance reference.
