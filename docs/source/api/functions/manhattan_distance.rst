manhattan_distance
==================

.. currentmodule:: eo_processor

.. autofunction:: manhattan_distance

Description
-----------
Computes pairwise Manhattan (L1) distances between two 2D point sets:

Given two input arrays:
- ``points_a`` with shape ``(N, D)``
- ``points_b`` with shape ``(M, D)``

The output is a 2D array of shape ``(N, M)`` where element ``(i, j)`` is:

.. math::

   \sum_{k=1}^{D} | points\_a[i, k] - points\_b[j, k] |

Use this when L1 distance (sum of absolute coordinate differences) is more robust to outliers than Euclidean distance.

Parameters (see full docstring via autodoc)
-------------------------------------------
- ``points_a``: NumPy array, shape ``(N, D)``
- ``points_b``: NumPy array, shape ``(M, D)``

Returns
-------
``numpy.ndarray`` of shape ``(N, M)`` (float64). Each row corresponds to a point in ``points_a``, each column to a point in ``points_b``.

Constraints & Error Handling
----------------------------
- Both inputs must be 2D and have the same feature dimension ``D``.
- A mismatch in ``D`` raises a ``ValueError``.
- Non-numeric dtypes are coerced internally to ``float64``.

Example
-------
.. code-block:: python

    import numpy as np
    from eo_processor import manhattan_distance

    A = np.array([[0.0, 1.0],
                  [2.0, 3.0]])          # (N=2, D=2)
    B = np.array([[1.0, 2.0],
                  [0.0, -1.0],
                  [2.0, 2.0]])          # (M=3, D=2)

    dist = manhattan_distance(A, B)
    # dist shape: (2, 3)
    # dist[0, 0] = |0-1| + |1-2| = 2
    # dist[1, 2] = |2-2| + |3-2| = 1
    print(dist)

Typical Use Cases
-----------------
- Clustering or nearest neighbor queries under L1 norm.
- Robust distance comparisons when each dimension is independent and outliers should have linear, not quadratic, effect.

Performance Notes
-----------------
For large ``N * M`` the Rust implementation may parallelize outer loops (depending on internal thresholds) while avoiding excessive temporary allocations.

Related Functions
-----------------
- ``euclidean_distance``: L2 (squared sum of differences) norm.
- ``chebyshev_distance``: L∞ (max absolute difference) norm.
- ``minkowski_distance``: General L^p norm (p ≥ 1).

End of manhattan_distance reference.
