euclidean_distance
==================

.. currentmodule:: eo_processor

.. autofunction:: euclidean_distance

Synopsis
--------
`euclidean_distance(points_a, points_b)` computes the pairwise Euclidean (L2) distance
matrix between two 2D numeric arrays: `(N, D)` and `(M, D)` → `(N, M)`.

This is a Rust-accelerated implementation releasing the Python GIL. For large
inputs it uses internal parallelism (outer-loop style) when beneficial.

Shape Requirements
------------------
- `points_a.ndim == 2`
- `points_b.ndim == 2`
- `points_a.shape[1] == points_b.shape[1]` (matching feature dimension)

Parameters
----------
points_a : numpy.ndarray
    Array of shape `(N, D)` containing N points with D features.
points_b : numpy.ndarray
    Array of shape `(M, D)` containing M points with D features.

Returns
-------
numpy.ndarray
    Distance matrix of shape `(N, M)` where element `(i, j)` is
    `sqrt(sum_k (points_a[i, k] - points_b[j, k])**2)`.

Notes
-----
- All numeric dtypes are coerced to `float64` internally for stable arithmetic.
- If either input violates the 2D requirement or feature size mismatch, a `ValueError`
  is raised.
- Memory complexity is `O(N*M)`—be cautious with very large N or M (e.g. > 50k) to avoid
  excessive allocation.
- Consider block-wise or approximate methods for extreme sizes (e.g., spatial indexing,
  locality sensitive hashing). Not implemented here.

Performance
-----------
Overview:
The implementation uses an outer-loop strategy with optional parallelization (Rayon) for large
pair counts. All arithmetic is performed in fused loops, avoiding Python-level broadcasting and
reducing temporary allocations. Inputs are coerced once to float64.

Representative Benchmarks (Single Run)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Platform: macOS (ARM64), CPython 3.10, release build, time.perf_counter(), warm cache
Validation: np.allclose(..., atol=1e-9)

.. list-table:: Euclidean Distance Benchmarks (Single Run)
   :header-rows: 1

   * - Workload (A shape × B shape)
     - Rust (s)
     - NumPy (s)
     - Rust Throughput (M/s)
     - NumPy Throughput (M/s)
     - Speedup
     - Notes
   * - (1200, 32) × (1200, 32)
     - 0.005
     - 0.008
     - 288.00
     - 180.00
     - 1.44x
     - Smaller matrix; fused loops
   * - (3000, 64) × (3000, 64)
     - 0.063
     - 0.041
     - 142.86
     - 219.51
     - 0.65x
     - NumPy BLAS matmul advantage

Interpretation:

- For moderate dimensions with relatively small feature size, Rust can outperform a pure NumPy
  formulation due to low Python overhead and early parallel thresholds.
- For larger point sets with feature dimensions amenable to efficient BLAS matmul,
  the vectorized NumPy approach (A^2 + B^2 - 2 A B^T) can be faster.
- Performance crossover depends on (N * M), D, and available cores. Benchmark on your workload.

Reproduction Snippet:
.. code-block:: python

    import numpy as np, time
    from eo_processor import euclidean_distance

    A = np.random.rand(3000, 64)
    B = np.random.rand(3000, 64)

    t0 = time.perf_counter()
    rust_dist = euclidean_distance(A, B)
    rust_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    A_sq = (A**2).sum(axis=1)[:, None]
    B_sq = (B**2).sum(axis=1)[None, :]
    prod = A @ B.T
    numpy_dist = np.sqrt(np.clip(A_sq + B_sq - 2 * prod, 0, None))
    numpy_t = time.perf_counter() - t0

    print(f"Rust {rust_t:.3f}s vs NumPy {numpy_t:.3f}s speedup {numpy_t/rust_t:.2f}x")
    assert np.allclose(rust_dist, numpy_dist, atol=1e-9)

Guidance:
- Use rust implementation when memory for full pair matrix is acceptable and (N * M) is large
  enough for parallel benefits or when integrating consistently with other Rust-accelerated kernels.
- For extremely large N, M (risking RAM exhaustion), consider blockwise strategies or approximate
  nearest-neighbor methods (not yet implemented here).
- Always validate correctness with np.allclose before adopting benchmarks.

Performance Claim Template:
.. code-block:: text

    Benchmark:
    Shapes: (3000, 64) × (3000, 64)
    NumPy: 0.041s
    Rust: 0.063s
    Speedup: 0.65x (NumPy faster)
    Methodology: single run, time.perf_counter(), float64 arrays
    Validation: np.allclose(rust, numpy, atol=1e-9)

Claim Guidelines:
Report both faster and slower cases transparently. Provide shapes, methodology, and validation
tolerance. Prefer multiple runs or median timing for publication-quality claims.

Examples
--------
Basic usage with small arrays:

.. code-block:: python

    import numpy as np
    from eo_processor import euclidean_distance

    A = np.array([[0.0, 1.0],
                  [1.0, 2.0],
                  [2.0, 2.0]], dtype=np.float64)   # (3, 2)

    B = np.array([[0.0, 0.0],
                  [2.0, 1.0]], dtype=np.float64)   # (2, 2)

    dist = euclidean_distance(A, B)
    # dist.shape == (3, 2)
    # Each row i corresponds to A[i], columns correspond to B[j]

Large problem sketch (be cautious of memory):

.. code-block:: python

    import numpy as np
    from eo_processor import euclidean_distance

    # 10k x 64 and 8k x 64 -> 80 million distances (requires several hundred MB)
    A = np.random.rand(10_000, 64)
    B = np.random.rand(8_000, 64)
    dist = euclidean_distance(A, B)
    # Consider chunking if this is near memory limits.

See Also
--------
- :func:`manhattan_distance` (L1 norm)
- :func:`chebyshev_distance` (L∞ norm)
- :func:`minkowski_distance` (general L^p)
- :func:`normalized_difference` (band-wise index primitive, unrelated but shared numeric stability patterns)

Error Handling
--------------
Raises `ValueError` for:
- Non-2D inputs
- Feature dimension mismatch

Raises `TypeError` if inputs are not interpretable as numeric arrays.

End of euclidean_distance documentation.
