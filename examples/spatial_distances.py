"""
Spatial distance function examples for eo-processor.

These examples demonstrate use of internal spatial distance utilities
implemented in Rust and exposed through the private `_core` module.

Currently, distance functions are not re-exported at the top level of the
package. You can access them via:

    from eo_processor import _core
    _core.euclidean_distance(...)

Functions demonstrated:
    - euclidean_distance(points_a, points_b)
    - manhattan_distance(points_a, points_b)
    - chebyshev_distance(points_a, points_b)
    - minkowski_distance(points_a, points_b, p)

Each function expects two 2D NumPy arrays:
    points_a: shape (N, D)
    points_b: shape (M, D)

Returns:
    A 2D array of shape (N, M) with pairwise distances.

Notes:
    - For large N and M these computations are O(N*M*D).
    - Consider spatial indexing or tiling strategies for very large datasets.
    - Minkowski with p=1 equals Manhattan; with p approaching infinity tends
      toward Chebyshev; with p=2 equals Euclidean.
"""

from __future__ import annotations

import numpy as np

# Import internal core module (distance functions live here)
from eo_processor import _core

def demo_point_to_point_distance():
    print("Demo 0a: Distance from single point to single point")
    print("-" * 40)
    point_a = np.array([[1.0, 2.0]], dtype=np.float64)  # Single point (1, D)
    point_b = np.array([[4.0, 6.0]], dtype=np.float64)  # Single point (1, D)

    dist_euclid = _core.euclidean_distance(point_a, point_b)
    dist_manhat = _core.manhattan_distance(point_a, point_b)
    dist_cheby = _core.chebyshev_distance(point_a, point_b)

    print("point_a:\n", point_a)
    print("point_b:\n", point_b)
    print("Euclidean distance:\n", dist_euclid)
    print("Manhattan distance:\n", dist_manhat)
    print("Chebyshev distance:\n", dist_cheby)
    print()

def demo_point_to_array_distance():
    print("Demo 0: Distance from single point to array of points")
    print("-" * 40)
    point = np.array([[0.0, 0.0]], dtype=np.float64)  # Single point (1, D)
    points_b = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )  # Multiple points (M, D)

    dist_euclid = _core.euclidean_distance(point, points_b)
    dist_manhat = _core.manhattan_distance(point, points_b)
    dist_cheby = _core.chebyshev_distance(point, points_b)

    print("point:\n", point)
    print("points_b:\n", points_b)
    print("Euclidean distances:\n", dist_euclid)
    print("Manhattan distances:\n", dist_manhat)
    print("Chebyshev distances:\n", dist_cheby)
    print()

def demo_small_points():
    print("Demo 1: Small 2D point sets")
    print("-" * 40)
    points_a = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    )
    points_b = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )

    dist_euclid = _core.euclidean_distance(points_a, points_b)
    dist_manhat = _core.manhattan_distance(points_a, points_b)
    dist_cheby = _core.chebyshev_distance(points_a, points_b)
    dist_mink_p2 = _core.minkowski_distance(points_a, points_b, 2.0)

    print("points_a:\n", points_a)
    print("points_b:\n", points_b)
    print("Euclidean distances:\n", dist_euclid)
    print("Manhattan distances:\n", dist_manhat)
    print("Chebyshev distances:\n", dist_cheby)
    print("Minkowski(p=2) distances:\n", dist_mink_p2)
    # Consistency checks (allow tiny floating differences)
    print("Check: euclidean == minkowski(p=2):",
          np.allclose(dist_euclid, dist_mink_p2, atol=1e-12))
    print()


def demo_random_points(n: int = 5, m: int = 4, dim: int = 3, seed: int = 42):
    print("Demo 2: Random points in R^D")
    print("-" * 40)
    rng = np.random.default_rng(seed)
    points_a = rng.uniform(-1.0, 1.0, size=(n, dim)).astype(np.float64)
    points_b = rng.uniform(-1.0, 1.0, size=(m, dim)).astype(np.float64)

    dist_euclid = _core.euclidean_distance(points_a, points_b)
    dist_manhat = _core.manhattan_distance(points_a, points_b)
    dist_cheby = _core.chebyshev_distance(points_a, points_b)
    dist_mink_p3 = _core.minkowski_distance(points_a, points_b, 3.0)

    print(f"Generated points_a (n={n}, dim={dim})")
    print(f"Generated points_b (m={m}, dim={dim})")
    print("Euclidean summary: min={:.4f} max={:.4f} mean={:.4f}".format(
        dist_euclid.min(), dist_euclid.max(), dist_euclid.mean()))
    print("Manhattan summary: min={:.4f} max={:.4f} mean={:.4f}".format(
        dist_manhat.min(), dist_manhat.max(), dist_manhat.mean()))
    print("Chebyshev summary: min={:.4f} max={:.4f} mean={:.4f}".format(
        dist_cheby.min(), dist_cheby.max(), dist_cheby.mean()))
    print("Minkowski(p=3) summary: min={:.4f} max={:.4f} mean={:.4f}".format(
        dist_mink_p3.min(), dist_mink_p3.max(), dist_mink_p3.mean()))
    print()

    # Inequality relationships:
    # For p >= 1: Chebyshev <= Minkowski(p) <= Manhattan (not strict; depends on distribution).
    # Actually for any single pair (x,y):
    #   Chebyshev <= Euclidean <= Manhattan
    # We demonstrate general range bounds.
    print("Pairwise comparison checks:")
    print("All(Euclidean <= Manhattan):",
          np.all(dist_euclid <= dist_manhat + 1e-12))
    print("All(Chebyshev <= Euclidean):",
          np.all(dist_cheby <= dist_euclid + 1e-12))
    print()


def demo_high_dimension(dim: int = 16):
    print("Demo 3: Higher-dimensional behavior (dim={})".format(dim))
    print("-" * 40)
    rng = np.random.default_rng(7)
    points_a = rng.normal(loc=0.0, scale=1.0, size=(3, dim)).astype(np.float64)
    points_b = rng.normal(loc=0.0, scale=1.0, size=(3, dim)).astype(np.float64)

    dist_euclid = _core.euclidean_distance(points_a, points_b)
    dist_manhat = _core.manhattan_distance(points_a, points_b)
    dist_cheby = _core.chebyshev_distance(points_a, points_b)

    # Demonstrate growth of Manhattan relative to Euclidean and Chebyshev
    ratio_mean_euclid_manhat = dist_manhat.mean() / dist_euclid.mean()
    ratio_mean_euclid_cheby = dist_euclid.mean() / dist_cheby.mean()

    print("Mean distances:")
    print("  Euclidean: {:.4f}".format(dist_euclid.mean()))
    print("  Manhattan: {:.4f}".format(dist_manhat.mean()))
    print("  Chebyshev: {:.4f}".format(dist_cheby.mean()))
    print("Manhattan / Euclidean mean ratio: {:.3f}".format(ratio_mean_euclid_manhat))
    print("Euclidean / Chebyshev mean ratio: {:.3f}".format(ratio_mean_euclid_cheby))
    print()


def demo_minkowski_transitions(p_values=(1.0, 2.0, 3.0, 10.0)):
    print("Demo 4: Minkowski transitions across p")
    print("-" * 40)
    rng = np.random.default_rng(9)
    points_a = rng.uniform(-0.5, 0.5, size=(4, 5)).astype(np.float64)
    points_b = rng.uniform(-0.5, 0.5, size=(3, 5)).astype(np.float64)

    dist_sets = {}
    for p in p_values:
        dist_sets[p] = _core.minkowski_distance(points_a, points_b, p)

    manhattan_ref = _core.manhattan_distance(points_a, points_b)
    euclid_ref = _core.euclidean_distance(points_a, points_b)
    cheby_ref = _core.chebyshev_distance(points_a, points_b)

    print("Sanity relationships:")
    print("p=1 (Minkowski) == Manhattan:",
          np.allclose(dist_sets[1.0], manhattan_ref, atol=1e-12))
    print("p=2 (Minkowski) == Euclidean:",
          np.allclose(dist_sets[2.0], euclid_ref, atol=1e-12))
    # Large p approximates Chebyshev (not exact for finite p)
    print("p=10 approximates Chebyshev (mean abs diff):",
          np.abs(dist_sets[10.0] - cheby_ref).mean())
    print()

    print("Distance mean progression by p:")
    for p in p_values:
        print(f"  p={p:>4}: mean={dist_sets[p].mean():.6f}")
    print()


def main():
    print("=" * 60)
    print("Spatial Distance Examples (internal `_core` usage)")
    print("=" * 60)
    print()

    demo_point_to_point_distance()
    demo_point_to_array_distance()
    demo_small_points()
    demo_random_points()
    demo_high_dimension()
    demo_minkowski_transitions()

    print("All spatial distance examples completed successfully.")
    print()


if __name__ == "__main__":
    main()
