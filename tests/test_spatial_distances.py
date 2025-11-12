import numpy as np
import pytest

from eo_processor import (
    euclidean_distance,
    manhattan_distance,
    chebyshev_distance,
    minkowski_distance,
)


def _manual_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            diff = a[i] - b[j]
            out[i, j] = np.sqrt(np.sum(diff * diff))
    return out


def _manual_manhattan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            out[i, j] = np.sum(np.abs(a[i] - b[j]))
    return out


def _manual_chebyshev(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            out[i, j] = np.max(np.abs(a[i] - b[j]))
    return out


def test_pairwise_distance_values_small():
    a = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    eu = euclidean_distance(a, b)
    ma = manhattan_distance(a, b)
    ch = chebyshev_distance(a, b)
    mi_p2 = minkowski_distance(a, b, 2.0)

    eu_manual = _manual_euclidean(a, b)
    ma_manual = _manual_manhattan(a, b)
    ch_manual = _manual_chebyshev(a, b)

    assert eu.shape == (2, 2)
    assert np.allclose(eu, eu_manual, atol=1e-12)
    assert np.allclose(ma, ma_manual, atol=1e-12)
    assert np.allclose(ch, ch_manual, atol=1e-12)
    # Minkowski p=2 should equal Euclidean
    assert np.allclose(mi_p2, eu, atol=1e-12)


def test_minkowski_equivalence_p1_p2():
    rng = np.random.default_rng(123)
    a = rng.normal(size=(5, 3)).astype(np.float64)
    b = rng.normal(size=(4, 3)).astype(np.float64)

    man = manhattan_distance(a, b)
    eu = euclidean_distance(a, b)
    mink_p1 = minkowski_distance(a, b, 1.0)
    mink_p2 = minkowski_distance(a, b, 2.0)

    assert np.allclose(mink_p1, man, atol=1e-12)
    assert np.allclose(mink_p2, eu, atol=1e-12)


def test_norm_inequalities():
    rng = np.random.default_rng(999)
    a = rng.uniform(-1.0, 1.0, size=(7, 5)).astype(np.float64)
    b = rng.uniform(-1.0, 1.0, size=(6, 5)).astype(np.float64)

    eu = euclidean_distance(a, b)
    man = manhattan_distance(a, b)
    ch = chebyshev_distance(a, b)

    # Chebyshev <= Euclidean <= Manhattan element-wise
    assert np.all(ch <= eu + 1e-12)
    assert np.all(eu <= man + 1e-12)


def test_distance_matrix_symmetry_same_set():
    # When A == B, distance matrix should be symmetric
    rng = np.random.default_rng(42)
    pts = rng.normal(size=(8, 4)).astype(np.float64)

    eu_same = euclidean_distance(pts, pts)
    man_same = manhattan_distance(pts, pts)
    ch_same = chebyshev_distance(pts, pts)
    mink_same = minkowski_distance(pts, pts, 2.0)

    for mat in (eu_same, man_same, ch_same, mink_same):
        assert mat.shape == (pts.shape[0], pts.shape[0])
        assert np.allclose(mat, mat.T, atol=1e-12)
        # Self-distances are zero
        assert np.allclose(np.diag(mat), 0.0, atol=1e-12)


def test_single_point_shapes():
    a = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)  # shape (1,3)
    b = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=np.float64)  # shape (2,3)

    eu = euclidean_distance(a, b)
    man = manhattan_distance(a, b)
    ch = chebyshev_distance(a, b)
    mi = minkowski_distance(a, b, 3.0)

    assert eu.shape == (1, 2)
    assert man.shape == (1, 2)
    assert ch.shape == (1, 2)
    assert mi.shape == (1, 2)
    # Check non-negative
    assert np.all(eu >= 0)
    assert np.all(man >= 0)
    assert np.all(ch >= 0)
    assert np.all(mi >= 0)


def test_random_large_consistency():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(50, 3)).astype(np.float64)
    b = rng.normal(size=(60, 3)).astype(np.float64)

    eu = euclidean_distance(a, b)
    man = manhattan_distance(a, b)
    ch = chebyshev_distance(a, b)
    mink = minkowski_distance(a, b, 2.0)

    assert eu.shape == (50, 60)
    assert not np.isnan(eu).any()
    assert np.allclose(eu, mink, atol=1e-12)
    # Norm relationships still hold statistically
    assert np.all(ch <= eu + 1e-12)
    assert np.all(eu <= man + 1e-12)


def test_minkowski_monotonic_in_p():
    rng = np.random.default_rng(101)
    a = rng.uniform(-2.0, 2.0, size=(10, 5)).astype(np.float64)
    b = rng.uniform(-2.0, 2.0, size=(12, 5)).astype(np.float64)

    d_p1 = minkowski_distance(a, b, 1.0)  # L1
    d_p2 = minkowski_distance(a, b, 2.0)  # L2
    d_p3 = minkowski_distance(a, b, 3.0)
    d_p10 = minkowski_distance(a, b, 10.0)  # approaches L∞

    # For fixed vectors: ||x||_p is non-increasing in p
    assert np.all(d_p1 >= d_p2 - 1e-12)
    assert np.all(d_p2 >= d_p3 - 1e-12)
    assert np.all(d_p3 >= d_p10 - 1e-12)


def test_minkowski_invalid_p():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(3, 2)).astype(np.float64)
    b = rng.normal(size=(4, 2)).astype(np.float64)
    # p < 1.0 should raise ValueError per updated implementation
    with pytest.raises(ValueError):
        minkowski_distance(a, b, 0.5)


def test_zero_vector_distances():
    zero = np.zeros((4, 3), dtype=np.float64)
    eu = euclidean_distance(zero, zero)
    man = manhattan_distance(zero, zero)
    ch = chebyshev_distance(zero, zero)
    mi = minkowski_distance(zero, zero, 2.0)
    assert np.allclose(eu, 0.0)
    assert np.allclose(man, 0.0)
    assert np.allclose(ch, 0.0)
    assert np.allclose(mi, 0.0)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
