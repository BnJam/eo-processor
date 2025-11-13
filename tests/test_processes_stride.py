import numpy as np
import pytest
from eo_processor import (
    moving_average_temporal,
    moving_average_temporal_stride,
)

# ---------------------------------------------------------------------------
# Helper reference implementations (pure NumPy/Python)
# ---------------------------------------------------------------------------

def _naive_moving_average_same(series: np.ndarray, window: int, skip_na: bool) -> np.ndarray:
    """
    Naive O(T*W) moving average with 'same' mode edge shrinking.
    Window centered: half_left = window // 2; half_right = window - half_left - 1.
    """
    T = series.shape[0]
    out = np.empty(T, dtype=float)
    half_left = window // 2
    half_right = window - half_left - 1
    for t in range(T):
        start = max(0, t - half_left)
        end = min(T - 1, t + half_right)
        window_vals = series[start:end + 1]
        if skip_na:
            valid = window_vals[~np.isnan(window_vals)]
            out[t] = np.nan if valid.size == 0 else valid.mean()
        else:
            out[t] = np.nan if np.isnan(window_vals).any() else window_vals.mean()
    return out


def _naive_moving_average_valid(series: np.ndarray, window: int, skip_na: bool) -> np.ndarray:
    """
    Naive moving average 'valid' mode (no edge shrink).
    """
    T = series.shape[0]
    out_len = T - window + 1
    out = np.empty(out_len, dtype=float)
    for i in range(out_len):
        window_vals = series[i:i + window]
        if skip_na:
            valid = window_vals[~np.isnan(window_vals)]
            out[i] = np.nan if valid.size == 0 else valid.mean()
        else:
            out[i] = np.nan if np.isnan(window_vals).any() else window_vals.mean()
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_moving_average_temporal_stride_basic_1d():
    series = np.array([1., 2., 3., 4., 5., 6.])
    window = 3
    stride = 2
    rust = moving_average_temporal_stride(series, window=window, stride=stride, mode="same")
    # Build naive same-mode then stride sample
    naive_full = _naive_moving_average_same(series, window, skip_na=True)
    expected = naive_full[::stride]
    assert rust.shape == expected.shape
    assert np.allclose(rust, expected, atol=1e-12)


def test_moving_average_temporal_stride_same_vs_direct_ma():
    series = np.random.rand(25)
    window = 5
    stride = 3
    full_rust = moving_average_temporal(series, window=window, mode="same")
    stride_rust = moving_average_temporal_stride(series, window=window, stride=stride, mode="same")
    assert np.allclose(stride_rust, full_rust[::stride], atol=1e-12)


def test_moving_average_temporal_stride_valid_mode():
    series = np.array([1., 2., 3., 4., 5., 6., 7.])
    window = 3
    stride = 2
    rust_valid = moving_average_temporal_stride(series, window=window, stride=stride, mode="valid")
    naive_valid = _naive_moving_average_valid(series, window, skip_na=True)
    expected = naive_valid[::stride]
    assert rust_valid.shape == expected.shape
    assert np.allclose(rust_valid, expected, atol=1e-12)


def test_moving_average_temporal_stride_nan_skip():
    series = np.array([1., np.nan, 3., 4., np.nan, 6.])
    window = 3
    stride = 2
    rust = moving_average_temporal_stride(series, window=window, stride=stride, skip_na=True, mode="same")
    naive_full = _naive_moving_average_same(series, window, skip_na=True)
    expected = naive_full[::stride]
    # NaNs in expected propagate; use nan-aware comparison
    assert rust.shape == expected.shape
    assert np.allclose(rust, expected, atol=1e-12, equal_nan=True)


def test_moving_average_temporal_stride_nan_no_skip():
    series = np.array([1., np.nan, 3., 4., 5., 6.])
    window = 3
    stride = 1
    rust = moving_average_temporal_stride(series, window=window, stride=stride, skip_na=False, mode="same")
    naive_full = _naive_moving_average_same(series, window, skip_na=False)
    assert rust.shape == naive_full.shape
    # Windows touching NaN become NaN
    assert np.all((np.isnan(rust) == np.isnan(naive_full)))
    mask = ~np.isnan(naive_full)
    assert np.allclose(rust[mask], naive_full[mask], atol=1e-12)


def test_moving_average_temporal_stride_3d_shape():
    # (T=12, Y=4, X=5)
    cube = np.random.rand(12, 4, 5)
    out_same = moving_average_temporal_stride(cube, window=3, stride=2, mode="same")
    out_valid = moving_average_temporal_stride(cube, window=3, stride=2, mode="valid")
    assert out_same.shape[1:] == cube.shape[1:]
    assert out_valid.shape[1:] == cube.shape[1:]
    # Expect lengths: same -> ceil(12/2)=6; valid base length=12-3+1=10 -> ceil(10/2)=5
    assert out_same.shape[0] == 6
    assert out_valid.shape[0] == 5


def test_moving_average_temporal_stride_window_stride_errors():
    series = np.random.rand(6)
    with pytest.raises(ValueError):
        moving_average_temporal_stride(series, window=0, stride=2)
    with pytest.raises(ValueError):
        moving_average_temporal_stride(series, window=3, stride=0)
    with pytest.raises(ValueError):
        moving_average_temporal_stride(series, window=10, stride=2, mode="valid")
    with pytest.raises(ValueError):
        moving_average_temporal_stride(series, window=3, stride=2, mode="unsupported")


def test_moving_average_temporal_stride_equivalence_with_stride_1():
    cube = np.random.rand(10, 3, 3)
    win = 3
    out_stride1 = moving_average_temporal_stride(cube, window=win, stride=1, mode="same")
    out_full = moving_average_temporal(cube, window=win, mode="same")
    assert np.allclose(out_stride1, out_full, atol=1e-12)


def test_moving_average_temporal_stride_large_random_consistency():
    # Larger random 1D series for robustness
    rng = np.random.default_rng(42)
    series = rng.random(257)
    window = 7
    stride = 5
    rust = moving_average_temporal_stride(series, window=window, stride=stride, mode="same")
    naive_full = _naive_moving_average_same(series, window, skip_na=True)
    expected = naive_full[::stride]
    assert rust.shape == expected.shape
    assert np.allclose(rust, expected, atol=1e-12)
