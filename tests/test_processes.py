import numpy as np
import pytest
from eo_processor import moving_average_temporal, pixelwise_transform


# -----------------------------
# Helpers
# -----------------------------
def py_moving_average_same(series: np.ndarray, window: int, skip_na: bool) -> np.ndarray:
    """
    Pure-Python/Numpy reference for 'same' mode moving average (variable window near edges).
    skip_na semantics:
      - skip_na=True: exclude NaNs from mean; empty window → NaN.
      - skip_na=False: any NaN in window → NaN; otherwise arithmetic mean.
    """
    t = series.shape[0]
    out = np.empty(t, dtype=float)
    half_left = window // 2
    half_right = window - half_left - 1
    for i in range(t):
        start = max(0, i - half_left)
        end = min(t - 1, i + half_right)
        window_vals = series[start : end + 1]
        if skip_na:
            valid = window_vals[~np.isnan(window_vals)]
            out[i] = np.nan if valid.size == 0 else valid.mean()
        else:
            out[i] = np.nan if np.isnan(window_vals).any() else window_vals.mean()
    return out


def py_moving_average_valid(series: np.ndarray, window: int, skip_na: bool) -> np.ndarray:
    """
    'valid' mode: only full windows (no edge shrink). Length = t - window + 1.
    """
    t = series.shape[0]
    out_len = t - window + 1
    out = np.empty(out_len, dtype=float)
    for i in range(out_len):
        window_vals = series[i : i + window]
        if skip_na:
            valid = window_vals[~np.isnan(window_vals)]
            out[i] = np.nan if valid.size == 0 else valid.mean()
        else:
            out[i] = np.nan if np.isnan(window_vals).any() else window_vals.mean()
    return out


# -----------------------------
# moving_average_temporal Tests
# -----------------------------
def test_moving_average_temporal_1d_same_skip_na():
    series = np.array([1.0, 2.0, 3.0, 4.0])
    rust = moving_average_temporal(series, window=3, skip_na=True, mode="same")
    ref = py_moving_average_same(series, window=3, skip_na=True)
    assert rust.shape == series.shape
    assert np.allclose(rust, ref, atol=1e-12)


def test_moving_average_temporal_1d_same_no_skip_na():
    series = np.array([1.0, np.nan, 3.0, 4.0])
    rust = moving_average_temporal(series, window=3, skip_na=False, mode="same")
    ref = py_moving_average_same(series, window=3, skip_na=False)
    # Positions touching the NaN should be NaN in no-skip mode
    assert np.isnan(rust[0]) == np.isnan(ref[0])
    assert np.isnan(rust[1]) == np.isnan(ref[1])
    assert np.isnan(rust[2]) == np.isnan(ref[2])
    assert np.isnan(rust[3]) == np.isnan(ref[3])
    # Where not NaN, values should match
    mask = ~np.isnan(ref)
    assert np.allclose(rust[mask], ref[mask], atol=1e-12)


def test_moving_average_temporal_1d_valid():
    series = np.array([1.0, 2.0, 3.0, 4.0])
    rust = moving_average_temporal(series, window=2, skip_na=True, mode="valid")
    ref = py_moving_average_valid(series, window=2, skip_na=True)
    assert rust.shape == ref.shape
    assert np.allclose(rust, ref, atol=1e-12)


def test_moving_average_temporal_2d_basic():
    arr = np.stack(
        [
            np.array([1.0, 10.0, 100.0]),  # t0
            np.array([2.0, 20.0, 200.0]),  # t1
            np.array([3.0, 30.0, 300.0]),  # t2
        ],
        axis=0,
    )  # shape (3,3)
    rust = moving_average_temporal(arr, window=3, skip_na=True, mode="same")
    # Reference per column
    ref_cols = []
    for c in range(arr.shape[1]):
        ref_cols.append(py_moving_average_same(arr[:, c], window=3, skip_na=True))
    ref = np.vstack(ref_cols).T  # shape (3,3)
    assert rust.shape == arr.shape
    assert np.allclose(rust, ref, atol=1e-12)


def test_moving_average_temporal_3d_nan_handling():
    # shape (T=4, Y=2, X=2)
    cube = np.array(
        [
            [[1.0, 2.0], [3.0, np.nan]],
            [[2.0, 4.0], [6.0, 8.0]],
            [[3.0, 6.0], [9.0, 12.0]],
            [[4.0, 8.0], [12.0, 16.0]],
        ]
    )
    rust_skip = moving_average_temporal(cube, window=3, skip_na=True, mode="same")
    rust_noskip = moving_average_temporal(cube, window=3, skip_na=False, mode="same")
    # Check shape stays same
    assert rust_skip.shape == cube.shape
    assert rust_noskip.shape == cube.shape
    # Pick pixel (y=1,x=1) which has NaN at time index 0
    pixel_series = cube[:, 1, 1]
    ref_skip = py_moving_average_same(pixel_series, window=3, skip_na=True)
    ref_noskip = py_moving_average_same(pixel_series, window=3, skip_na=False)
    assert np.allclose(rust_skip[:, 1, 1], ref_skip, equal_nan=True, atol=1e-12)
    # For no-skip, any window touching the NaN should be NaN
    assert np.isnan(rust_noskip[0, 1, 1]) == np.isnan(ref_noskip[0])
    assert np.isnan(rust_noskip[1, 1, 1]) == np.isnan(ref_noskip[1])
    # Later indices free of NaN should match numeric value
    mask = ~np.isnan(ref_noskip)
    assert np.allclose(rust_noskip[:, 1, 1][mask], ref_noskip[mask], atol=1e-12)


def test_moving_average_temporal_4d_shape():
    # (T=5, B=2, Y=2, X=2)
    cube4 = np.random.rand(5, 2, 2, 2)
    out_same = moving_average_temporal(cube4, window=3, skip_na=True, mode="same")
    out_valid = moving_average_temporal(cube4, window=3, skip_na=True, mode="valid")
    assert out_same.shape == cube4.shape
    assert out_valid.shape == (5 - 3 + 1, 2, 2, 2)


def test_moving_average_temporal_window_errors():
    arr = np.random.rand(4)
    with pytest.raises(ValueError):
        moving_average_temporal(arr, window=0)
    with pytest.raises(ValueError):
        moving_average_temporal(arr, window=5, mode="valid")
    with pytest.raises(ValueError):
        moving_average_temporal(arr, window=2, mode="unsupported")


# -----------------------------
# pixelwise_transform Tests
# -----------------------------
def test_pixelwise_transform_basic():
    arr = np.array([0.0, 0.5, 1.0])
    out = pixelwise_transform(arr, scale=2.0, offset=-0.5)
    expected = arr * 2.0 - 0.5
    assert np.allclose(out, expected, atol=1e-12)


def test_pixelwise_transform_clamp():
    arr = np.array([-1.0, 0.0, 0.5, 2.0])
    out = pixelwise_transform(arr, scale=1.5, offset=0.0, clamp_min=0.0, clamp_max=1.0)
    raw = arr * 1.5
    expected = np.clip(raw, 0.0, 1.0)
    assert np.allclose(out, expected, atol=1e-12)


def test_pixelwise_transform_nan_propagation():
    arr = np.array([np.nan, 1.0, 2.0])
    out = pixelwise_transform(arr, scale=3.0, offset=1.0)
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(1.0 * 3.0 + 1.0)
    assert out[2] == pytest.approx(2.0 * 3.0 + 1.0)


def test_pixelwise_transform_multi_dim():
    arr2d = np.array([[0.0, 1.0], [2.0, 3.0]])
    out2d = pixelwise_transform(arr2d, scale=2.0, offset=1.0)
    expected2d = arr2d * 2.0 + 1.0
    assert np.allclose(out2d, expected2d, atol=1e-12)

    arr3d = np.random.rand(3, 4, 5)
    out3d = pixelwise_transform(arr3d, scale=0.5, offset=-0.2)
    assert np.allclose(out3d, arr3d * 0.5 - 0.2, atol=1e-12)

    arr4d = np.random.rand(2, 3, 4, 5)
    out4d = pixelwise_transform(arr4d, scale=1.1, offset=0.0, clamp_min=0.0, clamp_max=1.0)
    # Since arr4d in [0,1), scaling may push some >1; clamp verifies
    expected4d = np.clip(arr4d * 1.1, 0.0, 1.0)
    assert np.allclose(out4d, expected4d, atol=1e-12)


# -----------------------------
# Combined Scenario
# -----------------------------
def test_moving_average_then_transform_chain():
    # Deep temporal stack (moderate size for test speed)
    cube = np.random.rand(12, 8, 8)
    ma = moving_average_temporal(cube, window=3, skip_na=True, mode="same")
    # Scale and clamp
    stretched = pixelwise_transform(ma, scale=1.2, offset=-0.1, clamp_min=0.0, clamp_max=1.0)
    assert stretched.shape == ma.shape
    assert np.all(stretched <= 1.0 + 1e-12)
    assert np.all(stretched >= -1e-12)  # clamp floor
