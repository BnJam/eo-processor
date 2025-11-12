import numpy as np
from eo_processor import temporal_mean, temporal_std


def test_temporal_mean_1d():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = temporal_mean(arr)
    assert result == 3.0


def test_temporal_mean_1d_with_nan_skip():
    arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    result = temporal_mean(arr, skip_na=True)
    assert result == 3.0


def test_temporal_mean_2d():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = temporal_mean(arr)
    np.testing.assert_array_equal(result, np.array([2.0, 3.0]))


def test_temporal_mean_3d():
    arr = np.arange(8, dtype=np.float64).reshape((2, 2, 2))
    result = temporal_mean(arr)
    expected = np.array([[2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_array_equal(result, expected)


def test_temporal_std_1d():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = temporal_std(arr)
    assert np.isclose(result, 1.58113883)


def test_temporal_std_1d_with_nan_skip():
    arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    result = temporal_std(arr, skip_na=True)
    assert np.isclose(result, 1.825741858)


def test_temporal_mean_4d():
    arr = np.arange(16, dtype=np.float64).reshape((2, 2, 2, 2))
    result = temporal_mean(arr)
    expected = np.array([[[4.0, 5.0], [6.0, 7.0]], [[8.0, 9.0], [10.0, 11.0]]])
    np.testing.assert_array_equal(result, expected)


def test_temporal_std_4d():
    arr = np.arange(16, dtype=np.float64).reshape((2, 2, 2, 2))
    result = temporal_std(arr)
    expected = np.std(arr, axis=0, ddof=1)
    np.testing.assert_allclose(result, expected)
