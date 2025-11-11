import numpy as np
from eo_processor import median, composite


def test_median_3d():
    arr = np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[3.0, 4.0], [5.0, 6.0]]]
    )
    result = median(arr)
    expected = np.array([[3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_array_equal(result, expected)


def test_median_4d():
    arr = np.array(
        [
            [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]],
            [[[5.0, 6.0], [7.0, 8.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [[[3.0, 4.0], [5.0, 6.0]], [[3.0, 4.0], [5.0, 6.0]]],
        ]
    )
    result = median(arr)
    expected = np.array([[[3.0, 4.0], [5.0, 6.0]], [[3.0, 4.0], [5.0, 6.0]]])
    np.testing.assert_array_equal(result, expected)


def test_median_with_nan_skip_na_true():
    arr = np.array(
        [
            [[1.0, 2.0], [np.nan, 4.0]],
            [[5.0, np.nan], [7.0, 8.0]],
            [[3.0, 4.0], [5.0, 6.0]],
        ]
    )
    result = median(arr, skip_na=True)
    expected = np.array([[3.0, 3.0], [6.0, 6.0]])
    np.testing.assert_array_equal(result, expected)


def test_median_with_nan_skip_na_false():
    arr = np.array(
        [
            [[1.0, 2.0], [np.nan, 4.0]],
            [[5.0, np.nan], [7.0, 8.0]],
            [[3.0, 4.0], [5.0, 6.0]],
        ]
    )
    result = median(arr, skip_na=False)
    expected = np.array([[3.0, np.nan], [np.nan, 6.0]])
    np.testing.assert_array_equal(result, expected)


def test_composite_median():
    arr = np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[3.0, 4.0], [5.0, 6.0]]]
    )
    result = composite(arr, method="median")
    expected = np.array([[3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_array_equal(result, expected)


def test_median_1d():
    arr = np.array([1.0, 5.0, 3.0, 4.0, 2.0])
    result = median(arr)
    expected = 3.0
    assert result == expected


def test_median_2d():
    arr = np.array(
        [
            [1.0, 5.0, 3.0],
            [4.0, 2.0, 6.0],
        ]
    )
    result = median(arr)
    expected = np.array([2.5, 3.5, 4.5])
    np.testing.assert_array_equal(result, expected)
