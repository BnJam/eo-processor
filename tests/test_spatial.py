import numpy as np
from eo_processor import median_composite

def test_median_composite_3d():
    arr = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
        [[3.0, 4.0], [5.0, 6.0]]
    ])
    result = median_composite(arr)
    expected = np.array([[3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_array_equal(result, expected)

def test_median_composite_4d():
    arr = np.array([
        [[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]],
        [[[5., 6.], [7., 8.]], [[5., 6.], [7., 8.]]],
        [[[3., 4.], [5., 6.]], [[3., 4.], [5., 6.]]]
    ])
    result = median_composite(arr)
    expected = np.array([[[3., 4.], [5., 6.]], [[3., 4.], [5., 6.]]])
    np.testing.assert_array_equal(result, expected)

def test_median_composite_with_nan_skip_na_true():
    arr = np.array([
        [[1.0, 2.0], [np.nan, 4.0]],
        [[5.0, np.nan], [7.0, 8.0]],
        [[3.0, 4.0], [5.0, 6.0]]
    ])
    result = median_composite(arr, skip_na=True)
    expected = np.array([[3.0, 3.0], [6.0, 6.0]])
    np.testing.assert_array_equal(result, expected)

def test_median_composite_with_nan_skip_na_false():
    arr = np.array([
        [[1.0, 2.0], [np.nan, 4.0]],
        [[5.0, np.nan], [7.0, 8.0]],
        [[3.0, 4.0], [5.0, 6.0]]
    ])
    result = median_composite(arr, skip_na=False)
    expected = np.array([[3.0, np.nan], [np.nan, 6.0]])
    np.testing.assert_array_equal(result, expected)
