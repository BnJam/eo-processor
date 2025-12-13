import numpy as np
import pytest
from eo_processor import composite

def test_composite_median_1d():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = composite(arr, method="median")
    assert result == 3.0

def test_composite_median_1d_with_nan_skip():
    arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    result = composite(arr, method="median", skip_na=True)
    assert result == 3.0

def test_composite_mean_1d():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = composite(arr, method="mean")
    assert result == 3.0

def test_composite_mean_1d_with_nan_skip():
    arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    result = composite(arr, method="mean", skip_na=True)
    assert result == 3.0

def test_composite_std_1d():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = composite(arr, method="std")
    assert np.isclose(result, 1.58113883)

def test_composite_std_1d_with_nan_skip():
    arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    result = composite(arr, method="std", skip_na=True)
    assert np.isclose(result, 1.825741858)

def test_composite_unknown_method():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError):
        composite(arr, method="unknown")
