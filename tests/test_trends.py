import numpy as np
import pytest

from eo_processor import linear_regression


def test_linear_regression_basic():
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    slope, intercept, residuals = linear_regression(y)
    assert np.isclose(slope, 1.0)
    assert np.isclose(intercept, 1.0)
    assert len(residuals) == len(y)


def test_linear_regression_rejects_too_short():
    with pytest.raises(ValueError, match="at least 2 samples"):
        linear_regression(np.array([1.0], dtype=np.float64))


def test_linear_regression_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        linear_regression(np.array([1.0, np.nan, 2.0], dtype=np.float64))
