import numpy as np
import pytest

from eo_processor import linear_regression, trend_analysis


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


def test_trend_analysis_no_break_single_segment():
    y = np.linspace(0.0, 10.0, 50)
    segments = trend_analysis(y.tolist(), threshold=1e9)
    assert len(segments) == 1
    assert segments[0].start_index == 0
    assert segments[0].end_index == 49
    assert segments[0].slope == pytest.approx(10.0 / 49.0, rel=1e-6)


def test_trend_analysis_detects_break():
    y = np.concatenate([np.linspace(0, 10, 50), np.linspace(10, 0, 50)])
    segments = trend_analysis(y.tolist(), threshold=1.0)
    assert len(segments) >= 2
    for segment in segments:
        assert segment.end_index >= segment.start_index


def test_trend_analysis_rejects_negative_threshold():
    y = np.linspace(0.0, 10.0, 20)
    with pytest.raises(ValueError, match="non-negative"):
        trend_analysis(y.tolist(), threshold=-1.0)


def test_trend_analysis_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        trend_analysis([1.0, np.nan, 2.0], threshold=0.5)


def test_star_import_exposes_trend_analysis():
    namespace = {}
    exec("from eo_processor import *", namespace)
    assert "trend_analysis" in namespace
