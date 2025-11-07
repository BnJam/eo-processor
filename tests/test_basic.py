"""Tests for basic eo-processor functionality."""

import numpy as np
import pytest


def test_imports():
    """Test that all functions can be imported."""
    from eo_processor import (
        normalized_difference,
        normalized_difference_1d,
        normalized_difference_2d,
        ndvi,
        ndvi_1d,
        ndvi_2d,
        ndwi,
        ndwi_1d,
        ndwi_2d,
    )
    assert normalized_difference is not None
    assert ndvi is not None
    assert ndwi is not None


def test_normalized_difference_1d():
    """Test normalized difference with 1D arrays."""
    from eo_processor import normalized_difference_1d
    
    a = np.array([0.8, 0.7, 0.6])
    b = np.array([0.2, 0.1, 0.3])
    
    result = normalized_difference_1d(a, b)
    
    expected = np.array([0.6, 0.75, 1.0 / 3.0])
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_normalized_difference_2d():
    """Test normalized difference with 2D arrays."""
    from eo_processor import normalized_difference_2d
    
    a = np.array([[0.8, 0.7], [0.6, 0.5]])
    b = np.array([[0.2, 0.1], [0.3, 0.5]])
    
    result = normalized_difference_2d(a, b)
    
    expected = np.array([[0.6, 0.75], [1.0 / 3.0, 0.0]])
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_normalized_difference_auto():
    """Test auto-detection of array dimensions."""
    from eo_processor import normalized_difference
    
    # Test 1D
    a_1d = np.array([0.8, 0.7])
    b_1d = np.array([0.2, 0.1])
    result_1d = normalized_difference(a_1d, b_1d)
    assert result_1d.shape == (2,)
    
    # Test 2D
    a_2d = np.array([[0.8, 0.7], [0.6, 0.5]])
    b_2d = np.array([[0.2, 0.1], [0.3, 0.5]])
    result_2d = normalized_difference(a_2d, b_2d)
    assert result_2d.shape == (2, 2)


def test_normalized_difference_invalid_dimension():
    """Test that 3D arrays raise an error."""
    from eo_processor import normalized_difference
    
    a = np.random.rand(2, 2, 2)
    b = np.random.rand(2, 2, 2)
    
    with pytest.raises(ValueError, match="Unsupported array dimension"):
        normalized_difference(a, b)


def test_ndvi_basic():
    """Test NDVI computation."""
    from eo_processor import ndvi
    
    nir = np.array([0.8, 0.7, 0.6])
    red = np.array([0.2, 0.1, 0.3])
    
    result = ndvi(nir, red)
    
    # NDVI = (NIR - Red) / (NIR + Red)
    expected = (nir - red) / (nir + red)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_ndvi_2d():
    """Test NDVI with 2D arrays."""
    from eo_processor import ndvi
    
    nir = np.random.rand(10, 10)
    red = np.random.rand(10, 10)
    
    result = ndvi(nir, red)
    
    assert result.shape == (10, 10)
    assert result.dtype == np.float64
    
    # Verify the computation
    expected = (nir - red) / (nir + red)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_ndwi_basic():
    """Test NDWI computation."""
    from eo_processor import ndwi
    
    green = np.array([0.3, 0.4, 0.5])
    nir = np.array([0.2, 0.1, 0.3])
    
    result = ndwi(green, nir)
    
    # NDWI = (Green - NIR) / (Green + NIR)
    expected = (green - nir) / (green + nir)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_zero_division_handling():
    """Test that zero division is handled gracefully."""
    from eo_processor import normalized_difference
    
    a = np.array([0.0, 0.5, 0.0])
    b = np.array([0.0, 0.5, 0.0])
    
    result = normalized_difference(a, b)
    
    # When both values are the same, result should be 0
    # When both are zero, result should be 0 (handled as edge case)
    assert result[0] == 0.0
    assert result[1] == 0.0
    assert result[2] == 0.0


def test_large_array_performance():
    """Test with large arrays to verify no crashes."""
    from eo_processor import ndvi
    
    # Create large arrays
    nir = np.random.rand(1000, 1000)
    red = np.random.rand(1000, 1000)
    
    result = ndvi(nir, red)
    
    assert result.shape == (1000, 1000)
    assert not np.isnan(result).any()  # No NaN values


def test_ndvi_range():
    """Test that NDVI values are in valid range [-1, 1]."""
    from eo_processor import ndvi
    
    nir = np.random.rand(100, 100)
    red = np.random.rand(100, 100)
    
    result = ndvi(nir, red)
    
    assert result.min() >= -1.0
    assert result.max() <= 1.0


def test_output_dtype():
    """Test that output is always float64."""
    from eo_processor import ndvi
    
    nir = np.array([0.8, 0.7], dtype=np.float64)
    red = np.array([0.2, 0.1], dtype=np.float64)
    
    result = ndvi(nir, red)
    
    assert result.dtype == np.float64
