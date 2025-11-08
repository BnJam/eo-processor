import numpy as np
import pytest

# If the Rust extension isn't built, skip these tests rather than erroring out.
eo = pytest.importorskip("eo_processor")
normalized_difference = eo.normalized_difference
enhanced_vegetation_index = eo.enhanced_vegetation_index

@pytest.fixture
def test_data_1d():
    """Fixture for standard 1D array test data."""
    return (
        np.array([0.8, 0.7, 0.6], dtype=np.float64),  # Band A (e.g., NIR)
        np.array([0.2, 0.1, 0.3], dtype=np.float64),  # Band B (e.g., Red)
    )

@pytest.fixture
def test_data_2d():
    """Fixture for standard 2D array test data."""
    a = np.array([[0.8, 0.7], [0.6, 0.5]], dtype=np.float64)
    b = np.array([[0.2, 0.1], [0.3, 0.5]], dtype=np.float64)
    return a, b


def test_normalized_difference_1d_basic(test_data_1d):
    """Tests the basic, expected calculation for ND."""
    a, b = test_data_1d
    result = normalized_difference(a, b)

    # Expected results:
    # (0.8 - 0.2) / (0.8 + 0.2) = 0.6 / 1.0 = 0.6
    # (0.7 - 0.1) / (0.7 + 0.1) = 0.6 / 0.8 = 0.75
    # (0.6 - 0.3) / (0.6 + 0.3) = 0.3 / 0.9 = 0.333333...
    expected = np.array([0.6, 0.75, 1/3], dtype=np.float64)

    np.testing.assert_allclose(result, expected, atol=1e-10)
    assert result.shape == a.shape


def test_normalized_difference_1d_zero_denominator():
    """Tests ND with inputs that sum to zero, expecting 0.0 as per Rust logic."""
    a = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    b = np.array([-0.5, 0.0, 0.0], dtype=np.float64)
    result = normalized_difference(a, b)

    # Expected:
    # Index 0: sum is 0.0, should be 0.0
    # Index 1: sum is 0.0, should be 0.0
    # Index 2: sum is 0.0, should be 0.0
    expected = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_normalized_difference_2d_basic(test_data_2d):
    """Tests the basic calculation for 2D ND arrays."""
    a, b = test_data_2d
    result = normalized_difference(a, b)

    # Expected results match 1D test for the first three elements
    # Index [1, 1]: (0.5 - 0.5) / (0.5 + 0.5) = 0.0 / 1.0 = 0.0
    expected = np.array([[0.6, 0.75], [1/3, 0.0]], dtype=np.float64)

    np.testing.assert_allclose(result, expected, atol=1e-10)
    assert result.shape == a.shape


def test_enhanced_vegetation_index_1d_basic():
    """Tests the EVI formula calculation with known constants."""
    # EVI = G * (NIR - RED) / (NIR + C1 * RED - C2 * BLUE + L)
    # G=2.5, L=1.0, C1=6.0, C2=7.5

    nir = np.array([0.4], dtype=np.float64)
    red = np.array([0.1], dtype=np.float64)
    blue = np.array([0.05], dtype=np.float64)

    # Calculation:
    # Numerator = 2.5 * (0.4 - 0.1) = 0.75
    # Denominator = 0.4 + 6.0*0.1 - 7.5*0.05 + 1.0 = 0.4 + 0.6 - 0.375 + 1.0 = 1.625
    # Expected = 0.75 / 1.625 ≈ 0.46153846

    expected = 0.75 / 1.625
    result = enhanced_vegetation_index(nir, red, blue)

    np.testing.assert_allclose(result[0], expected, atol=1e-10)
    assert result.shape == nir.shape
