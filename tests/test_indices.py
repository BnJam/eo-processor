import numpy as np
import pytest

from eo_processor import (
    ndvi,
    ndwi,
    normalized_difference,
    enhanced_vegetation_index as evi,
    savi,
    nbr,
    ndmi,
    nbr2,
    gci,
)


def test_normalized_difference_1d_basic():
    a = np.array([0.8, 0.7, 0.6], dtype=np.float64)
    b = np.array([0.2, 0.1, 0.3], dtype=np.float64)
    result = normalized_difference(a, b)
    expected = (a - b) / (a + b)
    assert result.shape == a.shape
    assert result.dtype == np.float64
    assert np.allclose(result, expected, rtol=1e-9, atol=1e-9)


def test_normalized_difference_antisymmetry():
    a = np.array([0.9, 0.3, 0.5], dtype=np.float64)
    b = np.array([0.1, 0.2, 0.4], dtype=np.float64)
    lhs = normalized_difference(a, b)
    rhs = normalized_difference(b, a)
    assert np.allclose(lhs, -rhs, rtol=1e-12, atol=0.0)


def test_normalized_difference_1d_zero_division():
    a = np.array([0.0, 0.5, 0.0], dtype=np.float64)
    b = np.array([0.0, -0.5, 0.0], dtype=np.float64)
    result = normalized_difference(a, b)
    assert np.isnan(result[0])
    assert np.isinf(result[1])
    assert np.isnan(result[2])


def test_normalized_difference_2d():
    a = np.array([[0.8, 0.7], [0.6, 0.5]], dtype=np.float64)
    b = np.array([[0.2, 0.1], [0.3, 0.5]], dtype=np.float64)
    result = normalized_difference(a, b)
    expected = (a - b) / (a + b)
    assert result.shape == a.shape
    assert np.allclose(result, expected, rtol=1e-9, atol=1e-9)


def test_ndvi_equivalence_with_normalized_difference():
    nir = np.array([0.8, 0.7, 0.6], dtype=np.float64)
    red = np.array([0.2, 0.1, 0.3], dtype=np.float64)
    ndvi_vals = ndvi(nir, red)
    nd_generic = normalized_difference(nir, red)
    assert np.allclose(ndvi_vals, nd_generic, rtol=1e-12, atol=0.0)


def test_ndvi_1d():
    nir = np.array([0.8, 0.7, 0.6])
    red = np.array([0.2, 0.1, 0.3])
    result = ndvi(nir, red)
    expected = (nir - red) / (nir + red)
    assert result.shape == nir.shape
    assert np.allclose(result, expected, rtol=1e-9, atol=1e-9)


def test_ndvi_2d():
    nir = np.array([[0.8, 0.7], [0.6, 0.5]])
    red = np.array([[0.2, 0.1], [0.3, 0.5]])
    result = ndvi(nir, red)
    expected = (nir - red) / (nir + red)
    assert result.shape == nir.shape
    assert np.allclose(result, expected, rtol=1e-9, atol=1e-9)


def test_ndwi_1d():
    green = np.array([0.4, 0.5, 0.6])
    nir = np.array([0.2, 0.1, 0.3])
    result = ndwi(green, nir)
    expected = (green - nir) / (green + nir)
    assert result.shape == green.shape
    assert np.allclose(result, expected, rtol=1e-9, atol=1e-9)


def test_ndwi_2d():
    green = np.array([[0.4, 0.5], [0.6, 0.7]])
    nir = np.array([[0.2, 0.1], [0.3, 0.4]])
    result = ndwi(green, nir)
    expected = (green - nir) / (green + nir)
    assert result.shape == green.shape
    assert np.allclose(result, expected, rtol=1e-9, atol=1e-9)


def test_evi_1d():
    G, C1, C2, L = 2.5, 6.0, 7.5, 1.0
    nir = np.array([0.6, 0.7])
    red = np.array([0.3, 0.2])
    blue = np.array([0.1, 0.05])
    result = evi(nir, red, blue)
    denom = nir + C1 * red - C2 * blue + L
    expected = G * (nir - red) / denom
    mask = np.isclose(denom, 0.0, atol=1e-10)
    assert np.allclose(result[~mask], expected[~mask], rtol=1e-12, atol=0.0)
    assert np.all(result[mask] == 0.0)


def test_evi_2d():
    G, C1, C2, L = 2.5, 6.0, 7.5, 1.0
    nir = np.array([[0.6, 0.7], [0.2, 0.3]])
    red = np.array([[0.3, 0.2], [0.1, 0.15]])
    blue = np.array([[0.1, 0.05], [0.02, 0.03]])
    result = evi(nir, red, blue)
    denom = nir + C1 * red - C2 * blue + L
    expected = G * (nir - red) / denom
    mask = np.isclose(denom, 0.0, atol=1e-10)
    assert np.allclose(result[~mask], expected[~mask], rtol=1e-12, atol=0.0)
    assert np.all(result[mask] == 0.0)


def test_evi_monotonic_in_nir():
    # With red & blue fixed, EVI should increase (non-decreasing) with NIR.
    nir = np.linspace(0.1, 0.9, 50)
    red = np.full_like(nir, 0.2)
    blue = np.full_like(nir, 0.05)
    out = evi(nir, red, blue)
    assert np.all(np.diff(out) >= -1e-9)


def test_shape_mismatch_ndvi():
    nir = np.array([0.8, 0.7, 0.6])
    red = np.array([0.2, 0.1])  # different length
    with pytest.raises(ValueError):
        ndvi(nir, red)


def test_shape_mismatch_evi():
    nir = np.array([0.6, 0.7])
    red = np.array([0.3, 0.2])
    blue = np.array([0.1])  # mismatched
    with pytest.raises(ValueError):
        evi(nir, red, blue)


def test_invalid_dimension():
    nir = np.random.rand(2, 2, 2, 2, 2)
    red = np.random.rand(2, 2, 2, 2, 2)
    with pytest.raises(TypeError):
        ndvi(nir, red)


def test_dtype_coercion():
    nir = np.array([1, 2, 3], dtype=np.int32)
    red = np.array([1, 0, 2], dtype=np.int32)
    out = ndvi(nir.astype(float), red.astype(float))
    assert out.dtype == np.float64
    expected = (nir - red) / (nir + red)
    assert np.allclose(out, expected, rtol=1e-12)


def test_ndwi_range():
    green = np.array([0.6, 0.2])
    nir = np.array([0.1, 0.25])
    out = ndwi(green, nir)
    assert np.all(out <= 1.0 + 1e-12)
    assert np.all(out >= -1.0 - 1e-12)


def test_ndvi_range():
    nir = np.array([0.9, 0.05])
    red = np.array([0.1, 0.02])
    out = ndvi(nir, red)
    assert np.all(out <= 1.0 + 1e-12)
    assert np.all(out >= -1.0 - 1e-12)


def test_ndvi_random_bounds_property():
    rng = np.random.default_rng(42)
    nir = rng.uniform(0.0, 1.0, 1000)
    red = rng.uniform(0.0, 1.0, 1000)
    out = ndvi(nir, red)
    assert np.all(out <= 1.0 + 1e-12)
    assert np.all(out >= -1.0 - 1e-12)


def test_evi_zero_denominator_safeguard():
    C2, L = 7.5, 1.0
    nir = np.array([0.0])
    red = np.array([0.0])
    blue = np.array([L / C2])  # makes denominator ~ 0
    out = evi(nir, red, blue)
    assert np.isclose(out[0], 0.0, atol=1e-12)


def test_savi_1d():
    nir = np.array([0.7, 0.6, 0.5], dtype=np.float64)
    red = np.array([0.2, 0.3, 0.1], dtype=np.float64)
    out = savi(nir, red)
    assert out.shape == nir.shape
    L = 0.5
    expected = (nir - red) / (nir + red + L) * (1.0 + L)
    # Zero denom safeguard (unlikely here, but keep logic consistent)
    mask = np.isclose(nir + red + L, 0.0, atol=1e-10)
    expected[mask] = 0.0
    assert np.allclose(out, expected, rtol=1e-12, atol=0.0)


def test_savi_2d():
    nir = np.array([[0.6, 0.7], [0.5, 0.4]], dtype=np.float64)
    red = np.array([[0.2, 0.3], [0.1, 0.2]], dtype=np.float64)
    out = savi(nir, red)
    assert out.shape == nir.shape
    L = 0.5
    expected = (nir - red) / (nir + red + L) * (1.0 + L)
    mask = np.isclose(nir + red + L, 0.0, atol=1e-10)
    expected[mask] = 0.0
    assert np.allclose(out, expected, rtol=1e-12, atol=0.0)


def test_savi_shape_mismatch():
    nir = np.array([0.7, 0.6], dtype=np.float64)
    red = np.array([0.2], dtype=np.float64)
    with pytest.raises(ValueError):
        savi(nir, red)


def test_nbr_1d():
    nir = np.array([0.8, 0.6, 0.5], dtype=np.float64)
    swir2 = np.array([0.3, 0.2, 0.1], dtype=np.float64)
    out = nbr(nir, swir2)
    assert out.shape == nir.shape
    denom = nir + swir2
    expected = (nir - swir2) / denom
    mask = np.isclose(denom, 0.0, atol=1e-10)
    expected[mask] = 0.0
    assert np.allclose(out, expected, rtol=1e-12)


def test_nbr_2d():
    nir = np.array([[0.8, 0.7], [0.6, 0.5]], dtype=np.float64)
    swir2 = np.array([[0.3, 0.25], [0.2, 0.15]], dtype=np.float64)
    out = nbr(nir, swir2)
    assert out.shape == nir.shape
    denom = nir + swir2
    expected = (nir - swir2) / denom
    mask = np.isclose(denom, 0.0, atol=1e-10)
    expected[mask] = 0.0
    assert np.allclose(out, expected, rtol=1e-12)


def test_nbr_shape_mismatch():
    nir = np.array([0.8, 0.7], dtype=np.float64)
    swir2 = np.array([0.3], dtype=np.float64)
    with pytest.raises(ValueError):
        nbr(nir, swir2)


def test_savi_variable_L():
    nir = np.array([0.7, 0.6], dtype=np.float64)
    red = np.array([0.2, 0.3], dtype=np.float64)
    for L in [0.0, 0.25, 0.5, 1.0]:
        out = savi(nir, red, l=L)
        expected = (nir - red) / (nir + red + L) * (1.0 + L)
        mask = np.isclose(nir + red + L, 0.0, atol=1e-10)
        expected[mask] = 0.0
        assert np.allclose(out, expected, rtol=1e-12, atol=0.0)


def test_savi_l_precedence_over_L():
    """
    When both L and l are passed, the wrapper should prioritize 'l'.
    """
    nir = np.array([0.7, 0.6], dtype=np.float64)
    red = np.array([0.2, 0.3], dtype=np.float64)
    out_explicit_l = savi(nir, red, L=0.0, l=0.25)  # L ignored, l used
    expected = (nir - red) / (nir + red + 0.25) * (1.0 + 0.25)
    mask = np.isclose(nir + red + 0.25, 0.0, atol=1e-10)
    expected[mask] = 0.0
    assert np.allclose(out_explicit_l, expected, rtol=1e-12, atol=0.0)


def test_ndmi_1d():
    nir = np.array([0.8, 0.6, 0.4], dtype=np.float64)
    swir1 = np.array([0.3, 0.2, 0.1], dtype=np.float64)
    out = ndmi(nir, swir1)
    denom = nir + swir1
    expected = (nir - swir1) / denom
    mask = np.isclose(denom, 0.0, atol=1e-10)
    expected[mask] = 0.0
    assert np.allclose(out, expected, rtol=1e-12)


def test_ndmi_2d():
    nir = np.array([[0.8, 0.7], [0.6, 0.5]], dtype=np.float64)
    swir1 = np.array([[0.3, 0.25], [0.2, 0.15]], dtype=np.float64)
    out = ndmi(nir, swir1)
    denom = nir + swir1
    expected = (nir - swir1) / denom
    mask = np.isclose(denom, 0.0, atol=1e-10)
    expected[mask] = 0.0
    assert np.allclose(out, expected, rtol=1e-12)


def test_nbr2_1d():
    swir1 = np.array([0.4, 0.5, 0.45], dtype=np.float64)
    swir2 = np.array([0.3, 0.25, 0.2], dtype=np.float64)
    out = nbr2(swir1, swir2)
    denom = swir1 + swir2
    expected = (swir1 - swir2) / denom
    mask = np.isclose(denom, 0.0, atol=1e-10)
    expected[mask] = 0.0
    assert np.allclose(out, expected, rtol=1e-12)


def test_nbr2_2d():
    swir1 = np.array([[0.4, 0.5], [0.45, 0.55]], dtype=np.float64)
    swir2 = np.array([[0.3, 0.25], [0.2, 0.15]], dtype=np.float64)
    out = nbr2(swir1, swir2)
    denom = swir1 + swir2
    expected = (swir1 - swir2) / denom
    mask = np.isclose(denom, 0.0, atol=1e-10)
    expected[mask] = 0.0
    assert np.allclose(out, expected, rtol=1e-12)


def test_gci_1d():
    nir = np.array([0.8, 0.9, 0.7], dtype=np.float64)
    green = np.array([0.4, 0.45, 0.35], dtype=np.float64)
    out = gci(nir, green)
    expected = (nir / green) - 1.0
    # Guard near-zero
    mask = np.isclose(green, 0.0, atol=1e-12)
    expected[mask] = 0.0
    assert np.allclose(out, expected, rtol=1e-12)


def test_gci_2d():
    nir = np.array([[0.8, 0.9], [0.7, 0.75]], dtype=np.float64)
    green = np.array([[0.4, 0.45], [0.35, 0.42]], dtype=np.float64)
    out = gci(nir, green)
    expected = (nir / green) - 1.0
    mask = np.isclose(green, 0.0, atol=1e-12)
    expected[mask] = 0.0
    assert np.allclose(out, expected, rtol=1e-12)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
