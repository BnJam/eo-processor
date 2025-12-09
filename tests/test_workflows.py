import numpy as np
from eo_processor import (
    detect_breakpoints,
    complex_classification,
    texture_entropy,
)

def test_detect_breakpoints():
    stack = np.random.rand(20, 10, 10)
    dates = list(range(20))
    threshold = 0.5
    result = detect_breakpoints(stack, dates, threshold)
    assert result.shape == (3, 10, 10)
    assert result.dtype == np.float64

def test_complex_classification():
    shape = (10, 10)
    blue = np.random.rand(*shape)
    red = np.random.rand(*shape)
    nir = np.random.rand(*shape)
    swir = np.random.rand(*shape)
    temp = np.random.rand(*shape) + 273 # Kelvin
    result = complex_classification(blue, red, nir, swir, temp)
    assert result.shape == shape
    assert result.dtype == np.uint8

def test_texture_entropy():
    input_data = np.random.rand(10, 10)
    window_size = 3
    result = texture_entropy(input_data, window_size)
    assert result.shape == input_data.shape
    assert result.dtype == np.float64
