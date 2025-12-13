import numpy as np
from eo_processor import detect_breakpoints, complex_classification, texture_entropy

def test_detect_breakpoints():
    """
    Test the detect_breakpoints function with a synthetic time series.
    """
    # Create a time series with a clear breakpoint
    time = 100
    breakpoint_time = 50

    np.random.seed(42)
    # Reduce noise to make the breakpoint more obvious and the test more stable
    noise = np.random.normal(0, 0.1, time)
    y = np.concatenate([
        np.linspace(0, 10, breakpoint_time),
        np.linspace(10, 0, time - breakpoint_time)
    ]) + noise

    # Create a 3D stack (time, y, x)
    stack = np.zeros((time, 1, 1))
    stack[:, 0, 0] = y

    # Create corresponding dates
    dates = np.arange(time).astype(np.int64)

    # Run the breakpoint detection
    result = detect_breakpoints(stack, dates.tolist(), threshold=0.1)

    # Extract the results
    break_date = result[0, 0, 0]
    magnitude = result[1, 0, 0]
    confidence = result[2, 0, 0]

    # Assert that a breakpoint was detected near the expected time
    assert np.isclose(break_date, breakpoint_time, atol=5)
    assert magnitude > 0
    assert confidence == 1.0

def test_complex_classification():
    """
    Test the complex_classification function.
    """
    shape = (10, 10)
    blue = np.random.rand(*shape)
    green = np.random.rand(*shape)
    red = np.random.rand(*shape)
    nir = np.random.rand(*shape)
    swir1 = np.random.rand(*shape)
    swir2 = np.random.rand(*shape)
    temp = np.random.rand(*shape) * 300

    result = complex_classification(blue, green, red, nir, swir1, swir2, temp)
    assert result.shape == shape
    assert result.dtype == np.uint8

def test_texture_entropy():
    """
    Test the texture_entropy function.
    """
    shape = (20, 20)
    data = np.random.rand(*shape)

    result = texture_entropy(data, window_size=3)
    assert result.shape == shape
    assert result.dtype == np.float64
