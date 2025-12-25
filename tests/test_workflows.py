import numpy as np
from eo_processor import bfast_monitor, complex_classification


def test_bfast_monitor_break_detected():
    """
    Test the bfast_monitor function with a synthetic time series
    where a breakpoint is expected.
    """
    time = 100
    history_len = 50
    monitor_len = 50

    np.random.seed(42)
    noise = np.random.normal(0, 0.1, time)
    # Stable history period, then a sudden drop in the monitoring period
    y = (
        np.concatenate(
            [np.linspace(10, 10, history_len), np.linspace(5, 5, monitor_len)]
        )
        + noise
    )

    # Create a 3D stack (time, y, x)
    stack = np.zeros((time, 1, 1))
    stack[:, 0, 0] = y

    # Create corresponding dates (as simple integers for this test)
    dates = np.arange(time, dtype=np.int64)
    history_start_date = 0
    monitor_start_date = 50

    # Run the bfast_monitor detection
    # Level is set low to ensure the break is detected
    result = bfast_monitor(
        stack, dates.tolist(), history_start_date, monitor_start_date, level=0.5
    )

    # Extract the results
    break_date = result[0, 0, 0]
    magnitude = result[1, 0, 0]

    # Assert that a breakpoint was detected at the start of the monitoring period
    assert break_date == monitor_start_date
    assert magnitude > 0


def test_bfast_monitor_no_break():
    """
    Test the bfast_monitor function with a stable time series
    where no breakpoint is expected.
    """
    time = 100
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, time)
    # Stable time series with no break
    y = np.linspace(10, 10, time) + noise

    stack = np.zeros((time, 1, 1))
    stack[:, 0, 0] = y
    dates = np.arange(time, dtype=np.int64)
    history_start_date = 0
    monitor_start_date = 50

    # Level is set high enough that noise shouldn't trigger a break
    result = bfast_monitor(
        stack, dates.tolist(), history_start_date, monitor_start_date, level=1.0
    )

    break_date = result[0, 0, 0]
    magnitude = result[1, 0, 0]

    # Assert that no breakpoint was detected
    assert break_date == 0.0
    assert magnitude == 0.0


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
