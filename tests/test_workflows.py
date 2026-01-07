import numpy as np
import pandas as pd
from eo_processor import bfast_monitor, complex_classification


def test_bfast_monitor_logic():
    """
    Test the bfast_monitor function with synthetic time series
    for both break and no-break scenarios.
    """
    # --- 1. Generate common data ---
    # Create a date range
    history_dates = pd.to_datetime(pd.date_range(start="2010-01-01", end="2014-12-31", freq="16D"))
    monitor_dates = pd.to_datetime(pd.date_range(start="2015-01-01", end="2017-12-31", freq="16D"))
    all_dates = history_dates.union(monitor_dates)

    # Convert dates to fractional years for generating the signal
    time_frac = all_dates.year + all_dates.dayofyear / 365.25

    # Convert dates to integer format YYYYMMDD for the function input
    dates_int = (all_dates.year * 10000 + all_dates.month * 100 + all_dates.day).to_numpy(dtype=np.int64)

    history_start_date = 20100101
    monitor_start_date = 20150101

    # Generate a base harmonic signal
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, len(all_dates))
    signal = 0.5 + 0.2 * np.cos(2 * np.pi * time_frac) + 0.1 * np.sin(4 * np.pi * time_frac) + noise

    # --- 2. Test break detection scenario ---

    # Introduce a sudden drop in the monitoring period
    break_signal = signal.values.copy()
    monitor_start_index = len(history_dates)
    break_signal[monitor_start_index:] -= 0.4

    # Create a 3D stack (Time, Y, X)
    stack_break = np.zeros((len(all_dates), 1, 1))
    stack_break[:, 0, 0] = break_signal

    # Run bfast_monitor for the break scenario
    result_break = bfast_monitor(
        stack_break,
        dates_int.tolist(),
        history_start_date=history_start_date,
        monitor_start_date=monitor_start_date,
        order=1,
        h=0.25,
        alpha=0.05,
    )

    break_date_frac = result_break[0, 0, 0]
    magnitude = result_break[1, 0, 0]

    # Assert that a breakpoint was detected near the start of the monitoring period
    # The exact date depends on the MOSUM window, so we check a range
    assert 2015.0 < break_date_frac < 2016.5
    assert magnitude > 0.3  # Should be around 0.4

    # --- 3. Test no-break scenario ---

    # Use the original stable signal
    stack_stable = np.zeros((len(all_dates), 1, 1))
    stack_stable[:, 0, 0] = signal

    # Run bfast_monitor for the stable scenario
    result_stable = bfast_monitor(
        stack_stable,
        dates_int.tolist(),
        history_start_date=history_start_date,
        monitor_start_date=monitor_start_date,
        order=1,
        h=0.25,
        alpha=0.05,
    )

    # Assert that no breakpoint was detected
    assert result_stable[0, 0, 0] == 0.0
    assert result_stable[1, 0, 0] == 0.0


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
