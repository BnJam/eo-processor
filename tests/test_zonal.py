import numpy as np
import pytest
from eo_processor import zonal_stats

def test_zonal_stats_basic_1d():
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    zones = np.array([1, 1, 2, 2, 3])
    
    stats = zonal_stats(values, zones)
    
    assert len(stats) == 3
    
    # Zone 1: [10, 20]
    z1 = stats[1]
    assert z1.count == 2
    assert z1.sum == 30.0
    assert z1.mean == 15.0
    assert z1.min == 10.0
    assert z1.max == 20.0
    assert np.isclose(z1.std, 7.0710678) # std sample of [10, 20] is ~7.07
    
    # Zone 2: [30, 40]
    z2 = stats[2]
    assert z2.count == 2
    assert z2.mean == 35.0
    
    # Zone 3: [50]
    z3 = stats[3]
    assert z3.count == 1
    assert z3.mean == 50.0
    assert z3.std == 0.0

def test_zonal_stats_2d():
    values = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    zones = np.array([
        [1, 1],
        [2, 2]
    ])
    
    stats = zonal_stats(values, zones)
    
    # Zone 1: [1, 2]
    assert stats[1].sum == 3.0
    assert stats[1].mean == 1.5
    
    # Zone 2: [3, 4]
    assert stats[2].sum == 7.0
    assert stats[2].mean == 3.5

def test_zonal_stats_with_nans():
    values = np.array([10.0, np.nan, 30.0])
    zones = np.array([1, 1, 1])
    
    stats = zonal_stats(values, zones)
    
    z1 = stats[1]
    assert z1.count == 2 # NaN ignored
    assert z1.sum == 40.0
    assert z1.mean == 20.0

def test_zonal_stats_shape_mismatch():
    values = np.array([1.0, 2.0])
    zones = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="Shape mismatch"):
        zonal_stats(values, zones)

def test_zonal_stats_dtype_coercion():
    # Int values should be coerced to float
    values = np.array([1, 2, 3], dtype=np.int32)
    zones = np.array([1, 1, 2], dtype=np.int32)
    
    stats = zonal_stats(values, zones)
    assert stats[1].mean == 1.5
    assert stats[2].mean == 3.0
