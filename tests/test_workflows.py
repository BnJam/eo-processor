import numpy as np
from eo_processor import (
    land_cover_classification,
    burn_severity_assessment,
    water_body_extraction,
)

def test_land_cover_classification():
    input_data = np.random.rand(10, 10)
    result = land_cover_classification(input_data)
    assert result.shape == input_data.shape
    assert result.dtype == np.int32
    assert np.all(result == 0)

def test_burn_severity_assessment():
    pre_fire_nbr = np.random.rand(10, 10)
    post_fire_nbr = np.random.rand(10, 10)
    result = burn_severity_assessment(pre_fire_nbr, post_fire_nbr)
    assert result.shape == pre_fire_nbr.shape
    assert result.dtype == np.float64
    assert np.all(result == 0)

def test_water_body_extraction():
    input_data = np.random.rand(10, 10)
    result = water_body_extraction(input_data)
    assert result.shape == input_data.shape
    assert result.dtype == np.uint8
    assert np.all(result == 0)
