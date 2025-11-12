import numpy as np
import pytest
from eo_processor import (
    mask_vals,
    replace_nans,
    mask_out_range,
    mask_invalid,
    mask_in_range,
    mask_scl,
)


def test_mask_vals_1d_basic_nan():
    arr = np.array([0, 1, 2, 0, 3], dtype=np.int16)
    out = mask_vals(arr, values=[0])
    assert out.dtype == np.float64
    assert np.isnan(out[[0, 3]]).all()
    assert np.array_equal(out[[1, 2, 4]], np.array([1.0, 2.0, 3.0]))


def test_mask_vals_1d_fill_and_nan_to():
    arr = np.array([0, 5, np.nan, 7], dtype=np.float64)
    out = mask_vals(arr, values=[0], fill_value=-9999.0, nan_to=-1.0)
    # 0 -> -9999 then NaNs (-9999 and original NaN) -> -1
    assert np.array_equal(out, np.array([-1.0, 5.0, -1.0, 7.0]))


def test_mask_vals_1d_only_nan_to():
    arr = np.array([1.0, np.nan, 2.0])
    out = mask_vals(arr, nan_to=0.0)
    assert np.array_equal(out, np.array([1.0, 0.0, 2.0]))


def test_mask_vals_no_values_list():
    arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    out = mask_vals(arr)
    assert np.array_equal(out, arr.astype(np.float64))


def test_mask_vals_values_not_present():
    arr = np.array([10, 11, 12], dtype=np.int32)
    out = mask_vals(arr, values=[99])
    assert np.array_equal(out, arr.astype(np.float64))


def test_replace_nans_simple():
    arr = np.array([[np.nan, 1.0], [2.0, np.nan]], dtype=np.float32)
    out = replace_nans(arr, -5.0)
    assert np.array_equal(out, np.array([[-5.0, 1.0], [2.0, -5.0]]))


def test_mask_vals_2d_multiple_values():
    arr = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.int16)
    out = mask_vals(arr, values=[0, 4])
    assert np.isnan(out[0, 0])
    assert np.isnan(out[1, 2])
    assert np.isnan(out[1, 1])
    assert out[0, 1] == 1
    assert out[0, 2] == 2
    assert out[1, 0] == 3


def test_mask_vals_3d_time_y_x():
    # shape (time, y, x)
    arr = np.array(
        [
            [[0, 1], [2, 0]],
            [[3, 4], [0, 5]],
            [[6, 0], [7, 8]],
        ],
        dtype=np.int16,
    )
    out = mask_vals(arr, values=[0])
    # All zeros masked
    zeros_positions = [(0, 0, 0), (0, 1, 1), (1, 1, 0), (2, 0, 1)]
    for idx in zeros_positions:
        assert np.isnan(out[idx])
    # Some untouched
    assert out[2, 1, 1] == 8


def test_mask_vals_4d_time_band_y_x_fill_value():
    # shape (time, band, y, x)
    arr = np.zeros((2, 2, 2, 2), dtype=np.int16)
    arr[0, 0, 0, 0] = 5
    arr[1, 1, 1, 1] = -9999
    out = mask_vals(arr, values=[0, -9999], fill_value=-1.0)
    # where mask applied becomes -1
    assert out[0, 0, 0, 0] == 5.0  # kept
    assert out[1, 1, 1, 1] == -1.0
    # random positions previously zero become -1
    assert out[0, 1, 0, 1] == -1.0
    assert out.dtype == np.float64


def test_mask_vals_order_mask_then_nan_to():
    arr = np.array([0.0, np.nan], dtype=np.float64)
    out = mask_vals(arr, values=[0.0], fill_value=np.nan, nan_to=9.0)
    assert np.array_equal(out, np.array([9.0, 9.0]))


def test_mask_vals_empty_values_equivalence():
    arr = np.random.default_rng(0).integers(0, 10, size=(3, 4)).astype(np.int16)
    out_a = mask_vals(arr, values=[])
    out_b = mask_vals(arr, values=None)
    assert np.array_equal(out_a, out_b)


def test_mask_vals_idempotent_on_float_input():
    arr = np.array([1.0, 2.0, 3.0])
    out = mask_vals(arr, values=[99.0])
    assert np.array_equal(out, arr)


def test_mask_vals_does_not_modify_input():
    arr = np.array([0, 1, 2])
    arr_copy = arr.copy()
    _ = mask_vals(arr, values=[0])
    assert np.array_equal(arr, arr_copy)


def test_mask_vals_nan_to_only_4d():
    arr = np.full((1, 1, 2, 2), np.nan, dtype=np.float64)
    out = mask_vals(arr, nan_to=0.0)
    assert np.array_equal(out, np.zeros((1, 1, 2, 2)))


def test_replace_nans_preserves_shape_and_type():
    arr = np.array([[[np.nan, 1.0]]], dtype=np.float32)  # (1,1,2)
    out = replace_nans(arr, 2.5)
    assert out.shape == (1, 1, 2)
    assert out.dtype == np.float64
    assert np.array_equal(out, np.array([[[2.5, 1.0]]]))


def test_mask_vals_values_none_nan_to_none_returns_float():
    arr = np.array([1, 2, 3], dtype=np.int8)
    out = mask_vals(arr)
    assert out.dtype == np.float64
    assert np.array_equal(out, np.array([1.0, 2.0, 3.0]))


def test_mask_vals_large_values_list():
    arr = np.array([0, 1, 2, 3, 4, 5])
    values = list(range(0, 5))
    out = mask_vals(arr, values=values)
    assert np.isnan(out[:5]).all()
    assert out[5] == 5.0


def test_mask_vals_mask_and_nan_to_no_fill_value():
    arr = np.array([0.0, 1.0, np.nan])
    out = mask_vals(arr, values=[0.0], nan_to=-7.0)
    # mask sets 0.0 -> NaN then nan_to -> -7.0
    assert np.array_equal(out, np.array([-7.0, 1.0, -7.0]))


def test_mask_vals_fill_value_without_nan_to():
    arr = np.array([1, 0, 2])
    out = mask_vals(arr, values=[0], fill_value=-5.0)
    assert np.array_equal(out, np.array([1.0, -5.0, 2.0]))


def test_mask_vals_multiple_dimensions_consistency():
    # create a 3D array replicate values along time
    base = np.array([[0, 1], [2, 3]], dtype=np.int16)
    arr = np.stack([base, base], axis=0)
    out = mask_vals(arr, values=[0, 3])
    assert np.isnan(out[0, 0, 0]) and np.isnan(out[1, 0, 0])
    assert np.isnan(out[0, 1, 1]) and np.isnan(out[1, 1, 1])
    assert out[0, 0, 1] == 1.0 and out[1, 0, 1] == 1.0
    assert out[0, 1, 0] == 2.0 and out[1, 1, 0] == 2.0


def test_mask_vals_4d_nan_to_after_fill():
    arr = np.zeros((1, 1, 1, 3), dtype=np.int16)
    arr[0, 0, 0, 1] = 9
    out = mask_vals(arr, values=[0], fill_value=np.nan, nan_to=99.0)
    # zeros -> NaN -> 99; the 9 remains
    assert np.array_equal(out, np.array([[[[99.0, 9.0, 99.0]]]]))


def test_mask_vals_accepts_tuple_values():
    arr = np.array([1, 2, 3, 4])
    out = mask_vals(arr, values=(1, 4))
    assert np.isnan(out[0]) and np.isnan(out[3])
    assert out[1] == 2.0 and out[2] == 3.0


def test_mask_vals_values_empty_list_behaviour():
    arr = np.array([0, 1, 2])
    out = mask_vals(arr, values=[])
    assert np.array_equal(out, np.array([0.0, 1.0, 2.0]))


def test_mask_vals_chained_replace_nans():
    arr = np.array([0, 1, 2])
    masked = mask_vals(arr, values=[0])
    replaced = replace_nans(masked, -1.0)
    assert np.array_equal(replaced, np.array([-1.0, 1.0, 2.0]))


def test_mask_out_range_basic():
    arr = np.array([-1.0, 0.5, 1.0, 1.5])
    out = mask_out_range(arr, min_val=0.0, max_val=1.0)
    assert np.isnan(out[0])
    assert out[1] == 0.5
    assert out[2] == 1.0
    assert np.isnan(out[3])


def test_mask_out_range_only_min():
    arr = np.array([-5, 0, 5])
    out = mask_out_range(arr, min_val=0)
    assert np.isnan(out[0])
    assert out[1] == 0
    assert out[2] == 5


def test_mask_out_range_only_max_with_fill_value():
    arr = np.array([99, 100, 101])
    out = mask_out_range(arr, max_val=100, fill_value=-999)
    assert out[0] == 99
    assert out[1] == 100
    assert out[2] == -999


def test_mask_invalid_basic():
    arr = np.array([0, 1, -9999, 2])
    out = mask_invalid(arr, invalid_values=[0, -9999])
    assert np.isnan(out[0])
    assert out[1] == 1.0
    assert np.isnan(out[2])
    assert out[3] == 2.0


def test_mask_invalid_fill_value():
    arr = np.array([0, 1, 2])
    out = mask_invalid(arr, invalid_values=[0], fill_value=-1)
    assert out[0] == -1
    assert out[1] == 1
    assert out[2] == 2


def test_mask_in_range_basic():
    arr = np.array([-1.0, 0.5, 1.0, 1.5])
    out = mask_in_range(arr, min_val=0.0, max_val=1.0)
    assert out[0] == -1.0
    assert np.isnan(out[1])
    assert np.isnan(out[2])
    assert out[3] == 1.5


def test_mask_scl_default():
    scl = np.array([0, 4, 5, 6, 7, 8, 9, 10, 11])
    out = mask_scl(scl)
    assert np.isnan(out[0])
    assert out[1] == 4.0
    assert out[2] == 5.0
    assert out[3] == 6.0
    assert out[4] == 7.0
    assert np.isnan(out[5])
    assert np.isnan(out[6])
    assert np.isnan(out[7])
    assert out[8] == 11.0


def test_mask_scl_custom_keep():
    scl = np.array([4, 8, 9])
    out = mask_scl(scl, keep_codes=[4, 9])
    assert out[0] == 4.0
    assert np.isnan(out[1])
    assert out[2] == 9.0
