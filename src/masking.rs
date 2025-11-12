//! Masking utilities for EO workflows.
//!
//! This module introduces a generic `mask_vals` function supporting 1D, 2D, 3D, and 4D
//! NumPy arrays (float64) with optional value-based masking and NaN replacement.
//!
//! Design Goals:
//! - Accept any numeric NumPy dtype (coerced to float64 internally).
//! - Allow masking of specific coded values (e.g., Sentinel-2 SCL classes, fill values).
//! - Allow replacing existing NaNs with a user-specified numeric value.
//! - Preserve input shape and dimensionality.
//! - No `unsafe` code; clear explicit dispatch.
//!
//! To expose these functions to Python you MUST add:
//!     `pub mod masking;`
//! to `src/lib.rs` and then add:
//!     `m.add_function(wrap_pyfunction!(masking::mask_vals, m)?)?;`
//!     `m.add_function(wrap_pyfunction!(masking::replace_nans, m)?)?;`
//! in the `_core` initialization.
//!
//! Suggested Convenience Wrappers (not yet implemented here):
//! - `mask_scl(scl_array, keep_codes=[4,5,6,7,11], fill_value=f64::NAN)`
//!   Sentinel-2 Scene Classification Layer: drop cloud/shadow/snow codes.
//! - `mask_in_range(arr, min=None, max=None, fill_value=f64::NAN)`
//!   Mask values inside a numeric range.
//! - `mask_out_range(arr, min=None, max=None, fill_value=f64::NAN)`
//!   Mask values outside a numeric range.
//! - `mask_invalid(arr, invalid=[0], fill_value=f64::NAN)`
//!   Quickly mask sentinel zeros.
//! - `replace_nans(arr, value=0.0)`
//!   Convenience alias already provided below.
//! - `mask_cloud_probability(prob_arr, threshold=0.5, fill_value=f64::NAN)`
//!
//! Checklist for integrating new public API (per repository instructions):
//! 1. Add module + function registration in Rust `lib.rs`
//! 2. Export Python wrappers in `python/eo_processor/__init__.py`
//! 3. Add type stubs in `python/eo_processor/__init__.pyi`
//! 4. Add unit tests in `tests/` (e.g., `tests/test_masking.py`) covering 1D–4D cases
//! 5. Update README (API Summary & examples)
//! 6. Bump version (likely a minor release)
//!
//! Example Python usage (after integration):
//! ```python
//! from eo_processor import mask_vals, replace_nans
//! import numpy as np
//! data = np.array([[0, 42, 5], [42, 1, 0]], dtype=np.int16)
//! masked = mask_vals(data, values=[0, 42])  # -> float64 array with 0 & 42 replaced by NaN
//! clean  = replace_nans(masked, value=-9999.0)  # convert all NaNs to -9999.0
//! ```
//!
//! Handling NaNs:
//! - If `fill_value` is `None` (or left as default) masked codes become `NaN`.
//! - If `fill_value` is provided (e.g., -9999.0) masked codes use that numeric value.
//! - If `nan_to` is provided, existing NaNs (including those just created) are converted
//!   to `nan_to` AFTER the value masking step.
//!
//! Performance Notes:
//! - Implements straightforward element-wise loops. For extremely large arrays, a future
//!   optimization may leverage rayon + owned buffers. Current approach keeps code minimal
//!   and safe without extra allocations (besides coercion).
//!
//! Edge Cases:
//! - Empty `values` list → no value masking (only possible NaN replacement).
//! - If input dtype is already float64, no coercion cost.
//! - For integer-coded masks, exact equality is used (no tolerance).
//!
//! Future Extensions:
//! - Optional tolerance parameter for near-equality float masking.
//! - Boolean mask input (separate function) for advanced workflows.
//! - Composable predicates (greater-than / less-than) integrated with value sets.

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

/// Attempt to coerce an arbitrary Python object to a readonly 1D float64 NumPy array.
fn coerce_1d<'py>(obj: &'py PyAny) -> PyResult<PyReadonlyArray1<'py, f64>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<f64>>() {
        return Ok(arr);
    }
    let coerced = obj.call_method1("astype", ("float64",))?;
    coerced.extract::<PyReadonlyArray1<f64>>()
}

/// Attempt to coerce an arbitrary Python object to a readonly 2D float64 NumPy array.
fn coerce_2d<'py>(obj: &'py PyAny) -> PyResult<PyReadonlyArray2<'py, f64>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f64>>() {
        return Ok(arr);
    }
    let coerced = obj.call_method1("astype", ("float64",))?;
    coerced.extract::<PyReadonlyArray2<f64>>()
}

/// Attempt to coerce an arbitrary Python object to a readonly 3D float64 NumPy array.
fn coerce_3d<'py>(obj: &'py PyAny) -> PyResult<PyReadonlyArray3<'py, f64>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray3<f64>>() {
        return Ok(arr);
    }
    let coerced = obj.call_method1("astype", ("float64",))?;
    coerced.extract::<PyReadonlyArray3<f64>>()
}

/// Attempt to coerce an arbitrary Python object to a readonly 4D float64 NumPy array.
fn coerce_4d<'py>(obj: &'py PyAny) -> PyResult<PyReadonlyArray4<'py, f64>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray4<f64>>() {
        return Ok(arr);
    }
    let coerced = obj.call_method1("astype", ("float64",))?;
    coerced.extract::<PyReadonlyArray4<f64>>()
}

/// Internal helper: apply masking logic to a slice of values (in place).
fn apply_mask_slice(
    values: &mut [f64],
    mask_vals: Option<&[f64]>,
    fill_value: f64,
    nan_to: Option<f64>,
) {
    for v in values.iter_mut() {
        let mut is_masked = false;

        if let Some(mask_list) = mask_vals {
            if mask_list.contains(v) {
                is_masked = true;
                *v = fill_value;
            }
        }

        // If the original value was NaN or became NaN, and user wants replacement.
        if let Some(nan_replacement) = nan_to {
            if v.is_nan() || is_masked {
                *v = nan_replacement;
            }
        }
    }
}

/// Generic value masking across dimensions.
/// - `arr`: Input NumPy array (1D, 2D, 3D, or 4D), any numeric dtype.
/// - `values`: Optional list of numeric codes to replace (exact equality).
/// - `fill_value`: Value to assign to masked codes (default NaN if None passed).
/// - `nan_to`: Optional replacement for existing NaNs (including those produced by masking).
///
/// Returns a new float64 NumPy array of identical shape.
///
/// Examples:
///     mask_vals(arr, Some(vec![0.0, 255.0]), None, None)        // mask 0 & 255 -> NaN
///     mask_vals(arr, Some(vec![0.0]), Some(-9999.0), None)      // mask 0 -> -9999.0
///     mask_vals(arr, None, None, Some(0.0))                     // replace NaNs with 0.0
///     mask_vals(arr, Some(vec![1.0, 2.0]), None, Some(-1.0))    // mask 1 & 2 -> NaN, then NaNs -> -1.0
#[pyfunction]
#[pyo3(signature = (arr, values=None, fill_value=None, nan_to=None))]
pub fn mask_vals(
    py: Python<'_>,
    arr: &PyAny,
    values: Option<Vec<f64>>,
    fill_value: Option<f64>,
    nan_to: Option<f64>,
) -> PyResult<PyObject> {
    let fill = fill_value.unwrap_or(f64::NAN);
    let mask_list: Option<Vec<f64>> = values;

    // Dispatch by dimension
    if let Ok(a1) = coerce_1d(arr) {
        let mut out = a1.as_array().to_owned();
        apply_mask_slice(
            out.as_slice_mut().unwrap(),
            mask_list.as_deref(),
            fill,
            nan_to,
        );
        return Ok(out.into_pyarray(py).into_py(py));
    } else if let Ok(a2) = coerce_2d(arr) {
        let mut out = a2.as_array().to_owned();
        for mut row in out.rows_mut() {
            apply_mask_slice(
                row.as_slice_mut().unwrap(),
                mask_list.as_deref(),
                fill,
                nan_to,
            );
        }
        return Ok(out.into_pyarray(py).into_py(py));
    } else if let Ok(a3) = coerce_3d(arr) {
        let mut out = a3.as_array().to_owned();
        for mut plane in out.outer_iter_mut() {
            for mut row in plane.rows_mut() {
                apply_mask_slice(
                    row.as_slice_mut().unwrap(),
                    mask_list.as_deref(),
                    fill,
                    nan_to,
                );
            }
        }
        return Ok(out.into_pyarray(py).into_py(py));
    } else if let Ok(a4) = coerce_4d(arr) {
        let mut out = a4.as_array().to_owned();
        for mut block in out.outer_iter_mut() {
            for mut plane in block.outer_iter_mut() {
                for mut row in plane.rows_mut() {
                    apply_mask_slice(
                        row.as_slice_mut().unwrap(),
                        mask_list.as_deref(),
                        fill,
                        nan_to,
                    );
                }
            }
        }
        return Ok(out.into_pyarray(py).into_py(py));
    }

    Err(PyTypeError::new_err(
        "Expected a 1D, 2D, 3D, or 4D numeric NumPy array.",
    ))
}

/// Replace NaNs with a specified numeric value across 1D–4D arrays.
/// More explicit alias to `mask_vals(arr, None, None, Some(value))`.
///
/// Example:
///     replace_nans(arr, 0.0)
#[pyfunction]
pub fn replace_nans(py: Python<'_>, arr: &PyAny, value: f64) -> PyResult<PyObject> {
    // Delegate to mask_vals with nan_to = value
    mask_vals(py, arr, None, None, Some(value))
}

/// Internal helper for range masking.
fn apply_mask_out_range_slice(
    values: &mut [f64],
    min: Option<f64>,
    max: Option<f64>,
    fill_value: f64,
) {
    for v in values.iter_mut() {
        let mut is_masked = false;
        if let Some(min_val) = min {
            if *v < min_val {
                is_masked = true;
            }
        }
        if let Some(max_val) = max {
            if *v > max_val {
                is_masked = true;
            }
        }

        if is_masked {
            *v = fill_value;
        }
    }
}

/// Mask values outside a specified numeric range [min, max].
///
/// Parameters:
/// - `arr`: Input NumPy array (1D–4D), any numeric dtype.
/// - `min`: Optional minimum valid value (inclusive).
/// - `max`: Optional maximum valid value (inclusive).
/// - `fill_value`: Value to assign to masked codes (default NaN).
///
/// If `min` is None, there is no lower bound. If `max` is None, no upper bound.
#[pyfunction]
#[pyo3(signature = (arr, min=None, max=None, fill_value=None))]
pub fn mask_out_range(
    py: Python<'_>,
    arr: &PyAny,
    min: Option<f64>,
    max: Option<f64>,
    fill_value: Option<f64>,
) -> PyResult<PyObject> {
    let fill = fill_value.unwrap_or(f64::NAN);

    // Dispatch by dimension
    if let Ok(a1) = coerce_1d(arr) {
        let mut out = a1.as_array().to_owned();
        apply_mask_out_range_slice(out.as_slice_mut().unwrap(), min, max, fill);
        return Ok(out.into_pyarray(py).into_py(py));
    } else if let Ok(a2) = coerce_2d(arr) {
        let mut out = a2.as_array().to_owned();
        for mut row in out.rows_mut() {
            apply_mask_out_range_slice(row.as_slice_mut().unwrap(), min, max, fill);
        }
        return Ok(out.into_pyarray(py).into_py(py));
    } else if let Ok(a3) = coerce_3d(arr) {
        let mut out = a3.as_array().to_owned();
        for mut plane in out.outer_iter_mut() {
            for mut row in plane.rows_mut() {
                apply_mask_out_range_slice(row.as_slice_mut().unwrap(), min, max, fill);
            }
        }
        return Ok(out.into_pyarray(py).into_py(py));
    } else if let Ok(a4) = coerce_4d(arr) {
        let mut out = a4.as_array().to_owned();
        for mut block in out.outer_iter_mut() {
            for mut plane in block.outer_iter_mut() {
                for mut row in plane.rows_mut() {
                    apply_mask_out_range_slice(row.as_slice_mut().unwrap(), min, max, fill);
                }
            }
        }
        return Ok(out.into_pyarray(py).into_py(py));
    }

    Err(PyTypeError::new_err(
        "Expected a 1D, 2D, 3D, or 4D numeric NumPy array.",
    ))
}

/// Convenience wrapper to mask a list of common invalid sentinel values.
///
/// Common invalid values in remote sensing data include 0, -9999, etc.
/// This function is an alias for `mask_vals`.
///
/// Parameters:
/// - `arr`: Input NumPy array.
/// - `invalid_values`: List of numeric codes to mask.
/// - `fill_value`: Value to use for masked positions (default NaN).
#[pyfunction]
#[pyo3(signature = (arr, invalid_values, fill_value=None))]
pub fn mask_invalid(
    py: Python<'_>,
    arr: &PyAny,
    invalid_values: Vec<f64>,
    fill_value: Option<f64>,
) -> PyResult<PyObject> {
    // This is a direct wrapper around `mask_vals`
    mask_vals(py, arr, Some(invalid_values), fill_value, None)
}

/// Internal helper for in-range masking.
fn apply_mask_in_range_slice(
    values: &mut [f64],
    min: Option<f64>,
    max: Option<f64>,
    fill_value: f64,
) {
    for v in values.iter_mut() {
        let mut is_masked = false;
        if let Some(min_val) = min {
            if *v >= min_val {
                if let Some(max_val) = max {
                    if *v <= max_val {
                        is_masked = true;
                    }
                } else {
                    is_masked = true; // No max, so anything >= min is masked
                }
            }
        } else if let Some(max_val) = max {
            if *v <= max_val {
                is_masked = true; // No min, so anything <= max is masked
            }
        }

        if is_masked {
            *v = fill_value;
        }
    }
}

/// Mask values inside a specified numeric range [min, max].
#[pyfunction]
#[pyo3(signature = (arr, min=None, max=None, fill_value=None))]
pub fn mask_in_range(
    py: Python<'_>,
    arr: &PyAny,
    min: Option<f64>,
    max: Option<f64>,
    fill_value: Option<f64>,
) -> PyResult<PyObject> {
    let fill = fill_value.unwrap_or(f64::NAN);

    if let Ok(a1) = coerce_1d(arr) {
        let mut out = a1.as_array().to_owned();
        apply_mask_in_range_slice(out.as_slice_mut().unwrap(), min, max, fill);
        return Ok(out.into_pyarray(py).into_py(py));
    } else if let Ok(a2) = coerce_2d(arr) {
        let mut out = a2.as_array().to_owned();
        for mut row in out.rows_mut() {
            apply_mask_in_range_slice(row.as_slice_mut().unwrap(), min, max, fill);
        }
        return Ok(out.into_pyarray(py).into_py(py));
    } else if let Ok(a3) = coerce_3d(arr) {
        let mut out = a3.as_array().to_owned();
        for mut plane in out.outer_iter_mut() {
            for mut row in plane.rows_mut() {
                apply_mask_in_range_slice(row.as_slice_mut().unwrap(), min, max, fill);
            }
        }
        return Ok(out.into_pyarray(py).into_py(py));
    } else if let Ok(a4) = coerce_4d(arr) {
        let mut out = a4.as_array().to_owned();
        for mut block in out.outer_iter_mut() {
            for mut plane in block.outer_iter_mut() {
                for mut row in plane.rows_mut() {
                    apply_mask_in_range_slice(row.as_slice_mut().unwrap(), min, max, fill);
                }
            }
        }
        return Ok(out.into_pyarray(py).into_py(py));
    }

    Err(PyTypeError::new_err(
        "Expected a 1D, 2D, 3D, or 4D numeric NumPy array.",
    ))
}

/// Mask a Sentinel-2 Scene Classification Layer (SCL) array.
#[pyfunction]
#[pyo3(signature = (scl, keep_codes=None, fill_value=None))]
pub fn mask_scl(
    py: Python<'_>,
    scl: &PyAny,
    keep_codes: Option<Vec<f64>>,
    fill_value: Option<f64>,
) -> PyResult<PyObject> {
    // Default S2 codes to KEEP: vegetation (4,5), water (6), bare soil (7), snow (11)
    let default_keep = vec![4.0, 5.0, 6.0, 7.0, 11.0];
    let codes_to_keep = keep_codes.unwrap_or(default_keep);

    let arr_float = if let Ok(a) = scl.extract::<PyReadonlyArray1<f64>>() {
        a.to_owned_array()
    } else {
        scl.call_method1("astype", ("float64",))?
            .extract::<PyReadonlyArray1<f64>>()?
            .to_owned_array()
    };

    let mut out = arr_float.to_owned();
    for v in out.iter_mut() {
        if !codes_to_keep.contains(v) {
            *v = fill_value.unwrap_or(f64::NAN);
        }
    }

    let result_array = out.into_dyn();
    Ok(result_array.into_pyarray(py).into_py(py))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArray2};

    #[test]
    fn test_mask_vals_basic_1d() {
        Python::with_gil(|py| {
            let data = vec![0.0, 1.0, 2.0, 42.0];
            let arr = PyArray1::from_vec(py, data);
            let result_obj = mask_vals(py, arr, Some(vec![0.0, 42.0]), None, None).unwrap();
            let result = result_obj
                .extract::<&PyArray1<f64>>(py)
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();
            assert!(result[0].is_nan());
            assert_eq!(result[1], 1.0);
            assert_eq!(result[2], 2.0);
            assert!(result[3].is_nan());
        });
    }

    #[test]
    fn test_mask_vals_with_fill_and_nan_to() {
        Python::with_gil(|py| {
            let data = vec![0.0, f64::NAN, 5.0];
            let arr = PyArray1::from_vec(py, data);
            // Mask 0.0 -> -9999, then replace all NaNs (including masked) with -1
            let result_obj =
                mask_vals(py, arr, Some(vec![0.0]), Some(-9999.0), Some(-1.0)).unwrap();
            let result = result_obj
                .extract::<&PyArray1<f64>>(py)
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();
            assert_eq!(result[0], -1.0); // masked then converted
            assert_eq!(result[1], -1.0); // original NaN replaced
            assert_eq!(result[2], 5.0);
        });
    }

    #[test]
    fn test_replace_nans() {
        Python::with_gil(|py| {
            let data = vec![1.0, f64::NAN, 3.0];
            let arr = PyArray1::from_vec(py, data);
            let result_obj = replace_nans(py, arr, 0.0).unwrap();
            let result = result_obj
                .extract::<&PyArray1<f64>>(py)
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();
            assert_eq!(result[0], 1.0);
            assert_eq!(result[1], 0.0);
            assert_eq!(result[2], 3.0);
        });
    }

    #[test]
    fn test_mask_vals_2d() {
        Python::with_gil(|py| {
            let data = vec![vec![1.0, 0.0], vec![42.0, 5.0]];
            let arr = PyArray2::from_vec2(py, &data).unwrap();
            let result_obj = mask_vals(py, arr, Some(vec![0.0, 42.0]), None, None).unwrap();
            let result = result_obj
                .extract::<&PyArray2<f64>>(py)
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();
            assert_eq!(result.shape(), &[2, 2]);
            assert_eq!(result[[0, 0]], 1.0);
            assert!(result[[0, 1]].is_nan());
            assert!(result[[1, 0]].is_nan());
            assert_eq!(result[[1, 1]], 5.0);
        });
    }
}
