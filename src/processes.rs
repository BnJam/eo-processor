use ndarray::{s, Array1, Array2, Array3, Array4, ArrayView4, Axis};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArray4,
};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Advanced temporal and pixel-wise processing utilities.
///
/// This module introduces higher-level operations intended to better
/// showcase Rust + Rayon performance on deep temporal stacks and large
/// spatial tiles typical of Earth Observation workloads.
///
/// Functions:
/// - `moving_average_temporal`: Sliding window mean along leading time axis (1D–4D).
/// - `pixelwise_transform`: Scaled / shifted (and optionally clamped) linear transform.
///
/// All functions coerce numeric inputs to float64 via the Python side; here
/// we assume arrays already arrived as f64.
///
/// Design Notes:
/// 1. Moving average uses prefix-sum + prefix-count (and optional nan counting)
///    per series to achieve O(T) complexity independent of window size.
/// 2. For 3D / 4D arrays we reshape (T, S) where S = product of spatial/band dims
///    to iterate column-wise in parallel.
/// 3. Parallelization: columns (pixels / band-pixels) processed independently.
/// 4. Skip-NaN semantics mirror existing temporal functions:
///    - skip_na=true: exclude NaNs from mean; empty window -> NaN.
///    - skip_na=false: any NaN inside window -> output NaN for that position.
///
/// Edge Handling Modes (moving average):
/// - "same": Output time axis length == input; boundary windows shrink to available range.
/// - "valid": Only windows fully inside the series; output length = T - window + 1.
///   (Requires window <= T; window >=1)
///
/// Safety: No `unsafe` blocks; reshaping uses `into_shape` on owned
/// contiguous arrays ensuring memory safety.
///
/// Potential Future Extensions:
/// - Add alternative aggregation (median, std, min/max) via same prefix strategy.
/// - Provide dilation/erosion morphological temporal filters.
///
/// Example (Python):
/// ```python
/// from eo_processor import moving_average_temporal
/// import numpy as np
///
/// cube = np.random.rand(48, 1024, 1024)  # (time, y, x)
/// smooth = moving_average_temporal(cube, window=5, mode="same", skip_na=True)
/// assert smooth.shape == cube.shape
/// ```
#[pyfunction]
#[pyo3(signature = (arr, window, skip_na=true, mode="same"))]
pub fn moving_average_temporal(
    py: Python<'_>,
    arr: &PyAny,
    window: usize,
    skip_na: bool,
    mode: &str,
) -> PyResult<PyObject> {
    if window == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be >= 1",
        ));
    }
    if mode != "same" && mode != "valid" {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mode must be 'same' or 'valid'",
        ));
    }

    // Dispatch by dimensionality.
    if let Ok(a1) = arr.downcast::<PyArray1<f64>>() {
        let out = moving_avg_1d(a1.readonly(), window, skip_na, mode, py)?;
        return Ok(out.into_py(py));
    } else if let Ok(a2) = arr.downcast::<PyArray2<f64>>() {
        let out = moving_avg_2d(py, a2.readonly(), window, skip_na, mode)?;
        return Ok(out.into_py(py));
    } else if let Ok(a3) = arr.downcast::<PyArray3<f64>>() {
        let out = moving_avg_3d(py, a3.readonly(), window, skip_na, mode)?;
        return Ok(out.into_py(py));
    } else if let Ok(a4) = arr.downcast::<PyArray4<f64>>() {
        let out = moving_avg_4d(py, a4.readonly(), window, skip_na, mode)?;
        return Ok(out.into_py(py));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected 1D, 2D, 3D, or 4D NumPy float64 array.",
    ))
}

/// Compute moving average for a 1D (T,) series.
fn moving_avg_1d<'py>(
    series: PyReadonlyArray1<'py, f64>,
    window: usize,
    skip_na: bool,
    mode: &str,
    py: Python<'py>,
) -> PyResult<&'py PyArray1<f64>> {
    let a = series.as_array();
    let t = a.len();
    if mode == "valid" && window > t {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window cannot exceed series length in 'valid' mode",
        ));
    }

    let owned_a = a.to_owned();
    let (prefix_sum, prefix_count, prefix_nan) = build_prefix_1d(&owned_a);

    let out_len = if mode == "same" { t } else { t - window + 1 };
    let mut out = Array1::<f64>::zeros(out_len);

    if mode == "same" {
        for idx in 0..t {
            let (start, end) = window_bounds_same(idx, t, window);
            out[idx] = compute_window_mean(
                &prefix_sum,
                &prefix_count,
                &prefix_nan,
                start,
                end,
                skip_na,
                true, // variable size at edges
            );
        }
    } else {
        // valid
        for i in 0..out_len {
            let start = i;
            let end = i + window - 1;
            out[i] = compute_window_mean(
                &prefix_sum,
                &prefix_count,
                &prefix_nan,
                start,
                end,
                skip_na,
                false,
            );
        }
    }

    Ok(out.into_pyarray(py))
}

/// 2D (T, F) moving average across time axis.
fn moving_avg_2d<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<f64>,
    window: usize,
    skip_na: bool,
    mode: &str,
) -> PyResult<&'py PyArray2<f64>> {
    let a = arr.as_array();
    let t = a.shape()[0];
    let f = a.shape()[1];
    if mode == "valid" && window > t {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window cannot exceed time length in 'valid' mode",
        ));
    }
    let out_t = if mode == "same" { t } else { t - window + 1 };
    let mut out = Array2::<f64>::zeros((out_t, f));

    // Precompute prefix statistics for each feature column outside the parallel region
    // to avoid capturing non-Send PyArray references inside the Rayon closure.
    let prefixes: Vec<(Vec<f64>, Vec<usize>, Vec<usize>)> = (0..f)
        .map(|col_idx| {
            let series = a.column(col_idx).to_owned();
            build_prefix_1d(&series)
        })
        .collect();

    // Parallel over feature columns using precomputed prefixes.
    out.axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(col_idx, mut col_out)| {
            let (ref prefix_sum, ref prefix_count, ref prefix_nan) = prefixes[col_idx];
            if mode == "same" {
                for i in 0..t {
                    let (start, end) = window_bounds_same(i, t, window);
                    col_out[i] = compute_window_mean(
                        prefix_sum,
                        prefix_count,
                        prefix_nan,
                        start,
                        end,
                        skip_na,
                        true,
                    );
                }
            } else {
                for i in 0..out_t {
                    let start = i;
                    let end = i + window - 1;
                    col_out[i] = compute_window_mean(
                        prefix_sum,
                        prefix_count,
                        prefix_nan,
                        start,
                        end,
                        skip_na,
                        false,
                    );
                }
            }
        });

    Ok(out.into_pyarray(py))
}

/// 3D (T, Y, X) moving average via (T, S) reshape.
fn moving_avg_3d<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray3<f64>,
    window: usize,
    skip_na: bool,
    mode: &str,
) -> PyResult<&'py PyArray3<f64>> {
    let a = arr.as_array();
    let t = a.shape()[0];
    let y = a.shape()[1];
    let x = a.shape()[2];
    if mode == "valid" && window > t {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window cannot exceed time length in 'valid' mode",
        ));
    }
    let out_t = if mode == "same" { t } else { t - window + 1 };

    // Own contiguous copy then reshape (T, S).
    let owned = a.to_owned();
    let s = y * x;
    let reshaped = owned.into_shape((t, s)).expect("contiguous reshape (3D)");
    let mut out2 = Array2::<f64>::zeros((out_t, s));

    out2.axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(pixel_idx, mut col_out)| {
            let series = reshaped.column(pixel_idx);
            let owned_series = series.to_owned();
            let (prefix_sum, prefix_count, prefix_nan) = build_prefix_1d(&owned_series);
            if mode == "same" {
                for i in 0..t {
                    let (start, end) = window_bounds_same(i, t, window);
                    col_out[i] = compute_window_mean(
                        &prefix_sum,
                        &prefix_count,
                        &prefix_nan,
                        start,
                        end,
                        skip_na,
                        true,
                    );
                }
            } else {
                for i in 0..out_t {
                    let start = i;
                    let end = i + window - 1;
                    col_out[i] = compute_window_mean(
                        &prefix_sum,
                        &prefix_count,
                        &prefix_nan,
                        start,
                        end,
                        skip_na,
                        false,
                    );
                }
            }
        });

    // Reshape back (out_t, Y, X)
    let out3 = out2
        .into_shape((out_t, y, x))
        .expect("reshape back to (T,Y,X)");
    Ok(out3.into_pyarray(py))
}

/// 4D (T, B, Y, X) moving average via (T, S) reshape (S=B*Y*X).
fn moving_avg_4d<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray4<f64>,
    window: usize,
    skip_na: bool,
    mode: &str,
) -> PyResult<&'py PyArray4<f64>> {
    let a = arr.as_array();
    let t = a.shape()[0];
    let b = a.shape()[1];
    let y = a.shape()[2];
    let x = a.shape()[3];
    if mode == "valid" && window > t {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window cannot exceed time length in 'valid' mode",
        ));
    }
    let out_t = if mode == "same" { t } else { t - window + 1 };

    let owned = a.to_owned();
    let s = b * y * x;
    let reshaped = owned.into_shape((t, s)).expect("contiguous reshape (4D)");
    let mut out2 = Array2::<f64>::zeros((out_t, s));

    out2.axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(pixel_idx, mut col_out)| {
            let series = reshaped.column(pixel_idx);
            let owned_series = series.to_owned();
            let (prefix_sum, prefix_count, prefix_nan) = build_prefix_1d(&owned_series);
            if mode == "same" {
                for i in 0..t {
                    let (start, end) = window_bounds_same(i, t, window);
                    col_out[i] = compute_window_mean(
                        &prefix_sum,
                        &prefix_count,
                        &prefix_nan,
                        start,
                        end,
                        skip_na,
                        true,
                    );
                }
            } else {
                for i in 0..out_t {
                    let start = i;
                    let end = i + window - 1;
                    col_out[i] = compute_window_mean(
                        &prefix_sum,
                        &prefix_count,
                        &prefix_nan,
                        start,
                        end,
                        skip_na,
                        false,
                    );
                }
            }
        });

    let out4 = out2
        .into_shape((out_t, b, y, x))
        .expect("reshape back to (T,B,Y,X)");
    Ok(out4.into_pyarray(py))
}

/// Build prefix arrays for a single 1D series:
/// - prefix_sum[i] = sum of non-NaN values 0..i
/// - prefix_count[i] = count of non-NaN values 0..i
/// - prefix_nan[i] = count of NaNs 0..i
fn build_prefix_1d(series: &Array1<f64>) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    let t = series.len();
    let mut prefix_sum = Vec::with_capacity(t);
    let mut prefix_count = Vec::with_capacity(t);
    let mut prefix_nan = Vec::with_capacity(t);
    let mut s_acc = 0.0f64;
    let mut c_acc: usize = 0;
    let mut n_acc: usize = 0;

    for &v in series.iter() {
        if v.is_nan() {
            n_acc += 1;
        } else {
            s_acc += v;
            c_acc += 1;
        }
        prefix_sum.push(s_acc);
        prefix_count.push(c_acc);
        prefix_nan.push(n_acc);
    }
    (prefix_sum, prefix_count, prefix_nan)
}

/// Determine window bounds for 'same' mode (variable at edges).
fn window_bounds_same(center: usize, t: usize, window: usize) -> (usize, usize) {
    let half_left = window / 2;
    let half_right = window - half_left - 1;
    let start = center.saturating_sub(half_left);
    let end = (center + half_right).min(t - 1);
    (start, end)
}

/// Compute window mean using prefix arrays.
/// variable_size: when true divides by actual non-NaN count (skip_na=true)
/// or actual window length (skip_na=false); at edges window length shrinks.
fn compute_window_mean(
    prefix_sum: &[f64],
    prefix_cnt_non_nan: &[usize],
    prefix_nan_count: &[usize],
    start: usize,
    end: usize,
    skip_na: bool,
    _variable_size: bool,
) -> f64 {
    let range_sum = prefix_sum[end]
        - if start > 0 {
            prefix_sum[start - 1]
        } else {
            0.0
        };
    let non_nan_count = prefix_cnt_non_nan[end]
        - if start > 0 {
            prefix_cnt_non_nan[start - 1]
        } else {
            0
        };
    let nan_count = prefix_nan_count[end]
        - if start > 0 {
            prefix_nan_count[start - 1]
        } else {
            0
        };
    let window_len = end - start + 1;

    if skip_na {
        if non_nan_count == 0 {
            f64::NAN
        } else {
            range_sum / non_nan_count as f64
        }
    } else if nan_count > 0 {
        f64::NAN
    } else {
        // If variable size (same mode at edges), denominator = effective window length
        let denom = window_len;
        range_sum / denom as f64
    }
}

/// Pixel-wise linear transform with optional clamping.
///
/// result = clamp( value * scale + offset )
///
/// NaNs propagate unchanged.
///
/// Supports 1D–4D arrays.
///
/// Parameters:
/// - scale (default 1.0)
/// - offset (default 0.0)
/// - clamp_min (None = no lower clamp)
/// - clamp_max (None = no upper clamp)
///
/// Example:
/// ```python
/// from eo_processor import pixelwise_transform
/// import numpy as np
/// arr = np.random.rand(2048, 2048)
/// stretched = pixelwise_transform(arr, scale=1.2, offset=-0.1, clamp_min=0, clamp_max=1)
/// ```
#[pyfunction]
#[pyo3(signature = (arr, scale=1.0, offset=0.0, clamp_min=None, clamp_max=None))]
pub fn pixelwise_transform(
    py: Python<'_>,
    arr: &PyAny,
    scale: f64,
    offset: f64,
    clamp_min: Option<f64>,
    clamp_max: Option<f64>,
) -> PyResult<PyObject> {
    let apply = |v: f64| -> f64 {
        if v.is_nan() {
            return v;
        }
        let mut r = v * scale + offset;
        if let Some(mn) = clamp_min {
            if r < mn {
                r = mn;
            }
        }
        if let Some(mx) = clamp_max {
            if r > mx {
                r = mx;
            }
        }
        r
    };

    if let Ok(a1) = arr.downcast::<PyArray1<f64>>() {
        let out_arr = a1.readonly().as_array().mapv(apply);
        return Ok(out_arr.into_pyarray(py).into_py(py));
    } else if let Ok(a2) = arr.downcast::<PyArray2<f64>>() {
        let out_arr = a2.readonly().as_array().mapv(apply);
        return Ok(out_arr.into_pyarray(py).into_py(py));
    } else if let Ok(a3) = arr.downcast::<PyArray3<f64>>() {
        let out_arr = a3.readonly().as_array().mapv(apply);
        return Ok(out_arr.into_pyarray(py).into_py(py));
    } else if let Ok(a4) = arr.downcast::<PyArray4<f64>>() {
        let out_arr = a4.readonly().as_array().mapv(apply);
        return Ok(out_arr.into_pyarray(py).into_py(py));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected float64 NumPy array (1D–4D).",
    ))
}

/// Stride-based moving average along time axis.
/// Computes the moving average (same or valid mode) and then samples
/// window centers every `stride` steps to reduce temporal resolution.
/// Useful for large deep temporal stacks where full-resolution output
/// is unnecessary.
///
/// Parameters
/// ----------
/// arr : NumPy array (1D–4D)
/// window : usize >= 1 (same semantics as moving_average_temporal)
/// stride : usize >= 1 (sampling interval along time in output)
/// skip_na : bool (NaN handling identical to moving_average_temporal)
/// mode : {"same","valid"}
///
/// Output shape
/// ------------
/// If mode == "same":
///   input T -> ceil(T / stride)
/// If mode == "valid":
///   valid_len = T - window + 1 -> ceil(valid_len / stride)
///
/// Implementation notes:
/// 1. Reuses existing per-dimension helpers (moving_avg_*).
/// 2. After computing the full moving average output, creates a strided
///    sampling along the leading time axis and reshapes remaining axes.
/// 3. No additional parallelism added (cost dominated by the base moving average).
///
/// Example (Python)
/// ----------------
/// ```python
/// from eo_processor import moving_average_temporal_stride
/// import numpy as np
///
/// cube = np.random.rand(96, 512, 512)
/// downsampled = moving_average_temporal_stride(cube, window=7, stride=4, mode="same")
/// print(downsampled.shape)  # (24, 512, 512)
/// ```
#[pyfunction]
#[pyo3(signature = (arr, window, stride, skip_na=true, mode="same"))]
pub fn moving_average_temporal_stride(
    py: Python<'_>,
    arr: &PyAny,
    window: usize,
    stride: usize,
    skip_na: bool,
    mode: &str,
) -> PyResult<PyObject> {
    if window == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be >= 1",
        ));
    }
    if stride == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "stride must be >= 1",
        ));
    }
    if mode != "same" && mode != "valid" {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mode must be 'same' or 'valid'",
        ));
    }

    // Helper to stride a 1D first axis out of an ndarray and return owned Array
    fn stride_take_1d<T: Clone>(full: &ndarray::Array1<T>, stride: usize) -> ndarray::Array1<T> {
        assert!(stride >= 1, "stride must be >= 1");
        let len = full.len();
        let iter = (0..len).step_by(stride).map(|i| full[i].clone());
        ndarray::Array1::from_iter(iter)
    }

    // 1D
    if let Ok(a1) = arr.downcast::<PyArray1<f64>>() {
        let full = moving_avg_1d(a1.readonly(), window, skip_na, mode, py)?;
        let full_arr = full.readonly().as_array().to_owned();
        let out = stride_take_1d(&full_arr, stride);
        return Ok(out.into_pyarray(py).into_py(py));
    }
    // 2D
    if let Ok(a2) = arr.downcast::<PyArray2<f64>>() {
        let t = a2.shape()[0];
        if mode == "valid" && window > t {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "window cannot exceed time length in 'valid' mode",
            ));
        }
        let full = moving_avg_2d(py, a2.readonly(), window, skip_na, mode)?;
        // Convert to an owned ndarray::Array2<f64>
        let full_owned = full.readonly().as_array().to_owned();

        let out_t = full_owned.shape()[0];
        let sampled_len = out_t.div_ceil(stride);
        let mut sampled = Array2::<f64>::zeros((sampled_len, full_owned.shape()[1]));

        for (dst_t, src_t) in (0..out_t).step_by(stride).enumerate() {
            sampled
                .index_axis_mut(Axis(0), dst_t)
                .assign(&full_owned.index_axis(Axis(0), src_t));
        }

        return Ok(sampled.into_pyarray(py).into_py(py));
    }
    // 3D
    if let Ok(a3) = arr.downcast::<PyArray3<f64>>() {
        let t = a3.shape()[0];
        if mode == "valid" && window > t {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "window cannot exceed time length in 'valid' mode",
            ));
        }
        let full = moving_avg_3d(py, a3.readonly(), window, skip_na, mode)?;
        let full_owned = full.readonly().as_array().to_owned();
        let out_t = full_owned.shape()[0];
        let sampled_len = out_t.div_ceil(stride);
        let mut sampled =
            Array3::<f64>::zeros((sampled_len, full_owned.shape()[1], full_owned.shape()[2]));
        let mut src_t = 0usize;
        let mut dst_t = 0usize;
        while src_t < out_t {
            sampled
                .index_axis_mut(Axis(0), dst_t)
                .assign(&full_owned.index_axis(Axis(0), src_t));
            dst_t += 1;
            src_t += stride;
        }
        return Ok(sampled.into_pyarray(py).into_py(py));
    }
    // 4D
    if let Ok(a4) = arr.downcast::<PyArray4<f64>>() {
        let t = a4.shape()[0];
        if mode == "valid" && window > t {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "window cannot exceed time length in 'valid' mode",
            ));
        }
        let full = moving_avg_4d(py, a4.readonly(), window, skip_na, mode)?;
        let full_owned = full.readonly().as_array().to_owned();
        let out_t = full_owned.shape()[0];
        let sampled_len = out_t.div_ceil(stride);
        let mut sampled = Array4::<f64>::zeros((
            sampled_len,
            full_owned.shape()[1],
            full_owned.shape()[2],
            full_owned.shape()[3],
        ));
        let mut src_t = 0usize;
        let mut dst_t = 0usize;
        while src_t < out_t {
            sampled
                .index_axis_mut(Axis(0), dst_t)
                .assign(&full_owned.index_axis(Axis(0), src_t));
            dst_t += 1;
            src_t += stride;
        }
        return Ok(sampled.into_pyarray(py).into_py(py));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected 1D, 2D, 3D, or 4D NumPy float64 array.",
    ))
}
#[pyfunction]
#[pyo3(signature = (arr, weights, skip_na = true))]
pub fn temporal_composite(
    py: Python<'_>,
    arr: &PyAny,
    weights: PyReadonlyArray1<f64>,
    skip_na: bool,
) -> PyResult<PyObject> {
    if let Ok(arr4d) = arr.downcast::<numpy::PyArray4<f64>>() {
        Ok(temporal_composite_4d(py, arr4d.readonly().as_array(), weights, skip_na)?.into_py(py))
    } else if let Ok(arr4d_u16) = arr.downcast::<numpy::PyArray4<u16>>() {
        let arr4d_f64 = arr4d_u16.readonly().as_array().mapv(|x| x as f64);
        Ok(temporal_composite_4d(py, arr4d_f64.view(), weights, skip_na)?.into_py(py))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected a 4D NumPy array of f64 or u16.",
        ))
    }
}

pub fn temporal_composite_4d<'py>(
    py: Python<'py>,
    arr: ArrayView4<f64>,
    weights: PyReadonlyArray1<f64>,
    skip_na: bool,
) -> PyResult<&'py PyArray3<f64>> {
    let weights_array = weights.as_array();
    let shape = arr.shape();
    let (num_bands, height, width) = (shape[1], shape[2], shape[3]);

    if shape[0] != weights_array.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "The length of the weights array must match the temporal dimension of the input array.",
        ));
    }

    let mut result = Array3::<f64>::zeros((num_bands, height, width));

    result
        .indexed_iter_mut()
        .par_bridge()
        .for_each(|((b, r, c), pixel)| {
            let mut series: Vec<(f64, f64)> = arr
                .slice(s![.., b, r, c])
                .iter()
                .zip(weights_array.iter())
                .map(|(v, w)| (*v, *w))
                .collect();

            if skip_na {
                series.retain(|(v, _)| !v.is_nan());
            }

            if series.is_empty() {
                *pixel = f64::NAN;
                return;
            }

            series.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let total_weight: f64 = series.iter().map(|(_, w)| *w).sum();
            let mut accumulated_weight = 0.0;
            let median_weight = total_weight / 2.0;

            for (value, weight) in series {
                accumulated_weight += weight;
                if accumulated_weight >= median_weight {
                    *pixel = value;
                    return;
                }
            }
        });

    Ok(result.into_pyarray(py))
}
#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArray1;

    #[test]
    fn test_moving_avg_1d_same_basic() {
        Python::with_gil(|py| {
            let data = vec![1.0, 2.0, 3.0, 4.0];
            let arr = PyArray1::from_vec(py, data.clone());
            let out_obj = moving_average_temporal(py, arr, 3, true, "same").unwrap();
            let out = out_obj.extract::<&PyArray1<f64>>(py).unwrap().readonly();
            let rust = out.as_array();
            // Manual edge handling (variable window sizes):
            // idx0: window [0..1] -> (1+2)/2 = 1.5
            // idx1: [0..2] -> (1+2+3)/3 = 2.0
            // idx2: [1..3] -> (2+3+4)/3 = 3.0
            // idx3: [2..3] -> (3+4)/2 = 3.5
            let expected = [1.5, 2.0, 3.0, 3.5];
            for (a, b) in rust.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn test_moving_avg_1d_valid() {
        Python::with_gil(|py| {
            let data = vec![1.0, 2.0, 3.0, 4.0];
            let arr = PyArray1::from_vec(py, data.clone());
            let out_obj = moving_average_temporal(py, arr, 2, true, "valid").unwrap();
            let out = out_obj.extract::<&PyArray1<f64>>(py).unwrap().readonly();
            let rust = out.as_array();
            // Windows: [1,2] [2,3] [3,4]
            let expected = [1.5, 2.5, 3.5];
            for (a, b) in rust.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn test_pixelwise_transform_basic() {
        Python::with_gil(|py| {
            let data = vec![0.0, 0.5, 1.0];
            let arr = PyArray1::from_vec(py, data.clone());
            let out_obj = pixelwise_transform(py, arr, 2.0, -0.5, Some(0.0), Some(1.0)).unwrap();
            let out = out_obj.extract::<&PyArray1<f64>>(py).unwrap().readonly();
            let rust = out.as_array();
            // Transform: clamp((v*2 - 0.5), 0, 1)
            // 0.0 -> -0.5 -> 0.0
            // 0.5 -> 0.5 -> 0.5
            // 1.0 -> 1.5 -> clamp to 1.0
            let expected = [0.0, 0.5, 1.0];
            for (a, b) in rust.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }
}
