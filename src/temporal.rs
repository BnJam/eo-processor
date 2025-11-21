use crate::CoreError;
use ndarray::{s, Array1, Array2, Array3, Zip};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArray4,
};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
#[pyo3(signature = (arr, skip_na=true))]
pub fn temporal_mean(py: Python<'_>, arr: &PyAny, skip_na: bool) -> PyResult<PyObject> {
    if let Ok(arr1d) = arr.downcast::<numpy::PyArray1<f64>>() {
        Ok(temporal_mean_1d(arr1d.readonly(), skip_na).into_py(py))
    } else if let Ok(arr2d) = arr.downcast::<numpy::PyArray2<f64>>() {
        Ok(temporal_mean_2d(py, arr2d.readonly(), skip_na).into_py(py))
    } else if let Ok(arr3d) = arr.downcast::<numpy::PyArray3<f64>>() {
        Ok(temporal_mean_3d(py, arr3d.readonly(), skip_na).into_py(py))
    } else if let Ok(arr4d) = arr.downcast::<numpy::PyArray4<f64>>() {
        Ok(temporal_mean_4d(py, arr4d.readonly(), skip_na).into_py(py))
    } else {
        Err(
            CoreError::InvalidArgument("Expected a 1D, 2D, 3D, or 4D NumPy array.".to_string())
                .into(),
        )
    }
}

#[pyfunction]
#[pyo3(signature = (arr, skip_na=true))]
pub fn temporal_std(py: Python<'_>, arr: &PyAny, skip_na: bool) -> PyResult<PyObject> {
    if let Ok(arr1d) = arr.downcast::<numpy::PyArray1<f64>>() {
        Ok(temporal_std_1d(arr1d.readonly(), skip_na).into_py(py))
    } else if let Ok(arr2d) = arr.downcast::<numpy::PyArray2<f64>>() {
        Ok(temporal_std_2d(py, arr2d.readonly(), skip_na).into_py(py))
    } else if let Ok(arr3d) = arr.downcast::<numpy::PyArray3<f64>>() {
        Ok(temporal_std_3d(py, arr3d.readonly(), skip_na).into_py(py))
    } else if let Ok(arr4d) = arr.downcast::<numpy::PyArray4<f64>>() {
        Ok(temporal_std_4d(py, arr4d.readonly(), skip_na).into_py(py))
    } else {
        Err(
            CoreError::InvalidArgument("Expected a 1D, 2D, 3D, or 4D NumPy array.".to_string())
                .into(),
        )
    }
}

fn temporal_mean_1d(arr: PyReadonlyArray1<f64>, skip_na: bool) -> f64 {
    let array = arr.as_array();
    if skip_na {
        let (sum, count) = array.iter().fold((0.0, 0), |(acc_s, acc_c), &v| {
            if v.is_nan() {
                (acc_s, acc_c)
            } else {
                (acc_s + v, acc_c + 1)
            }
        });
        if count == 0 {
            f64::NAN
        } else {
            sum / count as f64
        }
    } else if array.iter().any(|v| v.is_nan()) {
        f64::NAN
    } else {
        array.mean().unwrap_or(f64::NAN)
    }
}

fn temporal_mean_2d<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<f64>,
    skip_na: bool,
) -> &'py PyArray1<f64> {
    let array = arr.as_array();
    let (t_len, num_bands) = array.dim();
    let mut sum_arr = Array1::<f64>::zeros(num_bands);
    let mut count_arr = Array1::<u64>::zeros(num_bands);

    for t in 0..t_len {
        let frame = array.slice(s![t, ..]);
        Zip::from(&mut sum_arr)
            .and(&mut count_arr)
            .and(&frame)
            .for_each(|s, c, &v| {
                if skip_na {
                    if !v.is_nan() {
                        *s += v;
                        *c += 1;
                    }
                } else {
                    *s += v;
                    *c += 1;
                }
            });
    }

    let mut result = Array1::<f64>::zeros(num_bands);
    Zip::from(&mut result)
        .and(&sum_arr)
        .and(&count_arr)
        .for_each(|r, &s, &c| {
            if skip_na {
                *r = if c == 0 { f64::NAN } else { s / c as f64 };
            } else {
                *r = s / c as f64;
            }
        });
    result.into_pyarray(py)
}

fn temporal_mean_3d<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray3<f64>,
    skip_na: bool,
) -> &'py PyArray2<f64> {
    let array = arr.as_array();
    let (t_len, height, width) = array.dim();
    let mut sum_arr = Array2::<f64>::zeros((height, width));
    let mut count_arr = Array2::<u64>::zeros((height, width));

    for t in 0..t_len {
        let frame = array.slice(s![t, .., ..]);
        Zip::from(&mut sum_arr)
            .and(&mut count_arr)
            .and(&frame)
            .par_for_each(|s, c, &v| {
                if skip_na {
                    if !v.is_nan() {
                        *s += v;
                        *c += 1;
                    }
                } else {
                    *s += v;
                    *c += 1;
                }
            });
    }

    let mut result = Array2::<f64>::zeros((height, width));
    Zip::from(&mut result)
        .and(&sum_arr)
        .and(&count_arr)
        .par_for_each(|r, &s, &c| {
            if skip_na {
                *r = if c == 0 { f64::NAN } else { s / c as f64 };
            } else {
                *r = s / c as f64;
            }
        });

    result.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (arr, skip_na=true))]
pub fn temporal_sum(py: Python<'_>, arr: &PyAny, skip_na: bool) -> PyResult<PyObject> {
    if let Ok(arr1d) = arr.downcast::<numpy::PyArray1<f64>>() {
        Ok(temporal_sum_1d(arr1d.readonly(), skip_na).into_py(py))
    } else if let Ok(arr2d) = arr.downcast::<numpy::PyArray2<f64>>() {
        Ok(temporal_sum_2d(py, arr2d.readonly(), skip_na).into_py(py))
    } else if let Ok(arr3d) = arr.downcast::<numpy::PyArray3<f64>>() {
        Ok(temporal_sum_3d(py, arr3d.readonly(), skip_na).into_py(py))
    } else if let Ok(arr4d) = arr.downcast::<numpy::PyArray4<f64>>() {
        Ok(temporal_sum_4d(py, arr4d.readonly(), skip_na).into_py(py))
    } else {
        Err(
            CoreError::InvalidArgument("Expected a 1D, 2D, 3D, or 4D NumPy array.".to_string())
                .into(),
        )
    }
}

fn temporal_sum_1d(arr: PyReadonlyArray1<f64>, skip_na: bool) -> f64 {
    let array = arr.as_array();
    if skip_na {
        array.iter().filter(|v| !v.is_nan()).sum()
    } else if array.iter().any(|v| v.is_nan()) {
        f64::NAN
    } else {
        array.sum()
    }
}

fn temporal_sum_2d<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<f64>,
    skip_na: bool,
) -> &'py PyArray1<f64> {
    let array = arr.as_array();
    let shape = array.shape();
    let num_bands = shape[1];
    let mut result = Array1::<f64>::zeros(num_bands);

    for i in 0..num_bands {
        let series = array.column(i);
        if skip_na {
            result[i] = series.iter().filter(|v| !v.is_nan()).sum();
        } else if series.iter().any(|v| v.is_nan()) {
            result[i] = f64::NAN;
        } else {
            result[i] = series.sum();
        }
    }
    result.into_pyarray(py)
}

fn temporal_sum_3d<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray3<f64>,
    skip_na: bool,
) -> &'py PyArray2<f64> {
    let array = arr.as_array();
    let shape = array.shape();
    let (height, width) = (shape[1], shape[2]);
    let mut result = Array2::<f64>::zeros((height, width));

    result
        .indexed_iter_mut()
        .par_bridge()
        .for_each(|((r, c), pixel)| {
            let series = array.slice(s![.., r, c]);
            if skip_na {
                *pixel = series.iter().filter(|v| !v.is_nan()).sum();
            } else if series.iter().any(|v| v.is_nan()) {
                *pixel = f64::NAN;
            } else {
                *pixel = series.sum();
            }
        });

    result.into_pyarray(py)
}

fn temporal_sum_4d<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray4<f64>,
    skip_na: bool,
) -> &'py PyArray3<f64> {
    let array = arr.as_array();
    let shape = array.shape();
    let (num_bands, height, width) = (shape[1], shape[2], shape[3]);
    let mut result = Array3::<f64>::zeros((num_bands, height, width));

    result
        .indexed_iter_mut()
        .par_bridge()
        .for_each(|((b, r, c), pixel)| {
            let series = array.slice(s![.., b, r, c]);
            if skip_na {
                *pixel = series.iter().filter(|v| !v.is_nan()).sum();
            } else if series.iter().any(|v| v.is_nan()) {
                *pixel = f64::NAN;
            } else {
                *pixel = series.sum();
            }
        });

    result.into_pyarray(py)
}

fn temporal_mean_4d<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray4<f64>,
    skip_na: bool,
) -> &'py PyArray3<f64> {
    let array = arr.as_array();
    let (t_len, num_bands, height, width) = array.dim();
    let mut sum_arr = Array3::<f64>::zeros((num_bands, height, width));
    let mut count_arr = Array3::<u64>::zeros((num_bands, height, width));

    for t in 0..t_len {
        let frame = array.slice(s![t, .., .., ..]);
        Zip::from(&mut sum_arr)
            .and(&mut count_arr)
            .and(&frame)
            .par_for_each(|s, c, &v| {
                if skip_na {
                    if !v.is_nan() {
                        *s += v;
                        *c += 1;
                    }
                } else {
                    *s += v;
                    *c += 1;
                }
            });
    }

    let mut result = Array3::<f64>::zeros((num_bands, height, width));
    Zip::from(&mut result)
        .and(&sum_arr)
        .and(&count_arr)
        .par_for_each(|r, &s, &c| {
            if skip_na {
                *r = if c == 0 { f64::NAN } else { s / c as f64 };
            } else {
                *r = s / c as f64;
            }
        });

    result.into_pyarray(py)
}

fn temporal_std_1d(arr: PyReadonlyArray1<f64>, skip_na: bool) -> f64 {
    let array = arr.as_array();
    if skip_na {
        let (sum, sum_sq, count) =
            array
                .iter()
                .fold((0.0, 0.0, 0), |(acc_s, acc_sq, acc_c), &v| {
                    if v.is_nan() {
                        (acc_s, acc_sq, acc_c)
                    } else {
                        (acc_s + v, acc_sq + v * v, acc_c + 1)
                    }
                });
        if count < 2 {
            let nan = f64::NAN;
            return nan;
        }
        let mean = sum / count as f64;
        let variance = (sum_sq - sum * mean) / (count - 1) as f64;
        // Ensure non-negative before sqrt (floating point errors)
        if variance < 0.0 {
            0.0
        } else {
            variance.sqrt()
        }
    } else if array.iter().any(|v| v.is_nan()) {
        f64::NAN
    } else {
        array.std(1.0)
    }
}

fn temporal_std_2d<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<f64>,
    skip_na: bool,
) -> &'py PyArray1<f64> {
    let array = arr.as_array();
    let (t_len, num_bands) = array.dim();
    let mut sum_arr = Array1::<f64>::zeros(num_bands);
    let mut count_arr = Array1::<u64>::zeros(num_bands);

    // Pass 1: Mean
    for t in 0..t_len {
        let frame = array.slice(s![t, ..]);
        Zip::from(&mut sum_arr)
            .and(&mut count_arr)
            .and(&frame)
            .for_each(|s, c, &v| {
                if skip_na {
                    if !v.is_nan() {
                        *s += v;
                        *c += 1;
                    }
                } else {
                    *s += v;
                    *c += 1;
                }
            });
    }

    let mut mean_arr = Array1::<f64>::zeros(num_bands);
    Zip::from(&mut mean_arr)
        .and(&sum_arr)
        .and(&count_arr)
        .for_each(|m, &s, &c| {
            *m = if c == 0 { f64::NAN } else { s / c as f64 };
        });

    // Pass 2: Sum of squared differences
    let mut sum_sq_diff = Array1::<f64>::zeros(num_bands);
    for t in 0..t_len {
        let frame = array.slice(s![t, ..]);
        Zip::from(&mut sum_sq_diff)
            .and(&mean_arr)
            .and(&frame)
            .for_each(|sq, &m, &v| {
                if skip_na {
                    if !v.is_nan() && !m.is_nan() {
                        let diff = v - m;
                        *sq += diff * diff;
                    }
                } else {
                    let diff = v - m;
                    *sq += diff * diff;
                }
            });
    }

    let mut result = Array1::<f64>::zeros(num_bands);
    Zip::from(&mut result)
        .and(&sum_sq_diff)
        .and(&count_arr)
        .for_each(|r, &sq, &c| {
            if c < 2 {
                *r = f64::NAN;
            } else {
                let variance = sq / (c - 1) as f64;
                *r = if variance < 0.0 { 0.0 } else { variance.sqrt() };
            }
        });

    result.into_pyarray(py)
}

fn temporal_std_3d<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray3<f64>,
    skip_na: bool,
) -> &'py PyArray2<f64> {
    let array = arr.as_array();
    let (t_len, height, width) = array.dim();
    let mut sum_arr = Array2::<f64>::zeros((height, width));
    let mut count_arr = Array2::<u64>::zeros((height, width));

    // Pass 1: Mean
    for t in 0..t_len {
        let frame = array.slice(s![t, .., ..]);
        Zip::from(&mut sum_arr)
            .and(&mut count_arr)
            .and(&frame)
            .par_for_each(|s, c, &v| {
                if skip_na {
                    if !v.is_nan() {
                        *s += v;
                        *c += 1;
                    }
                } else {
                    *s += v;
                    *c += 1;
                }
            });
    }

    let mut mean_arr = Array2::<f64>::zeros((height, width));
    Zip::from(&mut mean_arr)
        .and(&sum_arr)
        .and(&count_arr)
        .par_for_each(|m, &s, &c| {
            *m = if c == 0 { f64::NAN } else { s / c as f64 };
        });

    // Pass 2: Sum of squared differences
    let mut sum_sq_diff = Array2::<f64>::zeros((height, width));
    for t in 0..t_len {
        let frame = array.slice(s![t, .., ..]);
        Zip::from(&mut sum_sq_diff)
            .and(&mean_arr)
            .and(&frame)
            .par_for_each(|sq, &m, &v| {
                if skip_na {
                    if !v.is_nan() && !m.is_nan() {
                        let diff = v - m;
                        *sq += diff * diff;
                    }
                } else {
                    let diff = v - m;
                    *sq += diff * diff;
                }
            });
    }

    let mut result = Array2::<f64>::zeros((height, width));
    Zip::from(&mut result)
        .and(&sum_sq_diff)
        .and(&count_arr)
        .par_for_each(|r, &sq, &c| {
            if c < 2 {
                *r = f64::NAN;
            } else {
                let variance = sq / (c - 1) as f64;
                *r = if variance < 0.0 { 0.0 } else { variance.sqrt() };
            }
        });

    result.into_pyarray(py)
}

fn temporal_std_4d<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray4<f64>,
    skip_na: bool,
) -> &'py PyArray3<f64> {
    let array = arr.as_array();
    let shape = array.shape();
    let (num_bands, height, width) = (shape[1], shape[2], shape[3]);
    let mut result = Array3::<f64>::zeros((num_bands, height, width));

    result
        .indexed_iter_mut()
        .par_bridge()
        .for_each(|((b, r, c), pixel)| {
            let series = array.slice(s![.., b, r, c]);
            if skip_na {
                let (sum, sum_sq, count) =
                    series
                        .iter()
                        .fold((0.0, 0.0, 0), |(acc_s, acc_sq, acc_c), &v| {
                            if v.is_nan() {
                                (acc_s, acc_sq, acc_c)
                            } else {
                                (acc_s + v, acc_sq + v * v, acc_c + 1)
                            }
                        });
                *pixel = if count < 2 {
                    f64::NAN
                } else {
                    let mean = sum / count as f64;
                    let variance = (sum_sq - sum * mean) / (count - 1) as f64;
                    if variance < 0.0 {
                        0.0
                    } else {
                        variance.sqrt()
                    }
                };
            } else {
                *pixel = if series.iter().any(|v| v.is_nan()) {
                    f64::NAN
                } else {
                    series.std(1.0)
                };
            }
        });

    result.into_pyarray(py)
}
