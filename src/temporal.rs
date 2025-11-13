use ndarray::{s, Array1, Array2, Array3};
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
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected a 1D, 2D, 3D, or 4D NumPy array.",
        ))
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
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected a 1D, 2D, 3D, or 4D NumPy array.",
        ))
    }
}

fn temporal_mean_1d(arr: PyReadonlyArray1<f64>, skip_na: bool) -> f64 {
    let mut series: Vec<f64> = arr.as_array().to_vec();
    if skip_na {
        series.retain(|v| !v.is_nan());
    }
    if series.is_empty() {
        f64::NAN
    } else {
        series.iter().sum::<f64>() / series.len() as f64
    }
}

fn temporal_mean_2d<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<f64>,
    skip_na: bool,
) -> &'py PyArray1<f64> {
    let array = arr.as_array();
    let shape = array.shape();
    let num_bands = shape[1];
    let mut result = Array1::<f64>::zeros(num_bands);

    for i in 0..num_bands {
        let mut series: Vec<f64> = array.column(i).to_vec();
        if skip_na {
            series.retain(|v| !v.is_nan());
        }
        if series.is_empty() {
            result[i] = f64::NAN;
        } else {
            result[i] = series.iter().sum::<f64>() / series.len() as f64;
        }
    }
    result.into_pyarray(py)
}

fn temporal_mean_3d<'py>(
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
            let mut series: Vec<f64> = array.slice(s![.., r, c]).to_vec();
            if skip_na {
                series.retain(|v| !v.is_nan());
            }
            if series.is_empty() {
                *pixel = f64::NAN;
            } else {
                *pixel = series.iter().sum::<f64>() / series.len() as f64;
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
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected a 1D, 2D, 3D, or 4D NumPy array.",
        ))
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
    let shape = array.shape();
    let (num_bands, height, width) = (shape[1], shape[2], shape[3]);
    let mut result = Array3::<f64>::zeros((num_bands, height, width));

    result
        .indexed_iter_mut()
        .par_bridge()
        .for_each(|((b, r, c), pixel)| {
            let mut series: Vec<f64> = array.slice(s![.., b, r, c]).to_vec();
            if skip_na {
                series.retain(|v| !v.is_nan());
            }
            if series.is_empty() {
                *pixel = f64::NAN;
            } else {
                *pixel = series.iter().sum::<f64>() / series.len() as f64;
            }
        });

    result.into_pyarray(py)
}

fn temporal_std_1d(arr: PyReadonlyArray1<f64>, skip_na: bool) -> f64 {
    let mut series: Vec<f64> = arr.as_array().to_vec();
    if skip_na {
        series.retain(|v| !v.is_nan());
    }
    if series.len() < 2 {
        return f64::NAN;
    }
    let mean = series.iter().sum::<f64>() / series.len() as f64;
    let variance =
        series.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (series.len() - 1) as f64;
    variance.sqrt()
}

fn temporal_std_2d<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<f64>,
    skip_na: bool,
) -> &'py PyArray1<f64> {
    let array = arr.as_array();
    let shape = array.shape();
    let num_bands = shape[1];
    let mut result = Array1::<f64>::zeros(num_bands);

    for i in 0..num_bands {
        let mut series: Vec<f64> = array.column(i).to_vec();
        if skip_na {
            series.retain(|v| !v.is_nan());
        }
        if series.len() < 2 {
            result[i] = f64::NAN;
            continue;
        }
        let mean = series.iter().sum::<f64>() / series.len() as f64;
        let variance =
            series.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (series.len() - 1) as f64;
        result[i] = variance.sqrt();
    }

    result.into_pyarray(py)
}

fn temporal_std_3d<'py>(
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
            let mut series: Vec<f64> = array.slice(s![.., r, c]).to_vec();
            if skip_na {
                series.retain(|v| !v.is_nan());
            }
            if series.len() < 2 {
                *pixel = f64::NAN;
                return;
            }
            let mean = series.iter().sum::<f64>() / series.len() as f64;
            let variance =
                series.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (series.len() - 1) as f64;
            *pixel = variance.sqrt();
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
            let mut series: Vec<f64> = array.slice(s![.., b, r, c]).to_vec();
            if skip_na {
                series.retain(|v| !v.is_nan());
            }
            if series.len() < 2 {
                *pixel = f64::NAN;
                return;
            }
            let mean = series.iter().sum::<f64>() / series.len() as f64;
            let variance =
                series.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (series.len() - 1) as f64;
            *pixel = variance.sqrt();
        });

    result.into_pyarray(py)
}
