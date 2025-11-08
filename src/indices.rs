use ndarray::{Array1, Array2, Zip};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Threshold for detecting near-zero values to avoid division by zero
const EPSILON: f64 = 1e-10;

/// Public function to compute normalized difference between two arrays.
/// Acts as a wrapper to expose both 1D and 2D versions.
#[pyfunction]
pub fn normalized_difference(py: Python<'_>, a: &PyAny, b: &PyAny) -> PyResult<PyObject> {
    if let Ok(a_1d) = a.extract::<PyReadonlyArray1<f64>>() {
        let b_1d = b.extract::<PyReadonlyArray1<f64>>()?;
        normalized_difference_1d(py, a_1d, b_1d).map(|res| res.into_py(py))
    } else if let Ok(a_2d) = a.extract::<PyReadonlyArray2<f64>>() {
        let b_2d = b.extract::<PyReadonlyArray2<f64>>()?;
        normalized_difference_2d(py, a_2d, b_2d).map(|res| res.into_py(py))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input arrays must be either 1D or 2D numpy arrays of type float64.",
        ))
    }
}

/// Compute normalized difference between two arrays.
///
/// This function computes (a - b) / (a + b) element-wise, handling division by zero
/// by returning 0.0 when the denominator is zero.
///
/// # Arguments
/// * `a` - First input array (e.g., NIR band for NDVI)
/// * `b` - Second input array (e.g., Red band for NDVI)
///
/// # Returns
/// Array with the same shape as inputs containing the normalized difference values
///
/// # Example (from Python)
/// ```python
/// import numpy as np
/// from eo_processor import normalized_difference
///
/// nir = np.array([0.8, 0.7, 0.6])
/// red = np.array([0.2, 0.1, 0.3])
/// ndvi = normalized_difference(nir, red)
/// ```
fn normalized_difference_1d<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>,
) -> PyResult<&'py PyArray1<f64>> {
    let a = a.as_array();
    let b = b.as_array();

    let mut result = Array1::<f64>::zeros(a.len());

    Zip::from(&mut result)
        .and(&a)
        .and(&b)
        .for_each(|r, &a_val, &b_val| {
            let sum = a_val + b_val;
            *r = if sum.abs() < EPSILON {
                0.0
            } else {
                (a_val - b_val) / sum
            };
        });

    Ok(result.into_pyarray(py))
}

/// Compute normalized difference between two 2D arrays.
///
/// This function computes (a - b) / (a + b) element-wise for 2D arrays,
/// handling division by zero by returning 0.0 when the denominator is zero.
///
/// # Arguments
/// * `a` - First input 2D array (e.g., NIR band for NDVI)
/// * `b` - Second input 2D array (e.g., Red band for NDVI)
///
/// # Returns
/// 2D array with the same shape as inputs containing the normalized difference values
///
/// # Example (from Python)
/// ```python
/// import numpy as np
/// from eo_processor import normalized_difference_2d
///
/// nir = np.random.rand(100, 100)
/// red = np.random.rand(100, 100)
/// ndvi = normalized_difference_2d(nir, red)
/// ```
fn normalized_difference_2d<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<f64>,
    b: PyReadonlyArray2<f64>,
) -> PyResult<&'py PyArray2<f64>> {
    let a = a.as_array();
    let b = b.as_array();

    let shape = a.dim();
    let mut result = Array2::<f64>::zeros(shape);

    Zip::from(&mut result)
        .and(&a)
        .and(&b)
        .for_each(|r, &a_val, &b_val| {
            let sum = a_val + b_val;
            *r = if sum.abs() < EPSILON {
                0.0
            } else {
                (a_val - b_val) / sum
            };
        });

    Ok(result.into_pyarray(py))
}

/// Compute NDVI (Normalized Difference Vegetation Index) from NIR and Red bands.
///
/// Thin wrapper around normalized_difference.
///
/// NDVI = (NIR - Red) / (NIR + Red)
///
/// This is a convenience wrapper around normalized_difference for 1D arrays.
///
/// # Arguments
/// * `nir` - Near-infrared band values
/// * `red` - Red band values
///
/// # Returns
/// NDVI values ranging from -1 to 1
#[pyfunction]
pub fn ndvi(py: Python<'_>, nir: &PyAny, red: &PyAny) -> PyResult<PyObject> {
    normalized_difference(py, nir, red)
}

/// Compute NDWI (Normalized Difference Water Index) from Green and NIR bands.
///
/// Thin wrapper around normalized_difference.
///
/// NDWI = (Green - NIR) / (Green + NIR)
///
/// # Arguments
/// * `green` - Green band values
/// * `nir` - Near-infrared band values
///
/// # Returns
/// NDWI values ranging from -1 to 1
#[pyfunction]
pub fn ndwi(py: Python<'_>, green: &PyAny, nir: &PyAny) -> PyResult<PyObject> {
    normalized_difference(py, green, nir)
}

#[pyfunction]
pub fn enhanced_vegetation_index(
    py: Python<'_>,
    nir: &PyAny,
    red: &PyAny,
    blue: &PyAny,
) -> PyResult<PyObject> {
    if let Ok(nir_1d) = nir.extract::<PyReadonlyArray1<f64>>() {
        let red_1d = red.extract::<PyReadonlyArray1<f64>>()?;
        let blue_1d = blue.extract::<PyReadonlyArray1<f64>>()?;
        enhanced_vegetation_index_1d(py, nir_1d, red_1d, blue_1d).map(|res| res.into_py(py))
    } else if let Ok(nir_2d) = nir.extract::<PyReadonlyArray2<f64>>() {
        let red_2d = red.extract::<PyReadonlyArray2<f64>>()?;
        let blue_2d = blue.extract::<PyReadonlyArray2<f64>>()?;
        enhanced_vegetation_index_2d(py, nir_2d, red_2d, blue_2d).map(|res| res.into_py(py))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input arrays must be either 1D or 2D numpy arrays of type float64.",
        ))
    }
}

// 1D Enhanced Vegetation Index helper
fn enhanced_vegetation_index_1d<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray1<f64>,
    red: PyReadonlyArray1<f64>,
    blue: PyReadonlyArray1<f64>,
) -> PyResult<&'py PyArray1<f64>> {
    const G: f64 = 2.5;
    const C1: f64 = 6.0;
    const C2: f64 = 7.5;
    const L: f64 = 1.0;

    let nir = nir.as_array();
    let red = red.as_array();
    let blue = blue.as_array();

    let mut result = Array1::<f64>::zeros(nir.len());

    Zip::from(&mut result)
        .and(&nir)
        .and(&red)
        .and(&blue)
        .for_each(|r, &nir_v, &red_v, &blue_v| {
            let denom = nir_v + C1 * red_v - C2 * blue_v + L;
            *r = if denom.abs() < EPSILON {
                0.0
            } else {
                G * (nir_v - red_v) / denom
            };
        });

    Ok(result.into_pyarray(py))
}

// 2D Enhanced Vegetation Index helper
fn enhanced_vegetation_index_2d<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray2<f64>,
    red: PyReadonlyArray2<f64>,
    blue: PyReadonlyArray2<f64>,
) -> PyResult<&'py PyArray2<f64>> {
    const G: f64 = 2.5;
    const C1: f64 = 6.0;
    const C2: f64 = 7.5;
    const L: f64 = 1.0;

    let nir = nir.as_array();
    let red = red.as_array();
    let blue = blue.as_array();

    let shape = nir.dim();
    let mut result = Array2::<f64>::zeros(shape);

    Zip::from(&mut result)
        .and(&nir)
        .and(&red)
        .and(&blue)
        .for_each(|r, &nir_v, &red_v, &blue_v| {
            let denom = nir_v + C1 * red_v - C2 * blue_v + L;
            *r = if denom.abs() < EPSILON {
                0.0
            } else {
                G * (nir_v - red_v) / denom
            };
        });

    Ok(result.into_pyarray(py))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_normalized_difference_basic() {
        let a = Array1::from_vec(vec![0.8, 0.7, 0.6]);
        let b = Array1::from_vec(vec![0.2, 0.1, 0.3]);

        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let a_py = a.clone().into_pyarray(py);
            let b_py = b.clone().into_pyarray(py);

            let result = normalized_difference_1d(py, a_py.readonly(), b_py.readonly()).unwrap();

            let result_readonly = result.readonly();
            let result_array = result_readonly.as_array();

            // Expected: (0.8-0.2)/(0.8+0.2) = 0.6/1.0 = 0.6
            assert_relative_eq!(result_array[0], 0.6, epsilon = 1e-10);
            // Expected: (0.7-0.1)/(0.7+0.1) = 0.6/0.8 = 0.75
            assert_relative_eq!(result_array[1], 0.75, epsilon = 1e-10);
            // Expected: (0.6-0.3)/(0.6+0.3) = 0.3/0.9 = 1/3
            assert_relative_eq!(result_array[2], 1.0 / 3.0, epsilon = 1e-10);
        });
    }

    #[test]
    fn test_normalized_difference_zero_sum() {
        let a = Array1::from_vec(vec![0.0, 0.5, 0.0]);
        let b = Array1::from_vec(vec![0.0, -0.5, 0.0]);

        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let a_py = a.clone().into_pyarray(py);
            let b_py = b.clone().into_pyarray(py);

            let result = normalized_difference_1d(py, a_py.readonly(), b_py.readonly()).unwrap();

            let result_readonly = result.readonly();
            let result_array = result_readonly.as_array();

            // When sum is zero, should return 0.0
            assert_relative_eq!(result_array[0], 0.0, epsilon = 1e-10);
            // When sum is not zero: (0.5 - (-0.5)) / (0.5 + (-0.5)) = 1.0 / 0.0 -> undefined, but close to 0
            // Actually, this will be 0.0 because sum is 0.0
            assert_relative_eq!(result_array[1], 0.0, epsilon = 1e-10);
            assert_relative_eq!(result_array[2], 0.0, epsilon = 1e-10);
        });
    }

    #[test]
    fn test_normalized_difference_2d() {
        let a = Array2::from_shape_vec((2, 2), vec![0.8, 0.7, 0.6, 0.5]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![0.2, 0.1, 0.3, 0.5]).unwrap();

        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let a_py = a.clone().into_pyarray(py);
            let b_py = b.clone().into_pyarray(py);

            let result = normalized_difference_2d(py, a_py.readonly(), b_py.readonly()).unwrap();

            let result_readonly = result.readonly();
            let result_array = result_readonly.as_array();

            assert_relative_eq!(result_array[[0, 0]], 0.6, epsilon = 1e-10);
            assert_relative_eq!(result_array[[0, 1]], 0.75, epsilon = 1e-10);
            // (0.6 - 0.3) / (0.6 + 0.3) = 0.3 / 0.9 = 1/3
            assert_relative_eq!(result_array[[1, 0]], 1.0 / 3.0, epsilon = 1e-10);
            // (0.5 - 0.5) / (0.5 + 0.5) = 0.0 / 1.0 = 0.0
            assert_relative_eq!(result_array[[1, 1]], 0.0, epsilon = 1e-10);
        });
    }
}
