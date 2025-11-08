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
        if a_1d.shape() != b_1d.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch for 1D arrays: a {:?} vs b {:?}",
                a_1d.shape(),
                b_1d.shape()
            )));
        }
        normalized_difference_1d(py, a_1d, b_1d).map(|res| res.into_py(py))
    } else if let Ok(a_2d) = a.extract::<PyReadonlyArray2<f64>>() {
        let b_2d = b.extract::<PyReadonlyArray2<f64>>()?;
        if a_2d.shape() != b_2d.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch for 2D arrays: a {:?} vs b {:?}",
                a_2d.shape(),
                b_2d.shape()
            )));
        }
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
/// Thin wrapper around `normalized_difference`.
///
/// NDVI = (NIR - Red) / (NIR + Red)
///
/// This is a convenience wrapper around `normalized_difference` for both 1D and 2D arrays.
/// It will dispatch based on the dimensionality of the provided numpy arrays.
///
/// # Arguments
/// * `nir` - Near-infrared band values (1D or 2D float64 numpy array)
/// * `red` - Red band values (same shape and type as `nir`)
///
/// # Returns
/// NDVI values ranging from -1 to 1 with the same shape as inputs
///
/// # Example (1D)
/// ```python
/// import numpy as np
/// from eo_processor import ndvi
///
/// nir = np.array([0.8, 0.7, 0.6])
/// red = np.array([0.2, 0.1, 0.3])
/// ndvi_vals = ndvi(nir, red)
/// # Expected: [0.6, 0.75, (0.3/0.9)]
/// print(ndvi_vals)
/// ```
///
/// # Example (2D)
/// ```python
/// import numpy as np
/// from eo_processor import ndvi
///
/// nir = np.array([[0.8, 0.7],
///                 [0.6, 0.5]])
/// red = np.array([[0.2, 0.1],
///                 [0.3, 0.5]])
/// ndvi_vals = ndvi(nir, red)
/// print(ndvi_vals.shape)  # (2, 2)
/// ```
#[pyfunction]
pub fn ndvi(py: Python<'_>, nir: &PyAny, red: &PyAny) -> PyResult<PyObject> {
    normalized_difference(py, nir, red)
}

/// Compute NDWI (Normalized Difference Water Index) from Green and NIR bands.
///
/// Thin wrapper around `normalized_difference`.
///
/// NDWI = (Green - NIR) / (Green + NIR)
///
/// Dispatches for 1D and 2D arrays automatically.
///
/// # Arguments
/// * `green` - Green band values (1D or 2D float64 numpy array)
/// * `nir` - Near-infrared band values (same shape and type as `green`)
///
/// # Returns
/// NDWI values ranging from -1 to 1 with the same shape as inputs
///
/// # Example (1D)
/// ```python
/// import numpy as np
/// from eo_processor import ndwi
///
/// green = np.array([0.4, 0.5, 0.6])
/// nir   = np.array([0.2, 0.1, 0.3])
/// ndwi_vals = ndwi(green, nir)
/// print(ndwi_vals)
/// ```
///
/// # Example (2D)
/// ```python
/// import numpy as np
/// from eo_processor import ndwi
///
/// green = np.array([[0.4, 0.5],
///                   [0.6, 0.7]])
/// nir   = np.array([[0.2, 0.1],
///                   [0.3, 0.4]])
/// ndwi_vals = ndwi(green, nir)
/// print(ndwi_vals.shape)  # (2, 2)
/// ```
#[pyfunction]
pub fn ndwi(py: Python<'_>, green: &PyAny, nir: &PyAny) -> PyResult<PyObject> {
    normalized_difference(py, green, nir)
}

/// Compute Enhanced Vegetation Index (EVI).
///
/// Formula:
/// EVI = G * (NIR - Red) / (NIR + C1 * Red - C2 * Blue + L)
///
/// Constants (MODIS standard):
/// G = 2.5, C1 = 6.0, C2 = 7.5, L = 1.0
///
/// Automatically dispatches for 1D or 2D float64 numpy arrays.
///
/// # Arguments
/// * `nir`  - Near-infrared band values
/// * `red`  - Red band values
/// * `blue` - Blue band values
///
/// All three inputs must be the same shape & type (1D or 2D float64).
///
/// # Returns
/// EVI values (same shape as input).
///
/// # Example (1D)
/// ```python
/// import numpy as np
/// from eo_processor import enhanced_vegetation_index as evi
///
/// nir  = np.array([0.6, 0.7])
/// red  = np.array([0.3, 0.2])
/// blue = np.array([0.1, 0.05])
/// evi_vals = evi(nir, red, blue)
/// print(evi_vals)
/// ```
///
/// # Example (2D)
/// ```python
/// import numpy as np
/// from eo_processor import enhanced_vegetation_index as evi
///
/// nir  = np.array([[0.6, 0.7],
///                  [0.2, 0.3]])
/// red  = np.array([[0.3, 0.2],
///                  [0.1, 0.15]])
/// blue = np.array([[0.1, 0.05],
///                  [0.02, 0.03]])
/// evi_vals = evi(nir, red, blue)
/// print(evi_vals.shape)  # (2, 2)
/// ```
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
        if nir_1d.shape() != red_1d.shape() || nir_1d.shape() != blue_1d.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch for 1D EVI inputs: nir {:?}, red {:?}, blue {:?}",
                nir_1d.shape(),
                red_1d.shape(),
                blue_1d.shape()
            )));
        }
        enhanced_vegetation_index_1d(py, nir_1d, red_1d, blue_1d).map(|res| res.into_py(py))
    } else if let Ok(nir_2d) = nir.extract::<PyReadonlyArray2<f64>>() {
        let red_2d = red.extract::<PyReadonlyArray2<f64>>()?;
        let blue_2d = blue.extract::<PyReadonlyArray2<f64>>()?;
        if nir_2d.shape() != red_2d.shape() || nir_2d.shape() != blue_2d.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch for 2D EVI inputs: nir {:?}, red {:?}, blue {:?}",
                nir_2d.shape(),
                red_2d.shape(),
                blue_2d.shape()
            )));
        }
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

    #[test]
    fn test_ndvi_wrapper_dispatch() {
        let nir = Array1::from_vec(vec![0.8, 0.7, 0.6]);
        let red = Array1::from_vec(vec![0.2, 0.1, 0.3]);
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let nir_py = nir.clone().into_pyarray(py);
            let red_py = red.clone().into_pyarray(py);
            let ndvi_obj = ndvi(py, nir_py, red_py).unwrap();
            let ndvi_arr: &PyArray1<f64> = ndvi_obj.extract(py).unwrap();
            let ndvi_read = ndvi_arr.readonly();
            let ndvi_vals = ndvi_read.as_array();
            assert_relative_eq!(ndvi_vals[0], 0.6, epsilon = 1e-10);
            assert_relative_eq!(ndvi_vals[1], 0.75, epsilon = 1e-10);
            assert_relative_eq!(ndvi_vals[2], 1.0 / 3.0, epsilon = 1e-10);
        });
    }

    #[test]
    fn test_ndwi_wrapper_dispatch() {
        let green = Array1::from_vec(vec![0.4, 0.5, 0.6]);
        let nir = Array1::from_vec(vec![0.2, 0.1, 0.3]);
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let green_py = green.clone().into_pyarray(py);
            let nir_py = nir.clone().into_pyarray(py);
            let ndwi_obj = ndwi(py, green_py, nir_py).unwrap();
            let ndwi_arr: &PyArray1<f64> = ndwi_obj.extract(py).unwrap();
            let ndwi_read = ndwi_arr.readonly();
            let ndwi_vals = ndwi_read.as_array();
            // (0.4-0.2)/(0.4+0.2)=0.2/0.6=0.333...
            assert_relative_eq!(ndwi_vals[0], (0.4 - 0.2) / (0.4 + 0.2), epsilon = 1e-10);
            // (0.5-0.1)/(0.5+0.1)=0.4/0.6=0.666...
            assert_relative_eq!(ndwi_vals[1], (0.5 - 0.1) / (0.5 + 0.1), epsilon = 1e-10);
            // (0.6-0.3)/(0.6+0.3)=0.3/0.9=0.333...
            assert_relative_eq!(ndwi_vals[2], (0.6 - 0.3) / (0.6 + 0.3), epsilon = 1e-10);
        });
    }

    #[test]
    fn test_evi_1d() {
        // Small synthetic example
        let nir = Array1::from_vec(vec![0.6, 0.7]);
        let red = Array1::from_vec(vec![0.3, 0.2]);
        let blue = Array1::from_vec(vec![0.1, 0.05]);
        const G: f64 = 2.5;
        const C1: f64 = 6.0;
        const C2: f64 = 7.5;
        const L: f64 = 1.0;
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let nir_py = nir.clone().into_pyarray(py);
            let red_py = red.clone().into_pyarray(py);
            let blue_py = blue.clone().into_pyarray(py);
            let evi_obj = enhanced_vegetation_index(py, nir_py, red_py, blue_py).unwrap();
            let evi_arr: &PyArray1<f64> = evi_obj.extract(py).unwrap();
            let evi_read = evi_arr.readonly();
            let evi_vals = evi_read.as_array();
            let expected0 = G * (0.6 - 0.3) / (0.6 + C1 * 0.3 - C2 * 0.1 + L);
            let expected1 = G * (0.7 - 0.2) / (0.7 + C1 * 0.2 - C2 * 0.05 + L);
            assert_relative_eq!(evi_vals[0], expected0, epsilon = 1e-12);
            assert_relative_eq!(evi_vals[1], expected1, epsilon = 1e-12);
        });
    }

    #[test]
    fn test_evi_2d() {
        let nir = Array2::from_shape_vec((2, 2), vec![0.6, 0.7, 0.2, 0.3]).unwrap();
        let red = Array2::from_shape_vec((2, 2), vec![0.3, 0.2, 0.1, 0.15]).unwrap();
        let blue = Array2::from_shape_vec((2, 2), vec![0.1, 0.05, 0.02, 0.03]).unwrap();
        const G: f64 = 2.5;
        const C1: f64 = 6.0;
        const C2: f64 = 7.5;
        const L: f64 = 1.0;
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let nir_py = nir.clone().into_pyarray(py);
            let red_py = red.clone().into_pyarray(py);
            let blue_py = blue.clone().into_pyarray(py);
            let evi_obj = enhanced_vegetation_index(py, nir_py, red_py, blue_py).unwrap();
            let evi_arr: &PyArray2<f64> = evi_obj.extract(py).unwrap();
            let evi_read = evi_arr.readonly();
            let evi_vals = evi_read.as_array();
            for i in 0..2 {
                for j in 0..2 {
                    let nir_v = nir[[i, j]];
                    let red_v = red[[i, j]];
                    let blue_v = blue[[i, j]];
                    let expected = G * (nir_v - red_v) / (nir_v + C1 * red_v - C2 * blue_v + L);
                    assert_relative_eq!(evi_vals[[i, j]], expected, epsilon = 1e-12);
                }
            }
        });
    }
}
