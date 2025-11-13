use ndarray::{Array1, Array2, Zip};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArray4,
};
use pyo3::prelude::*;

/// Threshold for detecting near-zero values to avoid division by zero
const EPSILON: f64 = 1e-10;

/// Coerce a Python object to a readonly 1D float64 NumPy array.
/// Tries direct extraction; on failure, attempts `.astype("float64")`.
fn try_coerce_array1<'py>(obj: &'py PyAny) -> PyResult<PyReadonlyArray1<'py, f64>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<f64>>() {
        return Ok(arr);
    }
    let coerced = obj.call_method1("astype", ("float64",))?;
    let coerced_arr = coerced.extract::<PyReadonlyArray1<f64>>()?;
    Ok(coerced_arr)
}

/// Coerce a Python object to a readonly 2D float64 NumPy array.
/// Tries direct extraction; on failure, attempts `.astype("float64")`.
fn try_coerce_array2<'py>(obj: &'py PyAny) -> PyResult<PyReadonlyArray2<'py, f64>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f64>>() {
        return Ok(arr);
    }
    let coerced = obj.call_method1("astype", ("float64",))?;
    let coerced_arr = coerced.extract::<PyReadonlyArray2<f64>>()?;
    Ok(coerced_arr)
}

/// Public function to compute normalized difference between two arrays.
/// Acts as a wrapper to expose both 1D and 2D versions.
#[pyfunction]
pub fn normalized_difference(py: Python<'_>, a: &PyAny, b: &PyAny) -> PyResult<PyObject> {
    // Attempt direct float64 extraction first; if that fails, coerce dtype to float64.
    if let Ok(a_1d) = try_coerce_array1(a) {
        let b_1d = try_coerce_array1(b)?;
        if a_1d.shape() != b_1d.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch for 1D arrays: a {:?} vs b {:?}",
                a_1d.shape(),
                b_1d.shape()
            )));
        }
        normalized_difference_1d(py, a_1d, b_1d).map(|res| res.into_py(py))
    } else if let Ok(a_2d) = try_coerce_array2(a) {
        let b_2d = try_coerce_array2(b)?;
        if a_2d.shape() != b_2d.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch for 2D arrays: a {:?} vs b {:?}",
                a_2d.shape(),
                b_2d.shape()
            )));
        }
        normalized_difference_2d(py, a_2d, b_2d).map(|res| res.into_py(py))
    } else if let Ok(a_3d) = a.extract::<PyReadonlyArray3<f64>>() {
        let b_3d = b.extract::<PyReadonlyArray3<f64>>()?;
        if a_3d.shape() != b_3d.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch for 3D arrays: a {:?} vs b {:?}",
                a_3d.shape(),
                b_3d.shape()
            )));
        }
        normalized_difference_3d(py, a_3d, b_3d).map(|res| res.into_py(py))
    } else if let Ok(a_4d) = a.extract::<PyReadonlyArray4<f64>>() {
        let b_4d = b.extract::<PyReadonlyArray4<f64>>()?;
        if a_4d.shape() != b_4d.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch for 4D arrays: a {:?} vs b {:?}",
                a_4d.shape(),
                b_4d.shape()
            )));
        }
        normalized_difference_4d(py, a_4d, b_4d).map(|res| res.into_py(py))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Inputs must be numeric 1D, 2D, 3D, or 4D numpy arrays (will be coerced to float64).",
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
    let a_arr = a.as_array();
    let b_arr = b.as_array();
    // Branchless division; zero denominators yield NaN or Inf per IEEE 754.
    let out = py.allow_threads(|| {
        let numerator = &a_arr - &b_arr;
        let denom = &a_arr + &b_arr;
        numerator / denom
    });
    Ok(out.into_pyarray(py))
}

/// Compute normalized difference between two 2D arrays.
///
/// This function computes (a - b) / (a + b) element-wise for 2D arrays.
/// Division by zero yields NaN or Inf per IEEE 754.
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
    let a_arr = a.as_array();
    let b_arr = b.as_array();
    let out = py.allow_threads(|| {
        let numerator = &a_arr - &b_arr;
        let denom = &a_arr + &b_arr;
        numerator / denom
    });
    Ok(out.into_pyarray(py))
}

/// Compute normalized difference between two 3D arrays.
///
/// This function computes (a - b) / (a + b) element-wise for 3D arrays.
/// Division by zero yields NaN or Inf per IEEE 754.
/// # Arguments
/// * `a` - First input 3D array (e.g., NIR band for NDVI)
/// * `b` - Second input 3D array (e.g., Red band for NDVI)
/// # Returns
/// 3D array with the same shape as inputs containing the normalized difference values
/// # Example (from Python)
/// ```python
/// import numpy as np
/// from eo_processor import normalized_difference_3d
///
/// nir = np.random.rand(10, 100, 100)
/// red = np.random.rand(10, 100, 100)
/// ndvi = normalized_difference_3d(nir, red)
/// ```
fn normalized_difference_3d<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<f64>,
    b: PyReadonlyArray3<f64>,
) -> PyResult<&'py PyArray3<f64>> {
    let a_arr = a.as_array();
    let b_arr = b.as_array();
    let out = py.allow_threads(|| {
        let numerator = &a_arr - &b_arr;
        let denom = &a_arr + &b_arr;
        numerator / denom
    });
    Ok(out.into_pyarray(py))
}

/// Compute normalized difference between two 4D arrays.
///
/// This function computes (a - b) / (a + b) element-wise for 4D arrays,
/// handling division by zero by returning 0.0 when the denominator is zero.
///
/// # Arguments
/// * `a` - First input 4D array (e.g., NIR band for NDVI)
/// * `b` - Second input 4D array (e.g., Red band for NDVI)
/// # Returns
/// 4D array with the same shape as inputs containing the normalized difference values
/// # Example (from Python)
/// ```python
/// import numpy as np
/// from eo_processor import normalized_difference_4d
///
/// nir = np.random.rand(5, 10, 100, 100)
/// red = np.random.rand(5, 10, 100, 100)
/// ndvi = normalized_difference_4d(nir, red)
/// ```
fn normalized_difference_4d<'py>(
    py: Python<'py>,
    a: PyReadonlyArray4<f64>,
    b: PyReadonlyArray4<f64>,
) -> PyResult<&'py PyArray4<f64>> {
    let a_arr = a.as_array();
    let b_arr = b.as_array();
    let out = py.allow_threads(|| {
        let numerator = &a_arr - &b_arr;
        let denom = &a_arr + &b_arr;
        numerator / denom
    });
    Ok(out.into_pyarray(py))
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

/// Delta NDVI (pre - post) for change detection.
/// Accepts any numeric dtype; coerces to float64.
#[pyfunction]
pub fn delta_ndvi(
    py: Python<'_>,
    pre_nir: &PyAny,
    pre_red: &PyAny,
    post_nir: &PyAny,
    post_red: &PyAny,
) -> PyResult<PyObject> {
    // Use normalized_difference for coercion & computation
    let pre = normalized_difference(py, pre_nir, pre_red)?;
    let post = normalized_difference(py, post_nir, post_red)?;
    // Attempt 1D extract first, else 2D
    if let Ok(pre1) = pre.extract::<&PyArray1<f64>>(py) {
        let post1 = post.extract::<&PyArray1<f64>>(py)?;
        if pre1.len() != post1.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Shape mismatch in delta_ndvi (1D)",
            ));
        }
        let pre_view = pre1.readonly();
        let post_view = post1.readonly();
        let a = pre_view.as_array();
        let b = post_view.as_array();
        let mut out = Array1::<f64>::zeros(a.len());
        Zip::from(&mut out)
            .and(a)
            .and(b)
            .for_each(|r, &p, &q| *r = p - q);
        Ok(out.into_pyarray(py).into_py(py))
    } else {
        let pre2 = pre.extract::<&PyArray2<f64>>(py)?;
        let post2 = post.extract::<&PyArray2<f64>>(py)?;
        if pre2.shape() != post2.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Shape mismatch in delta_ndvi (2D)",
            ));
        }
        let pre_view = pre2.readonly();
        let post_view = post2.readonly();
        let a = pre_view.as_array();
        let b = post_view.as_array();
        let shape = a.dim();
        let mut out = Array2::<f64>::zeros(shape);
        Zip::from(&mut out)
            .and(a)
            .and(b)
            .for_each(|r, &p, &q| *r = p - q);
        Ok(out.into_pyarray(py).into_py(py))
    }
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

// Normalized Burn Ratio (NBR)
// Formula: NBR = (NIR - SWIR2) / (NIR + SWIR2)
// Implemented as a thin wrapper around normalized_difference.
//
#[pyfunction]
pub fn nbr(py: Python<'_>, nir: &PyAny, swir2: &PyAny) -> PyResult<PyObject> {
    normalized_difference(py, nir, swir2)
}

/// Delta NBR (pre - post) for burn severity change detection.
#[pyfunction]
pub fn delta_nbr(
    py: Python<'_>,
    pre_nir: &PyAny,
    pre_swir2: &PyAny,
    post_nir: &PyAny,
    post_swir2: &PyAny,
) -> PyResult<PyObject> {
    let pre = normalized_difference(py, pre_nir, pre_swir2)?;
    let post = normalized_difference(py, post_nir, post_swir2)?;
    if let Ok(pre1) = pre.extract::<&PyArray1<f64>>(py) {
        let post1 = post.extract::<&PyArray1<f64>>(py)?;
        if pre1.len() != post1.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Shape mismatch in delta_nbr (1D)",
            ));
        }
        let pre_view = pre1.readonly();
        let post_view = post1.readonly();
        let a = pre_view.as_array();
        let b = post_view.as_array();
        let mut out = Array1::<f64>::zeros(a.len());
        Zip::from(&mut out)
            .and(a)
            .and(b)
            .for_each(|r, &p, &q| *r = p - q);
        Ok(out.into_pyarray(py).into_py(py))
    } else {
        let pre2 = pre.extract::<&PyArray2<f64>>(py)?;
        let post2 = post.extract::<&PyArray2<f64>>(py)?;
        if pre2.shape() != post2.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Shape mismatch in delta_nbr (2D)",
            ));
        }
        let pre_view = pre2.readonly();
        let post_view = post2.readonly();
        let a = pre_view.as_array();
        let b = post_view.as_array();
        let shape = a.dim();
        let mut out = Array2::<f64>::zeros(shape);
        Zip::from(&mut out)
            .and(a)
            .and(b)
            .for_each(|r, &p, &q| *r = p - q);
        Ok(out.into_pyarray(py).into_py(py))
    }
}

//
// Soil Adjusted Vegetation Index (SAVI)
// Formula: SAVI = (NIR - Red) / (NIR + Red + L) * (1 + L)
// Default L = 0.5 but user may supply alternative (0 <= L <= 1 usually).
// Zero-denominator safeguard using EPSILON similar to other indices.
//
#[pyfunction(signature = (nir, red, l=0.5))]
pub fn savi(py: Python<'_>, nir: &PyAny, red: &PyAny, l: f64) -> PyResult<PyObject> {
    if l < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "SAVI L must be non-negative, got {}",
            l
        )));
    }
    if let Ok(nir_1d) = nir.extract::<PyReadonlyArray1<f64>>() {
        let red_1d = red.extract::<PyReadonlyArray1<f64>>()?;
        if nir_1d.shape() != red_1d.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch for 1D SAVI inputs: nir {:?}, red {:?}",
                nir_1d.shape(),
                red_1d.shape()
            )));
        }
        savi_1d(py, nir_1d, red_1d, l).map(|res| res.into_py(py))
    } else if let Ok(nir_2d) = nir.extract::<PyReadonlyArray2<f64>>() {
        let red_2d = red.extract::<PyReadonlyArray2<f64>>()?;
        if nir_2d.shape() != red_2d.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch for 2D SAVI inputs: nir {:?}, red {:?}",
                nir_2d.shape(),
                red_2d.shape()
            )));
        }
        savi_2d(py, nir_2d, red_2d, l).map(|res| res.into_py(py))
    } else if let Ok(nir_3d) = nir.extract::<PyReadonlyArray3<f64>>() {
        let red_3d = red.extract::<PyReadonlyArray3<f64>>()?;
        if nir_3d.shape() != red_3d.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch for 3D SAVI inputs: nir {:?}, red {:?}",
                nir_3d.shape(),
                red_3d.shape()
            )));
        }
        savi_3d(py, nir_3d, red_3d, l).map(|res| res.into_py(py))
    } else if let Ok(nir_4d) = nir.extract::<PyReadonlyArray4<f64>>() {
        let red_4d = red.extract::<PyReadonlyArray4<f64>>()?;
        if nir_4d.shape() != red_4d.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch for 4D SAVI inputs: nir {:?}, red {:?}",
                nir_4d.shape(),
                red_4d.shape()
            )));
        }
        savi_4d(py, nir_4d, red_4d, l).map(|res| res.into_py(py))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input arrays must be either 1D, 2D, 3D, or 4D numpy arrays of type float64 for SAVI.",
        ))
    }
}

fn savi_1d<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray1<f64>,
    red: PyReadonlyArray1<f64>,
    l: f64,
) -> PyResult<&'py PyArray1<f64>> {
    let nir = nir.as_array();
    let red = red.as_array();
    let mut result = Array1::<f64>::zeros(nir.len());

    Zip::from(&mut result)
        .and(&nir)
        .and(&red)
        .for_each(|r, &nir_v, &red_v| {
            let denom = nir_v + red_v + l;
            *r = if denom.abs() < EPSILON {
                0.0
            } else {
                (nir_v - red_v) / denom * (1.0 + l)
            };
        });

    Ok(result.into_pyarray(py))
}

fn savi_2d<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray2<f64>,
    red: PyReadonlyArray2<f64>,
    l: f64,
) -> PyResult<&'py PyArray2<f64>> {
    let nir = nir.as_array();
    let red = red.as_array();
    let shape = nir.dim();
    let mut result = Array2::<f64>::zeros(shape);

    Zip::from(&mut result)
        .and(&nir)
        .and(&red)
        .for_each(|r, &nir_v, &red_v| {
            let denom = nir_v + red_v + l;
            *r = if denom.abs() < EPSILON {
                0.0
            } else {
                (nir_v - red_v) / denom * (1.0 + l)
            };
        });

    Ok(result.into_pyarray(py))
}

fn savi_3d<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray3<f64>,
    red: PyReadonlyArray3<f64>,
    l: f64,
) -> PyResult<&'py PyArray3<f64>> {
    let nir = nir.as_array();
    let red = red.as_array();
    let shape = nir.dim();
    let mut result = ndarray::Array3::<f64>::zeros(shape);

    Zip::from(&mut result)
        .and(&nir)
        .and(&red)
        .for_each(|r, &nir_v, &red_v| {
            let denom = nir_v + red_v + l;
            *r = if denom.abs() < EPSILON {
                0.0
            } else {
                (nir_v - red_v) / denom * (1.0 + l)
            };
        });

    Ok(result.into_pyarray(py))
}

fn savi_4d<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray4<f64>,
    red: PyReadonlyArray4<f64>,
    l: f64,
) -> PyResult<&'py PyArray4<f64>> {
    let nir = nir.as_array();
    let red = red.as_array();
    let shape = nir.dim();
    let mut result = ndarray::Array4::<f64>::zeros(shape);

    Zip::from(&mut result)
        .and(&nir)
        .and(&red)
        .for_each(|r, &nir_v, &red_v| {
            let denom = nir_v + red_v + l;
            *r = if denom.abs() < EPSILON {
                0.0
            } else {
                (nir_v - red_v) / denom * (1.0 + l)
            };
        });

    Ok(result.into_pyarray(py))
}

//
// Normalized Burn Ratio (NBR) already defined; below we add:
// NDMI: (NIR - SWIR1) / (NIR + SWIR1)
// NBR2: (SWIR1 - SWIR2) / (SWIR1 + SWIR2)
// GCI:  (NIR / Green) - 1
//

#[pyfunction]
pub fn ndmi(py: Python<'_>, nir: &PyAny, swir1: &PyAny) -> PyResult<PyObject> {
    normalized_difference(py, nir, swir1)
}

#[pyfunction]
pub fn nbr2(py: Python<'_>, swir1: &PyAny, swir2: &PyAny) -> PyResult<PyObject> {
    normalized_difference(py, swir1, swir2)
}

#[pyfunction]
pub fn gci(py: Python<'_>, nir: &PyAny, green: &PyAny) -> PyResult<PyObject> {
    if let Ok(nir_1d) = nir.extract::<PyReadonlyArray1<f64>>() {
        let green_1d = green.extract::<PyReadonlyArray1<f64>>()?;
        if nir_1d.shape() != green_1d.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch for 1D GCI inputs: nir {:?}, green {:?}",
                nir_1d.shape(),
                green_1d.shape()
            )));
        }
        let nir_arr = nir_1d.as_array();
        let green_arr = green_1d.as_array();
        let mut out = Array1::<f64>::zeros(nir_arr.len());
        Zip::from(&mut out)
            .and(&nir_arr)
            .and(&green_arr)
            .for_each(|r, &n, &g| {
                *r = if g.abs() < EPSILON {
                    0.0
                } else {
                    (n / g) - 1.0
                };
            });
        Ok(out.into_pyarray(py).into_py(py))
    } else if let Ok(nir_2d) = nir.extract::<PyReadonlyArray2<f64>>() {
        let green_2d = green.extract::<PyReadonlyArray2<f64>>()?;
        if nir_2d.shape() != green_2d.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch for 2D GCI inputs: nir {:?}, green {:?}",
                nir_2d.shape(),
                green_2d.shape()
            )));
        }
        let nir_arr = nir_2d.as_array();
        let green_arr = green_2d.as_array();
        let shape = nir_arr.dim();
        let mut out = Array2::<f64>::zeros(shape);
        Zip::from(&mut out)
            .and(&nir_arr)
            .and(&green_arr)
            .for_each(|r, &n, &g| {
                *r = if g.abs() < EPSILON {
                    0.0
                } else {
                    (n / g) - 1.0
                };
            });
        Ok(out.into_pyarray(py).into_py(py))
    } else if let Ok(nir_3d) = nir.extract::<PyReadonlyArray3<f64>>() {
        let green_3d = green.extract::<PyReadonlyArray3<f64>>()?;
        if nir_3d.shape() != green_3d.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch for 3D GCI inputs: nir {:?}, green {:?}",
                nir_3d.shape(),
                green_3d.shape()
            )));
        }
        let nir_arr = nir_3d.as_array();
        let green_arr = green_3d.as_array();
        let shape = nir_arr.dim();
        let mut out = ndarray::Array3::<f64>::zeros(shape);
        Zip::from(&mut out)
            .and(&nir_arr)
            .and(&green_arr)
            .for_each(|r, &n, &g| {
                *r = if g.abs() < EPSILON {
                    0.0
                } else {
                    (n / g) - 1.0
                };
            });
        Ok(out.into_pyarray(py).into_py(py))
    } else if let Ok(nir_4d) = nir.extract::<PyReadonlyArray4<f64>>() {
        let green_4d = green.extract::<PyReadonlyArray4<f64>>()?;
        if nir_4d.shape() != green_4d.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch for 4D GCI inputs: nir {:?}, green {:?}",
                nir_4d.shape(),
                green_4d.shape()
            )));
        }
        let nir_arr = nir_4d.as_array();
        let green_arr = green_4d.as_array();
        let shape = nir_arr.dim();
        let mut out = ndarray::Array4::<f64>::zeros(shape);
        Zip::from(&mut out)
            .and(&nir_arr)
            .and(&green_arr)
            .for_each(|r, &n, &g| {
                *r = if g.abs() < EPSILON {
                    0.0
                } else {
                    (n / g) - 1.0
                };
            });
        Ok(out.into_pyarray(py).into_py(py))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input arrays must be 1D, 2D, 3D, or 4D numpy float64 arrays for GCI.",
        ))
    }
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
    } else if let Ok(nir_3d) = nir.extract::<PyReadonlyArray3<f64>>() {
        let red_3d = red.extract::<PyReadonlyArray3<f64>>()?;
        let blue_3d = blue.extract::<PyReadonlyArray3<f64>>()?;
        if nir_3d.shape() != red_3d.shape() || nir_3d.shape() != blue_3d.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch for 3D EVI inputs: nir {:?}, red {:?}, blue {:?}",
                nir_3d.shape(),
                red_3d.shape(),
                blue_3d.shape()
            )));
        }
        enhanced_vegetation_index_3d(py, nir_3d, red_3d, blue_3d).map(|res| res.into_py(py))
    } else if let Ok(nir_4d) = nir.extract::<PyReadonlyArray4<f64>>() {
        let red_4d = red.extract::<PyReadonlyArray4<f64>>()?;
        let blue_4d = blue.extract::<PyReadonlyArray4<f64>>()?;
        if nir_4d.shape() != red_4d.shape() || nir_4d.shape() != blue_4d.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch for 4D EVI inputs: nir {:?}, red {:?}, blue {:?}",
                nir_4d.shape(),
                red_4d.shape(),
                blue_4d.shape()
            )));
        }
        enhanced_vegetation_index_4d(py, nir_4d, red_4d, blue_4d).map(|res| res.into_py(py))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input arrays must be either 1D, 2D, 3D, or 4D numpy arrays of type float64.",
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

fn enhanced_vegetation_index_3d<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray3<f64>,
    red: PyReadonlyArray3<f64>,
    blue: PyReadonlyArray3<f64>,
) -> PyResult<&'py PyArray3<f64>> {
    const G: f64 = 2.5;
    const C1: f64 = 6.0;
    const C2: f64 = 7.5;
    const L: f64 = 1.0;

    let nir = nir.as_array();
    let red = red.as_array();
    let blue = blue.as_array();

    let shape = nir.dim();
    let mut result = ndarray::Array3::<f64>::zeros(shape);

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

fn enhanced_vegetation_index_4d<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray4<f64>,
    red: PyReadonlyArray4<f64>,
    blue: PyReadonlyArray4<f64>,
) -> PyResult<&'py PyArray4<f64>> {
    const G: f64 = 2.5;
    const C1: f64 = 6.0;
    const C2: f64 = 7.5;
    const L: f64 = 1.0;

    let nir = nir.as_array();
    let red = red.as_array();
    let blue = blue.as_array();

    let shape = nir.dim();
    let mut result = ndarray::Array4::<f64>::zeros(shape);

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
mod savi_nbr_tests {
    use super::*;
    use numpy::{PyArray1, PyArray2};
    use pyo3::Python;

    #[test]
    fn test_savi_1d() {
        Python::with_gil(|py| {
            let nir = PyArray1::from_vec(py, vec![0.7_f64, 0.6_f64]);
            let red = PyArray1::from_vec(py, vec![0.2_f64, 0.3_f64]);
            let out = savi(py, nir, red, 0.5).expect("SAVI 1D failed");
            let arr = out
                .as_ref(py)
                .downcast::<PyArray1<f64>>()
                .unwrap()
                .to_owned_array();
            let l = 0.5;
            let expected: Vec<f64> = nir
                .readonly()
                .as_array()
                .iter()
                .zip(red.readonly().as_array().iter())
                .map(|(&n, &r)| {
                    let denom = n + r + l;
                    if denom.abs() < EPSILON {
                        0.0
                    } else {
                        (n - r) / denom * (1.0 + l)
                    }
                })
                .collect();
            for (i, v) in expected.iter().enumerate() {
                assert!((arr[i] - v).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn test_savi_2d() {
        Python::with_gil(|py| {
            let nir_vals = vec![vec![0.6_f64, 0.7_f64], vec![0.5_f64, 0.4_f64]];
            let red_vals = vec![vec![0.2_f64, 0.3_f64], vec![0.1_f64, 0.2_f64]];
            let nir = PyArray2::from_vec2(py, &nir_vals).unwrap();
            let red = PyArray2::from_vec2(py, &red_vals).unwrap();
            let out = savi(py, nir, red, 0.5).expect("SAVI 2D failed");
            let arr = out
                .as_ref(py)
                .downcast::<PyArray2<f64>>()
                .unwrap()
                .to_owned_array();
            assert_eq!(arr.shape(), &[2, 2]);
            let l = 0.5;
            for r in 0..2 {
                for c in 0..2 {
                    let n = nir.readonly().as_array()[[r, c]];
                    let rd = red.readonly().as_array()[[r, c]];
                    let denom = n + rd + l;
                    let expected = if denom.abs() < EPSILON {
                        0.0
                    } else {
                        (n - rd) / denom * (1.0 + l)
                    };
                    assert!((arr[[r, c]] - expected).abs() < 1e-12);
                }
            }
        });
    }

    #[test]
    fn test_savi_shape_mismatch() {
        Python::with_gil(|py| {
            let nir = PyArray1::from_vec(py, vec![0.7_f64, 0.6_f64]);
            let red = PyArray1::from_vec(py, vec![0.2_f64]);
            let err = savi(py, nir, red, 0.5).unwrap_err();
            assert!(err.to_string().contains("Shape mismatch"));
        });
    }

    #[test]
    fn test_savi_zero_denominator() {
        // Choose values so nir + red + L == 0 (nir=-0.25, red=-0.25, L=0.5)
        Python::with_gil(|py| {
            let nir = PyArray1::from_vec(py, vec![-0.25_f64]);
            let red = PyArray1::from_vec(py, vec![-0.25_f64]);
            let out = savi(py, nir, red, 0.5).unwrap();
            let arr = out
                .as_ref(py)
                .downcast::<PyArray1<f64>>()
                .unwrap()
                .to_owned_array();
            assert!((arr[0] - 0.0).abs() < 1e-12);
        });
    }

    #[test]
    fn test_nbr_1d() {
        Python::with_gil(|py| {
            let nir = PyArray1::from_vec(py, vec![0.8_f64, 0.6_f64]);
            let swir2 = PyArray1::from_vec(py, vec![0.3_f64, 0.2_f64]);
            let out = nbr(py, nir, swir2).unwrap();
            let arr = out
                .as_ref(py)
                .downcast::<PyArray1<f64>>()
                .unwrap()
                .to_owned_array();
            for i in 0..arr.len() {
                let n = nir.readonly().as_array()[i];
                let s = swir2.readonly().as_array()[i];
                let denom = n + s;
                let expected = if denom.abs() < EPSILON {
                    0.0
                } else {
                    (n - s) / denom
                };
                assert!((arr[i] - expected).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn test_nbr_2d() {
        Python::with_gil(|py| {
            let nir_vals = vec![vec![0.8_f64, 0.7_f64], vec![0.6_f64, 0.5_f64]];
            let swir_vals = vec![vec![0.3_f64, 0.25_f64], vec![0.2_f64, 0.15_f64]];
            let nir = PyArray2::from_vec2(py, &nir_vals).unwrap();
            let swir2 = PyArray2::from_vec2(py, &swir_vals).unwrap();
            let out = nbr(py, nir, swir2).unwrap();
            let arr = out
                .as_ref(py)
                .downcast::<PyArray2<f64>>()
                .unwrap()
                .to_owned_array();
            assert_eq!(arr.shape(), &[2, 2]);
            for r in 0..2 {
                for c in 0..2 {
                    let n = nir.readonly().as_array()[[r, c]];
                    let s = swir2.readonly().as_array()[[r, c]];
                    let denom = n + s;
                    let expected = if denom.abs() < EPSILON {
                        0.0
                    } else {
                        (n - s) / denom
                    };
                    assert!((arr[[r, c]] - expected).abs() < 1e-12);
                }
            }
        });
    }

    #[test]
    fn test_nbr_shape_mismatch() {
        Python::with_gil(|py| {
            let nir = PyArray1::from_vec(py, vec![0.8_f64, 0.7_f64]);
            let swir2 = PyArray1::from_vec(py, vec![0.3_f64]);
            let err = nbr(py, nir, swir2).unwrap_err();
            assert!(err.to_string().contains("Shape mismatch"));
        });
    }
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
