use ndarray::{Array1, Array2, Array3, Zip};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::prelude::*;

/// Threshold for detecting near-zero values to avoid division by zero
const EPSILON: f64 = 1e-10;

// EVI constants
const EVI_GAIN: f64 = 2.5;
const EVI_L: f64 = 1.0;
const EVI_C1: f64 = 6.0;
const EVI_C2: f64 = 7.5;

/// Compute normalized difference between two arrays.
///
/// This is the core function for (a - b) / (a + b).
///
/// # Arguments
/// * `a` - First input array (e.g., NIR band)
/// * `b` - Second input array (e.g., Red band)
///
/// # Returns
/// Array with the normalized difference values.
#[pyfunction]
pub fn normalized_difference_1d<'py>(
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
#[pyfunction]
pub fn normalized_difference_2d<'py>(
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

/// Compute normalized difference between two 3D arrays.
#[pyfunction]
pub fn normalized_difference_3d<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<f64>,
    b: PyReadonlyArray3<f64>,
) -> PyResult<&'py PyArray3<f64>> {
    let a = a.as_array();
    let b = b.as_array();
    let shape = a.dim();
    let mut result = Array3::<f64>::zeros(shape);

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
/// NDVI = (NIR - Red) / (NIR + Red)
#[pyfunction]
pub fn ndvi_1d<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray1<f64>,
    red: PyReadonlyArray1<f64>,
) -> PyResult<&'py PyArray1<f64>> {
    normalized_difference_1d(py, nir, red)
}

/// Compute NDVI (Normalized Difference Vegetation Index) from NIR and Red bands (2D).
#[pyfunction]
pub fn ndvi_2d<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray2<f64>,
    red: PyReadonlyArray2<f64>,
) -> PyResult<&'py PyArray2<f64>> {
    normalized_difference_2d(py, nir, red)
}

/// Compute NDVI (Normalized Difference Vegetation Index) from NIR and Red bands (3D).
#[pyfunction]
pub fn ndvi_3d<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray3<f64>,
    red: PyReadonlyArray3<f64>,
) -> PyResult<&'py PyArray3<f64>> {
    normalized_difference_3d(py, nir, red)
}

/// Compute NDWI (Normalized Difference Water Index) from Green and NIR bands.
/// NDWI = (Green - NIR) / (Green + NIR)
#[pyfunction]
pub fn ndwi_1d<'py>(
    py: Python<'py>,
    green: PyReadonlyArray1<f64>,
    nir: PyReadonlyArray1<f64>,
) -> PyResult<&'py PyArray1<f64>> {
    normalized_difference_1d(py, green, nir)
}

/// Compute NDWI (Normalized Difference Water Index) from Green and NIR bands (2D).
#[pyfunction]
pub fn ndwi_2d<'py>(
    py: Python<'py>,
    green: PyReadonlyArray2<f64>,
    nir: PyReadonlyArray2<f64>,
) -> PyResult<&'py PyArray2<f64>> {
    normalized_difference_2d(py, green, nir)
}

/// Compute EVI (Enhanced Vegetation Index) from NIR, Red, and Blue bands.
///
/// EVI = G * (NIR - RED) / (NIR + C1 * RED - C2 * BLUE + L)
///
/// G = 2.5, L = 1.0, C1 = 6.0, C2 = 7.5
///
/// # Arguments
/// * `nir` - Near-infrared band values
/// * `red` - Red band values
/// * `blue` - Blue band values
///
/// # Returns
/// EVI values ranging from -1 to 1
#[pyfunction]
pub fn enhanced_vegetation_index_1d<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray1<f64>,
    red: PyReadonlyArray1<f64>,
    blue: PyReadonlyArray1<f64>,
) -> PyResult<&'py PyArray1<f64>> {
    let nir = nir.as_array();
    let red = red.as_array();
    let blue = blue.as_array();

    let mut result = Array1::<f64>::zeros(nir.len());

    Zip::from(&mut result)
        .and(&nir)
        .and(&red)
        .and(&blue)
        .for_each(|r, &nir_val, &red_val, &blue_val| {
            // Denominator: NIR + C1 * RED - C2 * BLUE + L
            let denominator = nir_val + EVI_C1 * red_val - EVI_C2 * blue_val + EVI_L;

            // Numerator: NIR - RED
            let numerator = nir_val - red_val;

            *r = if denominator.abs() < EPSILON {
                0.0
            } else {
                EVI_GAIN * numerator / denominator
            };
        });

    Ok(result.into_pyarray(py))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_enhanced_vegetation_index_1d_basic() {
        // Example values:
        // NIR = 0.4, RED = 0.1, BLUE = 0.05
        let nir = array![0.4];
        let red = array![0.1];
        let blue = array![0.05];

        // EVI = 2.5 * (0.4 - 0.1) / (0.4 + 6.0 * 0.1 - 7.5 * 0.05 + 1.0)
        // Numerator = 2.5 * 0.3 = 0.75
        // Denominator = 0.4 + 0.6 - 0.375 + 1.0 = 1.625
        // Expected = 0.75 / 1.625 = 0.4615384615...

        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let nir_py = nir.into_pyarray(py);
            let red_py = red.into_pyarray(py);
            let blue_py = blue.into_pyarray(py);
            let result = enhanced_vegetation_index_1d(
                py,
                nir_py.readonly(),
                red_py.readonly(),
                blue_py.readonly(),
            )
            .unwrap();

            let readonly_result = result.readonly();
            let result_array = readonly_result.as_array();

            assert_relative_eq!(result_array[0], 0.46153846153846156, epsilon = 1e-10);
        });
    }

    #[test]
    fn test_enhanced_vegetation_index_1d_zero_denominator() {
        // Values designed to make the denominator near zero:
        // NIR + C1 * RED - C2 * BLUE + L = 0
        // 0.1 + 6.0 * 0.5 - 7.5 * 0.5466666667 + 1.0 = 0.1 + 3.0 - 4.1 + 1.0 = 0.0
        let nir = array![0.1];
        let red = array![0.5];
        let blue = array![0.5466666667];

        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let nir_py = nir.into_pyarray(py);
            let red_py = red.into_pyarray(py);
            let blue_py = blue.into_pyarray(py);
            let result = enhanced_vegetation_index_1d(
                py,
                nir_py.readonly(),
                red_py.readonly(),
                blue_py.readonly(),
            )
            .unwrap();

            let readonly_result = result.readonly();
            let result_array = readonly_result.as_array();
            // Should return 0.0 due to EPSILON check
            assert_relative_eq!(result_array[0], 0.0, epsilon = 1e-10);
        });
    }
}
