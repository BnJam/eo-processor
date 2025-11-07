use labeledarray::LabeledArray;
use ndarray::{ArrayD, IxDyn, Zip};
use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use pyo3::prelude::*;

/// Threshold for detecting near-zero values to avoid division by zero
const EPSILON: f64 = 1e-10;

// EVI constants
const EVI_GAIN: f64 = 2.5;
const EVI_L: f64 = 1.0;
const EVI_C1: f64 = 6.0;
const EVI_C2: f64 = 7.5;

/// Compute normalized difference between two labeled arrays.
///
/// This is the core function for (a - b) / (a + b).
///
/// # Arguments
/// * `a` - First input LabeledArray (e.g., NIR band)
/// * `b` - Second input LabeledArray (e.g., Red band)
///
/// # Returns
/// A LabeledArray with the normalized difference values.
fn _normalized_difference(
    a: &LabeledArray<f64>,
    b: &LabeledArray<f64>,
) -> Result<LabeledArray<f64>, String> {
    if a.shape() != b.shape() {
        return Err("Input arrays must have the same shape.".to_string());
    }
    if a.dims() != b.dims() {
        return Err("Input arrays must have the same dimension labels.".to_string());
    }

    let mut result_data = ArrayD::<f64>::zeros(a.shape());

    Zip::from(&mut result_data)
        .and(a.data())
        .and(b.data())
        .for_each(|r, &a_val, &b_val| {
            let sum = a_val + b_val;
            *r = if sum.abs() < EPSILON {
                0.0
            } else {
                (a_val - b_val) / sum
            };
        });

    Ok(LabeledArray::new(result_data, a.dims().to_vec()))
}

/// Compute normalized difference between two arrays.
///
/// This is the core function for (a - b) / (a + b).
///
/// # Arguments
/// * `a` - First input array (e.g., NIR band)
/// * `b` - Second input array (e.g., Red band)
/// * `dims` - A vector of strings with the dimension names.
///
/// # Returns
/// A tuple containing the array with the normalized difference values and the dimension names.
#[pyfunction]
pub fn normalized_difference<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<f64>,
    b: PyReadonlyArrayDyn<f64>,
    dims: Vec<String>,
) -> PyResult<PyObject> {
    let a_labeled = LabeledArray::new(a.as_array().to_owned(), dims.clone());
    let b_labeled = LabeledArray::new(b.as_array().to_owned(), dims);

    let result =
        _normalized_difference(&a_labeled, &b_labeled).map_err(pyo3::exceptions::PyValueError::new_err)?;

    let output_array = result.data().to_owned().into_pyarray(py);
    let output_dims = result.dims().to_vec();
    let tuple = (output_array, output_dims).into_py(py);
    Ok(tuple)
}

/// Compute NDVI (Normalized Difference Vegetation Index) from NIR and Red bands.
/// NDVI = (NIR - Red) / (NIR + Red)
#[pyfunction]
pub fn ndvi<'py>(
    py: Python<'py>,
    nir: PyReadonlyArrayDyn<f64>,
    red: PyReadonlyArrayDyn<f64>,
    dims: Vec<String>,
) -> PyResult<PyObject> {
    normalized_difference(py, nir, red, dims)
}

/// Compute NDWI (Normalized Difference Water Index) from Green and NIR bands.
/// NDWI = (Green - NIR) / (Green + NIR)
#[pyfunction]
pub fn ndwi<'py>(
    py: Python<'py>,
    green: PyReadonlyArrayDyn<f64>,
    nir: PyReadonlyArrayDyn<f64>,
    dims: Vec<String>,
) -> PyResult<PyObject> {
    normalized_difference(py, green, nir, dims)
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
/// * `dims` - A vector of strings with the dimension names.
///
/// # Returns
/// A tuple containing the EVI values array and the dimension names.
#[pyfunction]
pub fn enhanced_vegetation_index<'py>(
    py: Python<'py>,
    nir: PyReadonlyArrayDyn<f64>,
    red: PyReadonlyArrayDyn<f64>,
    blue: PyReadonlyArrayDyn<f64>,
    dims: Vec<String>,
) -> PyResult<PyObject> {
    let nir_labeled = LabeledArray::new(nir.as_array().to_owned(), dims.clone());
    let red_labeled = LabeledArray::new(red.as_array().to_owned(), dims.clone());
    let blue_labeled = LabeledArray::new(blue.as_array().to_owned(), dims);

    if !(nir.shape() == red.shape() && red.shape() == blue.shape()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Input arrays must have the same shape.",
        ));
    }

    let mut result_data = ArrayD::<f64>::zeros(IxDyn(nir.shape()));

    Zip::from(&mut result_data)
        .and(nir_labeled.data())
        .and(red_labeled.data())
        .and(blue_labeled.data())
        .for_each(|r, &nir_val, &red_val, &blue_val| {
            let denominator = nir_val + EVI_C1 * red_val - EVI_C2 * blue_val + EVI_L;
            let numerator = nir_val - red_val;
            *r = if denominator.abs() < EPSILON {
                0.0
            } else {
                EVI_GAIN * numerator / denominator
            };
        });

    let result_labeled = LabeledArray::new(result_data, nir_labeled.dims().to_vec());

    let output_array = result_labeled.data().to_owned().into_pyarray(py);
    let output_dims = result_labeled.dims().to_vec();
    let tuple = (output_array, output_dims).into_py(py);
    Ok(tuple)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{arr1, arr2};
    use numpy::{PyArrayDyn, ToPyArray};

    #[test]
    fn test_normalized_difference_labeled() {
        let a_data = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b_data = arr2(&[[2.0, 1.0], [1.0, 2.0]]);
        let dims = vec!["y".to_string(), "x".to_string()];

        let a = LabeledArray::new(a_data.into_dyn(), dims.clone());
        let b = LabeledArray::new(b_data.into_dyn(), dims);

        // Expected: (a - b) / (a + b)
        // [[-1/3, 1/3], [2/4, 2/6]] = [[-0.333..., 0.333...], [0.5, 0.333...]]
        let result = _normalized_difference(&a, &b).unwrap();

        let expected_data = arr2(&[[-1.0 / 3.0, 1.0 / 3.0], [0.5, 1.0 / 3.0]]);
        assert_eq!(result.dims(), &vec!["y".to_string(), "x".to_string()]);

        result
            .data()
            .iter()
            .zip(expected_data.iter())
            .for_each(|(v1, v2)| {
                assert_relative_eq!(*v1, *v2, epsilon = 1e-10);
            });
    }

    #[test]
    fn test_enhanced_vegetation_index_labeled() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let nir = arr1(&[0.4]).into_dyn();
            let red = arr1(&[0.1]).into_dyn();
            let blue = arr1(&[0.05]).into_dyn();
            let dims = vec!["band".to_string()];

            let result_obj = enhanced_vegetation_index(
                py,
                nir.to_pyarray(py).readonly(),
                red.to_pyarray(py).readonly(),
                blue.to_pyarray(py).readonly(),
                dims,
            )
            .unwrap();

            let result_tuple: (&PyArrayDyn<f64>, Vec<String>) = result_obj.extract(py).unwrap();
            let readonly_array = result_tuple.0.readonly();
            let result_array = readonly_array.as_array();
            let result_dims = result_tuple.1;

            assert_relative_eq!(
                *result_array.get([0]).unwrap(),
                0.46153846153846156,
                epsilon = 1e-10
            );
            assert_eq!(result_dims, vec!["band".to_string()]);
        });
    }
}
