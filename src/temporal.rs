use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Computes the temporal mean of a 1D array.
///
/// # Arguments
/// * `data` - A 1D array of temporal data.
///
/// # Returns
/// The mean of the array.
#[pyfunction]
pub fn temporal_mean_1d(
    data: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let input_array = data.as_array();
    let sum: f64 = input_array.sum();
    let count = input_array.len() as f64;
    Ok(sum / count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;
    use numpy::PyArray1;

    #[test]
    fn test_temporal_mean_1d_basic() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];

        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let data = PyArray1::from_array(py, &data);
            let result = temporal_mean_1d(data.readonly()).unwrap();
            assert_relative_eq!(result, 3.0, epsilon = 1e-10);
        });
    }
}
