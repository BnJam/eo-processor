use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Applies a 2D median filter to an array.
///
/// This is useful for reducing salt-and-pepper noise in imagery.
///
/// # Arguments
/// * `data` - A 2D array to filter.
/// * `kernel_size` - The size of the square kernel (e.g., 3 for a 3x3 window).
///
/// # Returns
/// A new 2D array with the median filter applied.
#[pyfunction]
pub fn median_filter_2d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    kernel_size: usize,
) -> PyResult<&'py PyArray2<f64>> {
    let input_array = data.as_array();
    let (height, width) = (input_array.shape()[0], input_array.shape()[1]);
    let mut output_array = Array2::<f64>::zeros((height, width));
    let half_kernel = kernel_size / 2;

    for y in 0..height {
        for x in 0..width {
            let mut neighborhood = Vec::new();
            for ky in 0..kernel_size {
                for kx in 0..kernel_size {
                    let sample_y = y as isize + ky as isize - half_kernel as isize;
                    let sample_x = x as isize + kx as isize - half_kernel as isize;

                    if sample_y >= 0 && sample_y < height as isize && sample_x >= 0 && sample_x < width as isize {
                        neighborhood.push(input_array[[sample_y as usize, sample_x as usize]]);
                    }
                }
            }
            neighborhood.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median_index = neighborhood.len() / 2;
            output_array[[y, x]] = neighborhood[median_index];
        }
    }

    Ok(output_array.into_pyarray(py))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;
    use numpy::PyArray2;

    #[test]
    fn test_median_filter_2d_basic() {
        let data = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];

        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let data = PyArray2::from_array(py, &data);
            let result = median_filter_2d(py, data.readonly(), 3).unwrap();
            let result_array = result.readonly();
            let result_array = result_array.as_array();
            assert_relative_eq!(result_array[[1, 1]], 5.0, epsilon = 1e-10);
        });
    }
}
