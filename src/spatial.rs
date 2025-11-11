use ndarray::{s, Array1, Array2, Array3, Axis};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArray4,
};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Spatial processing functions for Earth Observation data.
/// Dispatches median to 3D or 4D implementation based on array dimensions.
#[pyfunction]
#[pyo3(signature = (arr, skip_na=true))]
pub fn median(py: Python<'_>, arr: &PyAny, skip_na: bool) -> PyResult<PyObject> {
    if let Ok(arr1d) = arr.downcast::<numpy::PyArray1<f64>>() {
        Ok(median_1d(arr1d.readonly(), skip_na).into_py(py))
    } else if let Ok(arr2d) = arr.downcast::<numpy::PyArray2<f64>>() {
        Ok(median_2d(py, arr2d.readonly(), skip_na).into_py(py))
    } else if let Ok(arr3d) = arr.downcast::<numpy::PyArray3<f64>>() {
        Ok(median_3d(py, arr3d.readonly(), skip_na).into_py(py))
    } else if let Ok(arr4d) = arr.downcast::<numpy::PyArray4<f64>>() {
        Ok(median_4d(py, arr4d.readonly(), skip_na).into_py(py))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected a 1D, 2D, 3D, or 4D NumPy array.",
        ))
    }
}

/// Computes the median for a 3D array (time, y, x).
fn median_3d<'py>(
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
            if series.iter().any(|v| v.is_nan()) {
                if skip_na {
                    series.retain(|v| !v.is_nan());
                } else {
                    *pixel = f64::NAN;
                    return;
                }
            }

            if series.is_empty() {
                *pixel = f64::NAN;
            } else {
                series.sort_by(|a, b| a.total_cmp(b));
                let mid = series.len() / 2;
                *pixel = if series.len().is_multiple_of(2) {
                    (series[mid - 1] + series[mid]) / 2.0
                } else {
                    series[mid]
                };
            }
        });

    result.into_pyarray(py)
}

/// Computes the median for a 1D array.
fn median_1d(arr: PyReadonlyArray1<f64>, skip_na: bool) -> f64 {
    let mut series: Vec<f64> = arr.as_array().to_vec();
    if series.iter().any(|v| v.is_nan()) {
        if skip_na {
            series.retain(|v| !v.is_nan());
        } else {
            return f64::NAN;
        }
    }

    if series.is_empty() {
        f64::NAN
    } else {
        series.sort_by(|a, b| a.total_cmp(b));
        let mid = series.len() / 2;
        if series.len().is_multiple_of(2) {
            (series[mid - 1] + series[mid]) / 2.0
        } else {
            series[mid]
        }
    }
}

/// Computes the median for a 2D array (time, band).
fn median_2d<'py>(
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
        if series.iter().any(|v| v.is_nan()) {
            if skip_na {
                series.retain(|v| !v.is_nan());
            } else {
                result[i] = f64::NAN;
                continue;
            }
        }

        if series.is_empty() {
            result[i] = f64::NAN;
        } else {
            series.sort_by(|a, b| a.total_cmp(b));
            let mid = series.len() / 2;
            result[i] = if series.len().is_multiple_of(2) {
                (series[mid - 1] + series[mid]) / 2.0
            } else {
                series[mid]
            };
        }
    }

    result.into_pyarray(py)
}

/// Computes the median for a 4D array (time, band, y, x).
fn median_4d<'py>(
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
            if series.iter().any(|v| v.is_nan()) {
                if skip_na {
                    series.retain(|v| !v.is_nan());
                } else {
                    *pixel = f64::NAN;
                    return;
                }
            }

            if series.is_empty() {
                *pixel = f64::NAN;
            } else {
                series.sort_by(|a, b| a.total_cmp(b));
                let mid = series.len() / 2;
                *pixel = if series.len().is_multiple_of(2) {
                    (series[mid - 1] + series[mid]) / 2.0
                } else {
                    series[mid]
                };
            }
        });

    result.into_pyarray(py)
}

/// 1. Euclidean Distance
/// Computes pairwise Euclidean distances between two 2D point sets.
/// Always returns a 2D (N, M) matrix even when N == 1 or M == 1 to keep
/// shape semantics consistent with other distance functions.
#[pyfunction]
pub fn euclidean_distance(
    py: Python<'_>,
    points_a: &PyAny,
    points_b: &PyAny,
) -> PyResult<PyObject> {
    let points_a_array = points_a
        .downcast::<PyArray2<f64>>()
        .expect("points_a should be a 2D array");
    let points_b_array = points_b
        .downcast::<PyArray2<f64>>()
        .expect("points_b should be a 2D array");
    let result = euclidean_distance_2d(py, points_a_array.readonly(), points_b_array.readonly());
    Ok(result.into_py(py))
}

/// Computes the Euclidean distance between two sets of points arrays.
///
/// # Arguments
/// * `points_a` - A 2D array of shape (N, D) representing N points in D dimensions.
/// * `points_b` - A 2D array of shape (M, D) representing M points in D dimensions.
///
/// # Returns
/// A 2D array of shape (N, M) where the element at (i, j) is the distance between points_a[i] and points_b[j].
fn euclidean_distance_2d(
    py: Python,
    points_a: PyReadonlyArray2<f64>,
    points_b: PyReadonlyArray2<f64>,
) -> Py<PyArray2<f64>> {
    let a = points_a.as_array();
    let b = points_b.as_array();
    let n = a.shape()[0];
    let m = b.shape()[0];
    let mut distances = Array2::<f64>::zeros((n, m));
    let threshold = 10_000;
    if n * m > threshold {
        distances
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..m {
                    let dist = a
                        .row(i)
                        .iter()
                        .zip(b.row(j).iter())
                        .map(|(x, y)| (x - y).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    row[j] = dist;
                }
            });
    } else {
        for i in 0..n {
            for j in 0..m {
                let dist = a
                    .row(i)
                    .iter()
                    .zip(b.row(j).iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f64>()
                    .sqrt();
                distances[[i, j]] = dist;
            }
        }
    }
    distances.into_pyarray(py).to_owned()
}

/// Computes the Manhattan distance between two sets of points.
/// # Arguments
/// * `points_a` - A 2D array of shape (N, D) representing N points in D dimensions.
/// * `points_b` - A 2D array of shape (M, D) representing M points in D dimensions.
/// # Returns
/// A 2D array of shape (N, M) where the element at (i, j) is the Manhattan distance between points_a[i] and points_b[j].
#[pyfunction]
pub fn manhattan_distance(
    py: Python,
    points_a: PyReadonlyArray2<f64>,
    points_b: PyReadonlyArray2<f64>,
) -> Py<PyArray2<f64>> {
    let a = points_a.as_array();
    let b = points_b.as_array();
    let n = a.shape()[0];
    let m = b.shape()[0];
    let mut distances = Array2::<f64>::zeros((n, m));
    let threshold = 10_000;
    if n * m > threshold {
        distances
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..m {
                    let dist = a
                        .row(i)
                        .iter()
                        .zip(b.row(j).iter())
                        .map(|(x, y)| (x - y).abs())
                        .sum::<f64>();
                    row[j] = dist;
                }
            });
    } else {
        for i in 0..n {
            for j in 0..m {
                let dist = a
                    .row(i)
                    .iter()
                    .zip(b.row(j).iter())
                    .map(|(x, y)| (x - y).abs())
                    .sum::<f64>();
                distances[[i, j]] = dist;
            }
        }
    }
    distances.into_pyarray(py).to_owned()
}

/// Computes the Chebyshev distance between two sets of points.
/// # Arguments
/// * `points_a` - A 2D array of shape (N, D) representing N points in D dimensions.
/// * `points_b` - A 2D array of shape (M, D) representing M points in D dimensions.
/// # Returns
/// A 2D array of shape (N, M) where the element at (i, j) is the Chebyshev distance between points_a[i] and points_b[j].
#[pyfunction]
pub fn chebyshev_distance(
    py: Python,
    points_a: PyReadonlyArray2<f64>,
    points_b: PyReadonlyArray2<f64>,
) -> Py<PyArray2<f64>> {
    let a = points_a.as_array();
    let b = points_b.as_array();
    let n = a.shape()[0];
    let m = b.shape()[0];
    let mut distances = Array2::<f64>::zeros((n, m));
    let threshold = 10_000;
    if n * m > threshold {
        distances
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..m {
                    let dist = a
                        .row(i)
                        .iter()
                        .zip(b.row(j).iter())
                        .map(|(x, y)| (x - y).abs())
                        .fold(f64::NAN, f64::max);
                    row[j] = dist;
                }
            });
    } else {
        for i in 0..n {
            for j in 0..m {
                let dist = a
                    .row(i)
                    .iter()
                    .zip(b.row(j).iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(f64::NAN, f64::max); // max of absolute differences
                distances[[i, j]] = dist;
            }
        }
    }
    distances.into_pyarray(py).to_owned()
}

/// Computes the Minkowski distance between two sets of points.
/// # Arguments
/// * `points_a` - A 2D array of shape (N, D) representing N points in D dimensions.
/// * `points_b` - A 2D array of shape (M, D) representing M points in D dimensions.
/// * `p` - The order of the norm (p >= 1).
/// # Returns
/// A 2D array of shape (N, M) where the element at (i, j) is the Minkowski distance between points_a[i] and points_b[j].
#[pyfunction]
pub fn minkowski_distance(
    py: Python,
    points_a: PyReadonlyArray2<f64>,
    points_b: PyReadonlyArray2<f64>,
    p: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    if p < 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "p must be >= 1.0, got {}",
            p
        )));
    }
    let a = points_a.as_array();
    let b = points_b.as_array();
    let n = a.shape()[0];
    let m = b.shape()[0];
    let mut distances = Array2::<f64>::zeros((n, m));
    let threshold = 10_000;
    if n * m > threshold {
        distances
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..m {
                    let dist = a
                        .row(i)
                        .iter()
                        .zip(b.row(j).iter())
                        .map(|(x, y)| (x - y).abs().powf(p))
                        .sum::<f64>()
                        .powf(1.0 / p);
                    row[j] = dist;
                }
            });
    } else {
        for i in 0..n {
            for j in 0..m {
                let dist = a
                    .row(i)
                    .iter()
                    .zip(b.row(j).iter())
                    .map(|(x, y)| (x - y).abs().powf(p))
                    .sum::<f64>()
                    .powf(1.0 / p);
                distances[[i, j]] = dist;
            }
        }
    }
    Ok(distances.into_pyarray(py).to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArray2;

    #[test]
    fn test_euclidean_distance_2d() {
        Python::with_gil(|py| {
            let points_a = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
            let points_b = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
            let a_array = PyArray2::from_vec2(py, &points_a).unwrap();
            let b_array = PyArray2::from_vec2(py, &points_b).unwrap();
            let result = euclidean_distance_2d(py, a_array.readonly(), b_array.readonly());
            let result_array = result.as_ref(py).to_owned_array();
            let expected = ndarray::array![[1.0, 1.0], [1.0, 1.0]];
            assert_eq!(result_array, expected);
        });
    }

    #[test]
    fn test_manhattan_distance() {
        Python::with_gil(|py| {
            let points_a = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
            let points_b = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
            let a_array = PyArray2::from_vec2(py, &points_a).unwrap();
            let b_array = PyArray2::from_vec2(py, &points_b).unwrap();
            let result = manhattan_distance(py, a_array.readonly(), b_array.readonly());
            let result_array = result.as_ref(py).to_owned_array();
            let expected = ndarray::array![[1.0, 1.0], [1.0, 1.0]];
            assert_eq!(result_array, expected);
        });
    }

    #[test]
    fn test_chebyshev_distance() {
        Python::with_gil(|py| {
            let points_a = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
            let points_b = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
            let a_array = PyArray2::from_vec2(py, &points_a).unwrap();
            let b_array = PyArray2::from_vec2(py, &points_b).unwrap();
            let result = chebyshev_distance(py, a_array.readonly(), b_array.readonly());
            let result_array = result.as_ref(py).to_owned_array();
            let expected = ndarray::array![[1.0, 1.0], [1.0, 1.0]];
            assert_eq!(result_array, expected);
        });
    }
}
