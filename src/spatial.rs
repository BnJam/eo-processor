use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Spatial processing functions for Earth Observation data.

/// 1. Euclidean Distance
#[pyfunction]
pub fn euclidean_distance(
    py: Python<'_>,
    points_a: &PyAny,
    points_b: &PyAny,
) -> PyResult<PyObject> {
    if let Ok(points_a_shape) = points_a.getattr("shape") {
        if let Ok(shape) = points_a_shape.extract::<(usize, usize)>() {
            if shape.0 == 1 {
                // Single point in points_a
                let point_a = points_a
                    .downcast::<PyArray2<f64>>()
                    .expect("points_a should be a 2D array");
                let point_b = points_b
                    .downcast::<PyArray2<f64>>()
                    .expect("points_b should be a 2D array");
                let dist = euclidean_distance_single(point_a.readonly(), point_b.readonly());
                return Ok(dist.into_py(py));
            }
        }
    }
    // Otherwise, treat as 2D arrays of points
    let points_a_array = points_a
        .downcast::<PyArray2<f64>>()
        .expect("points_a should be a 2D array");
    let points_b_array = points_b
        .downcast::<PyArray2<f64>>()
        .expect("points_b should be a 2D array");
    let result = euclidean_distance_2d(py, points_a_array.readonly(), points_b_array.readonly());
    Ok(result.into_py(py))
}

/// euclidian distance for a single pair of points
/// Computes the Euclidean distance between two points.
/// # Arguments
/// * `point_a` - A 1D array representing a point in D dimensions.
/// * `point_b` - A 1D array representing a point in D dimensions.
/// # Returns
/// The Euclidean distance between point_a and point_b.
fn euclidean_distance_single(
    point_a: PyReadonlyArray2<f64>,
    point_b: PyReadonlyArray2<f64>,
) -> f64 {
    let a = point_a.as_array();
    let b = point_b.as_array();
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
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
    distances.into_pyarray(py).to_owned()
}

/// Computes the Manhattan distance between two sets of points.
////
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
    distances.into_pyarray(py).to_owned()
}

/// Computes the Chebyshev distance between two sets of points.
//// # Arguments
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
    for i in 0..n {
        for j in 0..m {
            let dist = a
                .row(i)
                .iter()
                .zip(b.row(j).iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0. / 0., f64::max); // max of absolute differences
            distances[[i, j]] = dist;
        }
    }
    distances.into_pyarray(py).to_owned()
}

/// Computes the Minkowski distance between two sets of points.
//// # Arguments
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
) -> Py<PyArray2<f64>> {
    let a = points_a.as_array();
    let b = points_b.as_array();
    let n = a.shape()[0];
    let m = b.shape()[0];
    let mut distances = Array2::<f64>::zeros((n, m));
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
    distances.into_pyarray(py).to_owned()
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
