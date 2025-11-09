use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Spatial processing functions for Earth Observation data.

/// Computes the Euclidean distance between two sets of points.
///
/// # Arguments
/// * `points_a` - A 2D array of shape (N, D) representing N points in D dimensions.
/// * `points_b` - A 2D array of shape (M, D) representing M points in D dimensions.
///
/// # Returns
/// A 2D array of shape (N, M) where the element at (i, j) is the distance between points_a[i] and points_b[j].
#[pyfunction]
pub fn euclidean_distance(
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
    fn test_euclidean_distance() {
        Python::with_gil(|py| {
            let points_a = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
            let points_b = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
            let a_array = PyArray2::from_vec2(py, &points_a).unwrap();
            let b_array = PyArray2::from_vec2(py, &points_b).unwrap();
            let result = euclidean_distance(py, a_array.readonly(), b_array.readonly());
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
