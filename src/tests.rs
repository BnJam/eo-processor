
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
