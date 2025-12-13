use ndarray::{Array2, Axis};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

#[pyfunction]
pub fn texture_entropy(
    py: Python<'_>,
    arr: PyReadonlyArray2<f64>,
    window_size: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let array = arr.as_array();
    let (height, width) = (array.shape()[0], array.shape()[1]);
    let mut result = Array2::<f64>::zeros((height, width));
    let half_window = window_size / 2;

    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(r, mut row)| {
            for c in 0..width {
                let r_min = r.saturating_sub(half_window);
                let r_max = (r + half_window).min(height - 1);
                let c_min = c.saturating_sub(half_window);
                let c_max = (c + half_window).min(width - 1);

                let window = array.slice(ndarray::s![r_min..=r_max, c_min..=c_max]);
                let mut hist = HashMap::new();
                for &val in window.iter() {
                    *hist.entry(val.to_bits()).or_insert(0) += 1;
                }

                let mut entropy = 0.0;
                let n_pixels = window.len() as f64;
                for &count in hist.values() {
                    let p = count as f64 / n_pixels;
                    if p > 0.0 {
                        entropy -= p * p.log2();
                    }
                }
                row[c] = entropy;
            }
        });

    Ok(result.into_pyarray(py).to_owned())
}
