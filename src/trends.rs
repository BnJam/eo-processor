// src/trends.rs

use ndarray::Array1;
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
pub struct TrendSegment {
    #[pyo3(get)]
    pub start_index: usize,
    #[pyo3(get)]
    pub end_index: usize,
    #[pyo3(get)]
    pub slope: f64,
    #[pyo3(get)]
    pub intercept: f64,
}

#[pyfunction]
pub fn trend_analysis(y: Vec<f64>, threshold: f64) -> PyResult<Vec<TrendSegment>> {
    let mut segments = Vec::new();
    let mut start = 0;

    while start < y.len() {
        let mut end = start + 1;
        let mut best_slope = 0.0;
        let mut best_intercept = 0.0;

        while end <= y.len() {
            let segment_y = &y[start..end];
            if segment_y.len() < 2 {
                end += 1;
                continue;
            }

            let (slope, intercept) = calculate_linear_regression(segment_y);
            let residuals: Vec<f64> = segment_y
                .iter()
                .enumerate()
                .map(|(i, &yi)| yi - (slope * i as f64 + intercept))
                .collect();
            let max_residual = residuals.iter().map(|&r| r.abs()).fold(0.0, f64::max);

            if max_residual > threshold {
                break;
            }

            best_slope = slope;
            best_intercept = intercept;
            end += 1;
        }

        segments.push(TrendSegment {
            start_index: start,
            end_index: end - 2,
            slope: best_slope,
            intercept: best_intercept,
        });

        start = end - 1;
    }

    Ok(segments)
}

fn calculate_linear_regression(y: &[f64]) -> (f64, f64) {
    let n = y.len() as f64;
    let x_sum: f64 = (0..y.len()).map(|i| i as f64).sum();
    let y_sum: f64 = y.iter().sum();
    let xy_sum: f64 = y.iter().enumerate().map(|(i, &yi)| i as f64 * yi).sum();
    let x_sq_sum: f64 = (0..y.len()).map(|i| (i as f64).powi(2)).sum();

    let slope = (n * xy_sum - x_sum * y_sum) / (n * x_sq_sum - x_sum.powi(2));
    let intercept = (y_sum - slope * x_sum) / n;

    (slope, intercept)
}

#[pyfunction]
pub fn linear_regression(y: Vec<f64>) -> PyResult<(f64, f64, Vec<f64>)> {
    let y_arr = Array1::from(y);
    let (slope, intercept) = calculate_linear_regression(y_arr.as_slice().unwrap());

    let residuals: Vec<f64> = y_arr
        .iter()
        .enumerate()
        .map(|(i, &yi)| yi - (slope * i as f64 + intercept))
        .collect();

    Ok((slope, intercept, residuals))
}
