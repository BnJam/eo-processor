
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
    recursive_trend_analysis(&y, 0, &mut segments, threshold);
    Ok(segments)
}

fn recursive_trend_analysis(y: &[f64], start_index: usize, segments: &mut Vec<TrendSegment>, threshold: f64) {
    if y.len() < 2 {
        return;
    }

    let (slope, intercept) = calculate_linear_regression(y);
    let residuals: Vec<f64> = y
        .iter()
        .enumerate()
        .map(|(i, &yi)| yi - (slope * i as f64 + intercept))
        .collect();

    let max_residual = residuals.iter().map(|&r| r.abs()).fold(0.0, f64::max);

    if max_residual <= threshold {
        segments.push(TrendSegment {
            start_index,
            end_index: start_index + y.len() - 1,
            slope,
            intercept,
        });
    } else {
        let max_residual_index = residuals
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let (left, right) = y.split_at(max_residual_index);
        recursive_trend_analysis(left, start_index, segments, threshold);
        recursive_trend_analysis(right, start_index + max_residual_index, segments, threshold);
    }
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
