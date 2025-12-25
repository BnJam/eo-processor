use crate::CoreError;
use ndarray::{Axis, IxDyn, Zip};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use rayon::prelude::*;

// --- 1. BFAST Monitor Workflow ---

// Placeholder struct for model parameters
struct HarmonicModel {
    mean: f64,
}

// Placeholder for fitting a harmonic model to the stable history period.
// In a real implementation, this would involve solving for harmonic coefficients.
fn fit_harmonic_model(y: &[f64]) -> HarmonicModel {
    if y.is_empty() {
        return HarmonicModel { mean: 0.0 };
    }
    let sum: f64 = y.iter().sum();
    HarmonicModel {
        mean: sum / y.len() as f64,
    }
}

// Placeholder for predicting values based on the fitted model.
fn predict_harmonic_model(_model: &HarmonicModel, dates: &[i64]) -> Vec<f64> {
    // For now, just return a constant prediction (the historical mean)
    vec![_model.mean; dates.len()]
}

// Placeholder for the MOSUM process to detect a break.
// Returns (break_date, magnitude)
fn detect_mosum_break(
    y_monitor: &[f64],
    y_pred: &[f64],
    monitor_dates: &[i64],
    level: f64,
) -> (f64, f64) {
    if y_monitor.is_empty() || y_monitor.len() != y_pred.len() {
        return (0.0, 0.0);
    }

    let residuals: Vec<f64> = y_monitor
        .iter()
        .zip(y_pred.iter())
        .map(|(obs, pred)| obs - pred)
        .collect();

    let mean_residual: f64 = residuals.iter().sum::<f64>() / residuals.len() as f64;

    // Simplified break detection: if the average residual in the monitoring period
    // exceeds the significance level, flag the start of the period as a break.
    if mean_residual.abs() > level {
        (monitor_dates[0] as f64, mean_residual.abs())
    } else {
        (0.0, 0.0) // No break detected
    }
}

// This is the main logic function that runs for each pixel.
fn run_bfast_monitor_per_pixel(
    pixel_ts: &[f64],
    dates: &[i64],
    history_start: i64,
    monitor_start: i64,
    level: f64,
) -> (f64, f64) {
    // 1. Find the indices for the history and monitoring periods
    let history_indices: Vec<usize> = dates
        .iter()
        .enumerate()
        .filter(|(_, &d)| d >= history_start && d < monitor_start)
        .map(|(i, _)| i)
        .collect();

    let monitor_indices: Vec<usize> = dates
        .iter()
        .enumerate()
        .filter(|(_, &d)| d >= monitor_start)
        .map(|(i, _)| i)
        .collect();

    if history_indices.is_empty() || monitor_indices.is_empty() {
        return (0.0, 0.0); // Not enough data
    }

    // 2. Extract the data for these periods
    let history_ts: Vec<f64> = history_indices.iter().map(|&i| pixel_ts[i]).collect();
    let monitor_ts: Vec<f64> = monitor_indices.iter().map(|&i| pixel_ts[i]).collect();
    let monitor_dates: Vec<i64> = monitor_indices.iter().map(|&i| dates[i]).collect();

    // 3. Fit model on the historical period
    let model = fit_harmonic_model(&history_ts);

    // 4. Predict for the monitoring period
    let predicted_ts = predict_harmonic_model(&model, &monitor_dates);

    // 5. Detect break using MOSUM process on residuals
    detect_mosum_break(&monitor_ts, &predicted_ts, &monitor_dates, level)
}

#[pyfunction]
pub fn bfast_monitor(
    py: Python,
    stack: PyReadonlyArrayDyn<f64>,
    dates: Vec<i64>,
    history_start_date: i64,
    monitor_start_date: i64,
    level: f64, // Significance level
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let stack_arr = stack.as_array();

    if stack_arr.ndim() != 3 {
        return Err(CoreError::InvalidArgument(format!(
            "Input stack must be 3-dimensional (Time, Y, X), but got {} dimensions",
            stack_arr.ndim()
        ))
        .into());
    }

    let time_len = stack_arr.shape()[0];
    let height = stack_arr.shape()[1];
    let width = stack_arr.shape()[2];

    if time_len != dates.len() {
        return Err(CoreError::InvalidArgument(format!(
            "Time dimension of stack ({}) does not match length of dates vector ({})",
            time_len,
            dates.len()
        ))
        .into());
    }

    // Output channels: [break_date, magnitude]
    let mut out_array = ndarray::ArrayD::<f64>::zeros(IxDyn(&[2, height, width]));

    // Flatten spatial dimensions for parallel processing
    let num_pixels = height * width;
    let stack_flat = stack_arr
        .into_shape((time_len, num_pixels))
        .map_err(|e| CoreError::ComputationError(e.to_string()))?;

    let mut out_flat = out_array
        .view_mut()
        .into_shape((2, num_pixels))
        .map_err(|e| CoreError::ComputationError(e.to_string()))?;

    // Get mutable 1D views for each output channel
    let mut out_slices = out_flat.axis_iter_mut(Axis(0));
    let mut break_dates = out_slices.next().unwrap();
    let mut magnitudes = out_slices.next().unwrap();

    // Iterate over each pixel's time series in parallel
    Zip::from(&mut break_dates)
        .and(&mut magnitudes)
        .and(stack_flat.axis_iter(Axis(1)))
        .par_for_each(|break_date, magnitude, pixel_ts| {
            let (bk_date, mag) = run_bfast_monitor_per_pixel(
                pixel_ts.as_slice().unwrap(),
                &dates,
                history_start_date,
                monitor_start_date,
                level,
            );
            *break_date = bk_date;
            *magnitude = mag;
        });

    Ok(out_array.into_pyarray(py).to_owned())
}

// --- 2. Short-Circuit Classifier Example ---

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn complex_classification(
    py: Python,
    blue: PyReadonlyArrayDyn<f64>,
    green: PyReadonlyArrayDyn<f64>,
    red: PyReadonlyArrayDyn<f64>,
    nir: PyReadonlyArrayDyn<f64>,
    swir1: PyReadonlyArrayDyn<f64>,
    swir2: PyReadonlyArrayDyn<f64>,
    temp: PyReadonlyArrayDyn<f64>,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    let blue_arr = blue.as_array();
    let green_arr = green.as_array();
    let red_arr = red.as_array();
    let nir_arr = nir.as_array();
    let swir1_arr = swir1.as_array();
    let swir2_arr = swir2.as_array();
    let temp_arr = temp.as_array();

    let mut out = ndarray::ArrayD::<u8>::zeros(blue_arr.raw_dim());

    out.indexed_iter_mut().par_bridge().for_each(|(idx, res)| {
        let b = blue_arr[&idx];
        let g = green_arr[&idx];
        let r = red_arr[&idx];
        let n = nir_arr[&idx];
        let s1 = swir1_arr[&idx];
        let s2 = swir2_arr[&idx];
        let t = temp_arr[&idx];
        *res = classify_pixel(b, g, r, n, s1, s2, t);
    });

    Ok(out.into_pyarray(py).to_owned())
}

// This function is where you gain 10x-50x speedups over NumPy
fn classify_pixel(b: f64, g: f64, r: f64, n: f64, s1: f64, s2: f64, t: f64) -> u8 {
    const EPSILON: f64 = 1e-10;

    // --- Class Definitions ---
    const UNCLASSIFIED: u8 = 0;
    const CLOUD_SHADOW: u8 = 1;
    const CLOUD: u8 = 2;
    const SNOW: u8 = 3;
    const WATER: u8 = 4;
    const VEGETATION: u8 = 5;
    const BARE_SOIL: u8 = 6;
    const URBAN: u8 = 7;

    // --- Pre-computation of Indices ---
    let ndvi = (n - r) / (n + r + EPSILON);
    let ndwi = (g - n) / (g + n + EPSILON);
    let ndsi = (g - s1) / (g + s1 + EPSILON);
    let brightness = (b + g + r + n + s1 + s2) / 6.0;

    // --- Rule-Based Classification Logic ---

    // 1. Cloud & Cloud Shadow Detection (using thermal and brightness)
    if t < 285.0 || brightness > 0.4 {
        if t < 280.0 && brightness < 0.1 {
            return CLOUD_SHADOW;
        }
        if b > 0.2 && g > 0.2 && r > 0.2 {
            return CLOUD;
        }
    }

    // 2. Snow/Ice Detection
    if ndsi > 0.4 && r > 0.15 && g > 0.2 {
        return SNOW;
    }

    // 3. Water Body Detection (multiple checks)
    if ndwi > 0.15 || (ndwi > 0.05 && n < 0.15) || (b > g && b > r) {
        return WATER;
    }

    // 4. Vegetation Detection
    if ndvi > 0.2 {
        return VEGETATION;
    }

    // 5. Bare Soil vs. Urban (using SWIR bands)
    // Bare soil reflects more in SWIR2 than SWIR1
    if s2 > s1 && (s1 - n) / (s1 + n + EPSILON) > 0.1 {
        return BARE_SOIL;
    }
    // Urban areas often have lower NDVI and similar SWIR reflectance
    if ndvi < 0.1 && (s1 - s2).abs() < 0.1 {
        return URBAN;
    }

    UNCLASSIFIED
}
