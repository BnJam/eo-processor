use crate::CoreError;
use nalgebra::{DMatrix, DVector};
use ndarray::{Axis, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use rayon::prelude::*;

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

// Chunk size (in pixels) for parallel work units in bfast_monitor.
// Larger chunks reduce scheduling overhead and enable safe disjoint mutable writes.
const BFAST_PIXELS_PER_CHUNK: usize = 1024;

// --- 1. BFAST Monitor Workflow ---

/// Represents the fitted harmonic model.
struct HarmonicModel {
    coefficients: DVector<f64>,
    sigma: f64,
}

/// Constructs the design matrix for a harmonic model.
///
/// # Arguments
///
/// * `dates` - A slice of fractional years.
/// * `order` - The order of the harmonic model (e.g., 1 for one sine/cosine pair).
///
/// # Returns
///
/// A 2D array representing the design matrix `X`.
fn build_design_matrix(dates: &[f64], order: usize) -> DMatrix<f64> {
    let n = dates.len();
    let num_coeffs = 2 * order + 2; // intercept, trend, and sin/cos pairs
    let mut x = DMatrix::<f64>::zeros(n, num_coeffs);

    for i in 0..n {
        let t = dates[i];
        x[(i, 0)] = 1.0; // Intercept
        x[(i, 1)] = t;   // Trend
        for j in 1..=order {
            let freq = TWO_PI * j as f64 * t;
            x[(i, 2 * j)] = freq.cos();
            x[(i, 2 * j + 1)] = freq.sin();
        }
    }
    x
}

/// Fits a harmonic model to the stable history period using Ordinary Least Squares (OLS).
fn fit_harmonic_model(y: &[f64], dates: &[f64], order: usize) -> Result<HarmonicModel, CoreError> {
    if y.len() < (2 * order + 2) {
        return Err(CoreError::NotEnoughData(
            "Not enough historical data to fit model".to_string(),
        ));
    }

    let y_vec = DVector::from_vec(y.to_vec());
    let x = build_design_matrix(dates, order);

    let decomp = x.clone().svd(true, true);
    let coeffs = decomp.solve(&y_vec, 1e-10).map_err(|e| {
        CoreError::ComputationError(format!("Failed to solve OLS with nalgebra: {}", e))
    })?;

    let y_pred = &x * &coeffs;
    let residuals = &y_vec - &y_pred;
    let sum_sq_err = residuals.iter().map(|&r| r * r).sum::<f64>();
    let df = (y.len() - (2 * order + 2)) as f64;
    if df <= 0.0 {
        return Err(CoreError::ComputationError(
            "Degrees of freedom is non-positive".to_string(),
        ));
    }
    let sigma = (sum_sq_err / df).sqrt();

    Ok(HarmonicModel {
        coefficients: coeffs,
        sigma,
    })
}

/// Predicts values for the monitoring period based on the fitted model.
fn predict_harmonic_model(model: &HarmonicModel, dates: &[f64], order: usize) -> DVector<f64> {
    let x_mon = build_design_matrix(dates, order);
    &x_mon * &model.coefficients
}

/// Detects a break using the OLS-MOSUM process.
fn detect_mosum_break(
    y_monitor: &[f64],
    y_pred: &DVector<f64>,
    monitor_dates: &[f64],
    hist_len: usize,
    sigma: f64,
    h: f64,
    alpha: f64,
) -> (f64, f64) {
    if y_monitor.is_empty() {
        return (0.0, 0.0);
    }

    let n_hist = hist_len as f64;
    let window_size = (h * n_hist).floor() as usize;

    let residuals: Vec<f64> = y_monitor
        .iter()
        .zip(y_pred.iter())
        .map(|(obs, pred)| obs - pred)
        .collect();

    let mut cusum = vec![0.0; residuals.len() + 1];
    for i in 0..residuals.len() {
        cusum[i + 1] = cusum[i] + residuals[i];
    }

    // We can only start calculating MOSUM after `window_size` observations
    if residuals.len() < window_size {
        return (0.0, 0.0);
    }

    let mosum_process: Vec<f64> = (window_size..residuals.len())
        .map(|i| cusum[i] - cusum[i - window_size])
        .collect();

    let standardizer = sigma * n_hist.sqrt();
    let standardized_mosum: Vec<f64> = mosum_process
        .iter()
        .map(|&m| (m / standardizer).abs())
        .collect();

    // Simplified critical boundary based on a lookup for alpha=0.05 and h=0.25
    // A full implementation would use a precomputed table or a more complex calculation.
    let critical_value = if alpha <= 0.05 { 1.36 } else { 1.63 }; // Approximations

    for (i, &mosum_val) in standardized_mosum.iter().enumerate() {
        // The index k starts from 1 for the monitoring period
        let k = (i + 1) as f64;
        let boundary = critical_value * (1.0 + k / n_hist).sqrt();

        if mosum_val > boundary {
            let break_idx = i + window_size;
            let magnitude = (y_monitor[break_idx] - y_pred[break_idx]).abs();
            return (monitor_dates[break_idx], magnitude);
        }
    }

    (0.0, 0.0) // No break detected
}

/// Converts integer dates (YYYYMMDD) to fractional years.
fn dates_to_frac_years(dates: &[i64]) -> Vec<f64> {
    dates
        .iter()
        .map(|&date| {
            let year = (date / 10000) as f64;
            let month = ((date % 10000) / 100) as f64;
            let day = (date % 100) as f64;
            // Simple approximation
            year + (month - 1.0) / 12.0 + (day - 1.0) / 365.25
        })
        .collect()
}

// This is the main logic function that runs for each pixel.


fn run_bfast_monitor_per_pixel_windows(
    history_ts: &[f64],
    history_dates: &[f64],
    monitor_ts: &[f64],
    monitor_dates: &[f64],
    order: usize,
    h: f64,
    alpha: f64,
) -> (f64, f64) {
    // Fit model on the historical period
    let model = match fit_harmonic_model(history_ts, history_dates, order) {
        Ok(m) => m,
        Err(_) => return (0.0, 0.0),
    };

    // Predict for the monitoring period
    let predicted_ts = predict_harmonic_model(&model, monitor_dates, order);

    // Detect break using MOSUM process on residuals
    detect_mosum_break(
        monitor_ts,
        &predicted_ts,
        monitor_dates,
        history_ts.len(),
        model.sigma,
        h,
        alpha,
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn bfast_monitor(
    py: Python,
    stack: PyReadonlyArrayDyn<f64>,
    dates: Vec<i64>,
    history_start_date: i64,
    monitor_start_date: i64,
    order: usize,
    h: f64,
    alpha: f64,
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

    // Convert integer dates to fractional years for modeling
    let frac_dates = dates_to_frac_years(&dates);
    let history_start_frac = dates_to_frac_years(&[history_start_date])[0];
    let monitor_start_frac = dates_to_frac_years(&[monitor_start_date])[0];

    // Output channels: [break_date, magnitude]
    let mut out_array = ndarray::ArrayD::<f64>::zeros(IxDyn(&[2, height, width]));

    // Flatten spatial dimensions for parallel processing
    //
    // IMPORTANT: In the input layout (time, y, x), each pixel's time series is strided
    // along the time axis. After reshaping to (time, num_pixels), iterating over Axis(1)
    // yields 1D views that are typically *not* contiguous, so `as_slice()` can return None.
    //
    // To avoid per-pixel allocations (and avoid panics), we instead fetch each time sample
    // by indexing the (time, num_pixels) array.
    let num_pixels = height * width;
    let stack_flat = stack_arr
        .into_shape((time_len, num_pixels))
        .map_err(|e| CoreError::ComputationError(e.to_string()))?;

    // Precompute the (time) indices for history/monitor windows once (they are the same
    // for all pixels), which avoids collecting per-pixel vectors in the hot loop.
    let history_indices: Vec<usize> = frac_dates
        .iter()
        .enumerate()
        .filter(|(_, &d)| d >= history_start_frac && d < monitor_start_frac)
        .map(|(i, _)| i)
        .collect();

    let monitor_indices: Vec<usize> = frac_dates
        .iter()
        .enumerate()
        .filter(|(_, &d)| d >= monitor_start_frac)
        .map(|(i, _)| i)
        .collect();

    if history_indices.is_empty() || monitor_indices.is_empty() {
        // No valid windows: output stays zeros (no break, no magnitude)
        return Ok(out_array.into_pyarray(py).to_owned());
    }

    let mut out_flat = out_array
        .view_mut()
        .into_shape((2, num_pixels))
        .map_err(|e| CoreError::ComputationError(e.to_string()))?;

    // Get mutable 1D views for each output channel (avoid panics; return Python-friendly errors)
    let mut out_slices = out_flat.axis_iter_mut(Axis(0));

    let mut break_dates = out_slices.next().ok_or_else(|| {
        CoreError::ComputationError(
            "Internal error: failed to create break_dates output view (missing channel 0)".to_string(),
        )
    })?;

    let mut magnitudes = out_slices.next().ok_or_else(|| {
        CoreError::ComputationError(
            "Internal error: failed to create magnitudes output view (missing channel 1)".to_string(),
        )
    })?;

    // NOTE: Pixel time series for a (time, y, x) input are strided, so we avoid relying on
    // `pixel_ts.as_slice()` entirely. Instead, we index into `stack_flat[(t, pix)]`.
    //
    // To write outputs in parallel safely, we:
    // 1) ensure each output channel is a contiguous 1D slice
    // 2) split each output slice into disjoint chunks
    // 3) process chunks in parallel, writing only within the current chunk
    let break_dates_slice = break_dates
        .as_slice_mut()
        .ok_or_else(|| CoreError::ComputationError("break_dates output buffer is not contiguous".to_string()))?;
    let magnitudes_slice = magnitudes
        .as_slice_mut()
        .ok_or_else(|| CoreError::ComputationError("magnitudes output buffer is not contiguous".to_string()))?;

    // Parallelize over disjoint mutable chunks for both outputs.
    break_dates_slice
        .par_chunks_mut(BFAST_PIXELS_PER_CHUNK)
        .zip(magnitudes_slice.par_chunks_mut(BFAST_PIXELS_PER_CHUNK))
        .enumerate()
        .for_each(|(chunk_idx, (break_chunk, mag_chunk))| {
            let start = chunk_idx * BFAST_PIXELS_PER_CHUNK;
            let end = start + break_chunk.len(); // last chunk may be shorter

            // Reusable buffers for this worker/chunk.
            let mut history_ts: Vec<f64> = Vec::with_capacity(history_indices.len());
            let mut history_dates: Vec<f64> = Vec::with_capacity(history_indices.len());
            let mut monitor_ts: Vec<f64> = Vec::with_capacity(monitor_indices.len());
            let mut monitor_dates: Vec<f64> = Vec::with_capacity(monitor_indices.len());

            for pix in start..end {
                history_ts.clear();
                history_dates.clear();
                monitor_ts.clear();
                monitor_dates.clear();

                // Fill history buffers
                for &ti in &history_indices {
                    history_ts.push(stack_flat[(ti, pix)]);
                    history_dates.push(frac_dates[ti]);
                }

                // Fill monitoring buffers
                for &ti in &monitor_indices {
                    monitor_ts.push(stack_flat[(ti, pix)]);
                    monitor_dates.push(frac_dates[ti]);
                }

                let (bk_date, mag) = run_bfast_monitor_per_pixel_windows(
                    &history_ts,
                    &history_dates,
                    &monitor_ts,
                    &monitor_dates,
                    order,
                    h,
                    alpha,
                );

                let local = pix - start;
                break_chunk[local] = bk_date;
                mag_chunk[local] = mag;
            }
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

    out.indexed_iter_mut()
        .par_bridge()
        .for_each(|(idx, res)| {
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
