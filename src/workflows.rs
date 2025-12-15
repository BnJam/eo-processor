use crate::CoreError;
use ndarray::{Axis, IxDyn, Zip};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use rayon::prelude::*;

// --- 1. Iterative Time-Series Fitter Example ---

#[pyfunction]
pub fn detect_breakpoints(
    py: Python,
    stack: PyReadonlyArrayDyn<f64>,
    dates: Vec<i64>, // Julian dates
    threshold: f64,
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

    // Output channels: [break_date, magnitude, confidence]
    let mut out_array = ndarray::ArrayD::<f64>::zeros(IxDyn(&[3, height, width]));

    // Flatten spatial dimensions for parallel processing
    let num_pixels = height * width;
    let stack_flat = stack_arr
        .into_shape((time_len, num_pixels))
        .map_err(|e| CoreError::ComputationError(e.to_string()))?;

    let mut out_flat = out_array
        .view_mut()
        .into_shape((3, num_pixels))
        .map_err(|e| CoreError::ComputationError(e.to_string()))?;

    // Get mutable 1D views for each output channel
    let mut out_slices = out_flat.axis_iter_mut(Axis(0));
    let mut break_dates = out_slices.next().unwrap();
    let mut magnitudes = out_slices.next().unwrap();
    let mut confidences = out_slices.next().unwrap();

    // Iterate over each pixel's time series in parallel
    Zip::from(&mut break_dates)
        .and(&mut magnitudes)
        .and(&mut confidences)
        .and(stack_flat.axis_iter(Axis(1)))
        .par_for_each(|break_date, magnitude, confidence, pixel_ts| {
            let (bk_date, mag, conf) =
                run_bfast_lite_logic(pixel_ts.as_slice().unwrap(), &dates, threshold);
            *break_date = bk_date;
            *magnitude = mag;
            *confidence = conf;
        });

    Ok(out_array.into_pyarray(py).to_owned())
}

// Pure Rust: The compiler optimizes this loop heavily.
fn run_bfast_lite_logic(pixel_ts: &[f64], dates: &[i64], thresh: f64) -> (f64, f64, f64) {
    if pixel_ts.len() <= 10 {
        return (-1.0, 0.0, 0.0);
    }

    let mut max_diff = 0.0;
    let mut break_idx = 0;

    // Iterate through possible breakpoints, ensuring enough data on each side
    for i in 5..(pixel_ts.len() - 5) {
        let (slope1, _) = calculate_linear_regression(&pixel_ts[..i]);
        let (slope2, _) = calculate_linear_regression(&pixel_ts[i..]);

        let diff = (slope1 - slope2).abs();
        if diff > max_diff {
            max_diff = diff;
            break_idx = i;
        }
    }

    if max_diff > thresh {
        (
            dates.get(break_idx).map_or(-1.0, |d| *d as f64),
            max_diff,
            1.0, // Confidence is simplified to 1.0 if a break is found
        )
    } else {
        (-1.0, 0.0, 0.0)
    }
}

// Local helper for linear regression, adapted from src/trends.rs
fn calculate_linear_regression(y: &[f64]) -> (f64, f64) {
    if y.is_empty() {
        return (0.0, 0.0);
    }
    let n = y.len() as f64;
    let x_sum: f64 = (0..y.len()).map(|i| i as f64).sum();
    let y_sum: f64 = y.iter().sum();
    let xy_sum: f64 = y.iter().enumerate().map(|(i, &yi)| i as f64 * yi).sum();
    let x_sq_sum: f64 = (0..y.len()).map(|i| (i as f64).powi(2)).sum();

    let denominator = n * x_sq_sum - x_sum.powi(2);
    if denominator.abs() < 1e-10 {
        return (0.0, y.iter().sum::<f64>() / n); // Vertical line, return mean as intercept
    }

    let slope = (n * xy_sum - x_sum * y_sum) / denominator;
    let intercept = (y_sum - slope * x_sum) / n;

    (slope, intercept)
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
