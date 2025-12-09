use ndarray::{s, ArrayViewD, Ix2, IxDyn, Zip};
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

    // We assume input is (Time, Y, X) and we want (3, Y, X) output
    // Output channels: [break_date, magnitude, confidence]
    let mut out_shape = stack_arr.shape().to_vec();
    if out_shape.len() != 3 {
        // For scaffolding, we'll proceed, but a real implementation would error here
    }
    out_shape[0] = 3;

    let mut out_array = ndarray::ArrayD::<f64>::zeros(IxDyn(&out_shape));

    // A real implementation would iterate over each pixel's time series in parallel.
    // This is complex to set up correctly with ndarray's zippers for a scaffold,
    // so we will use a simple, non-parallel loop for this placeholder.
    let height = stack_arr.shape()[1];
    let width = stack_arr.shape()[2];
    for y in 0..height {
        for x in 0..width {
            let pixel_ts = stack_arr.slice(s![.., y, x]);
            let (bk_date, mag, conf) = run_bfast_lite_logic(&pixel_ts, &dates, threshold);
            out_array[[0, y, x]] = bk_date;
            out_array[[1, y, x]] = mag;
            out_array[[2, y, x]] = conf;
        }
    }

    Ok(out_array.into_pyarray(py).to_owned())
}

// Pure Rust: The compiler optimizes this loop heavily.
fn run_bfast_lite_logic(
    pixel_ts: &ndarray::ArrayView1<f64>,
    dates: &[i64],
    thresh: f64,
) -> (f64, f64, f64) {
    if pixel_ts.len() <= 10 {
        return (-1.0, 0.0, 0.0);
    }

    let mut max_diff = 0.0;
    let mut break_idx = 0;

    for i in 5..(pixel_ts.len() - 5) {
        let (slope1, _) = simple_linreg(&pixel_ts.slice(s![..i]));
        let (slope2, _) = simple_linreg(&pixel_ts.slice(s![i..]));

        if (slope1 - slope2).abs() > max_diff {
            max_diff = (slope1 - slope2).abs();
            break_idx = i;
        }
    }

    if max_diff > thresh {
        (dates.get(break_idx).map_or(-1.0, |d| *d as f64), max_diff, 1.0)
    } else {
        (-1.0, 0.0, 0.0)
    }
}

// Stub for simple linear regression
fn simple_linreg(_y: &ndarray::ArrayView1<f64>) -> (f64, f64) {
    (0.0, 0.0) // Placeholder
}

// --- 2. Short-Circuit Classifier Example ---

#[pyfunction]
pub fn complex_classification(
    py: Python,
    blue: PyReadonlyArrayDyn<f64>,
    red: PyReadonlyArrayDyn<f64>,
    nir: PyReadonlyArrayDyn<f64>,
    swir: PyReadonlyArrayDyn<f64>,
    temp: PyReadonlyArrayDyn<f64>,
) -> PyResult<Py<PyArrayDyn<u8>>> {
    let blue_arr = blue.as_array();
    let red_arr = red.as_array();
    let nir_arr = nir.as_array();
    let swir_arr = swir.as_array();
    let temp_arr = temp.as_array();

    let mut out = ndarray::ArrayD::<u8>::zeros(blue_arr.raw_dim());

    Zip::from(&mut out)
        .and(&blue_arr)
        .and(&red_arr)
        .and(&nir_arr)
        .and(&swir_arr)
        .and(&temp_arr)
        .par_for_each(|res, &b, &r, &n, &s, &t| {
            *res = classify_pixel(b, r, n, s, t);
        });

    Ok(out.into_pyarray(py).to_owned())
}

// This function is where you gain 10x-50x speedups over NumPy
fn classify_pixel(b: f64, r: f64, n: f64, s: f64, t: f64) -> u8 {
    const EPSILON: f64 = 1e-10;

    // 1. Basic Cloud Check
    if t < 280.0 && (b + r) > 0.4 {
        return 1; // Cloud
    }

    // 2. Snow Check
    let ndsi = (r - s) / (r + s + EPSILON);
    if ndsi > 0.4 {
        return 2; // Snow
    }

    // 3. Water Check
    let ndwi = (r - n) / (r + n + EPSILON);
    if ndwi > 0.0 {
        return 3; // Water
    }

    0 // Land
}

// --- 3. Non-Linear Spatial Filter Example ---

#[pyfunction]
pub fn texture_entropy(
    py: Python,
    input: PyReadonlyArrayDyn<f64>,
    window_size: usize,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let arr = input.as_array();
    let mut out = ndarray::ArrayD::<f64>::zeros(arr.raw_dim());

    if arr.ndim() == 2 {
        let arr_2d = arr.into_dimensionality::<Ix2>().unwrap();
        let mut out_2d = out.view_mut().into_dimensionality::<Ix2>().unwrap();
        let y_len = arr_2d.shape()[0];
        let x_len = arr_2d.shape()[1];
        let r = window_size / 2;

        out_2d.indexed_iter_mut()
            .par_bridge()
            .for_each(|((y, x), val)| {
                if y >= r && y < y_len - r && x >= r && x < x_len - r {
                    let window = arr_2d.slice(s![y - r..=y + r, x - r..=x + r]);
                    *val = compute_entropy(&window.into_dyn());
                }
            });
    }

    Ok(out.into_pyarray(py).to_owned())
}

// Stub for entropy calculation
fn compute_entropy(_window: &ArrayViewD<f64>) -> f64 {
    // Bin values into histogram and calculate -sum(p * log(p))
    0.0 // Placeholder
}
