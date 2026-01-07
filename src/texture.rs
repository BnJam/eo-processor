use ndarray::{s, Array2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

type HaralickPyResult = PyResult<(
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
)>;

// Calculates the Gray-Level Co-occurrence Matrix (GLCM) for a given window.
fn glcm(window: &Array2<u8>, levels: u8, dx: isize, dy: isize) -> Array2<f64> {
    let (height, width) = (window.dim().0, window.dim().1);
    let mut glcm = Array2::<f64>::zeros((levels as usize, levels as usize));

    for y in 0..height {
        for x in 0..width {
            let next_y = y as isize + dy;
            let next_x = x as isize + dx;

            if next_y >= 0 && next_y < height as isize && next_x >= 0 && next_x < width as isize {
                let i = window[[y, x]] as usize;
                let j = window[[next_y as usize, next_x as usize]] as usize;
                if i < levels as usize && j < levels as usize {
                    glcm[[i, j]] += 1.0;
                }
            }
        }
    }
    glcm
}

// Normalizes the GLCM by dividing by the sum of its elements.
fn normalize_glcm(glcm: &mut Array2<f64>) {
    let sum = glcm.sum();
    if sum > 0.0 {
        *glcm /= sum;
    }
}

// Calculates the 'contrast' Haralick feature.
fn contrast(glcm: &Array2<f64>) -> f64 {
    let mut contrast = 0.0;
    for ((i, j), &p) in glcm.indexed_iter() {
        contrast += (i as f64 - j as f64).powi(2) * p;
    }
    contrast
}

// Calculates the 'dissimilarity' Haralick feature.
fn dissimilarity(glcm: &Array2<f64>) -> f64 {
    let mut dissimilarity = 0.0;
    for ((i, j), &p) in glcm.indexed_iter() {
        dissimilarity += (i as f64 - j as f64).abs() * p;
    }
    dissimilarity
}

// Calculates the 'homogeneity' Haralick feature.
fn homogeneity(glcm: &Array2<f64>) -> f64 {
    let mut homogeneity = 0.0;
    for ((i, j), &p) in glcm.indexed_iter() {
        homogeneity += p / (1.0 + (i as f64 - j as f64).powi(2));
    }
    homogeneity
}

// Calculates the 'entropy' Haralick feature.
fn entropy(glcm: &Array2<f64>) -> f64 {
    let mut entropy = 0.0;
    for &p in glcm.iter() {
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }
    entropy
}

// Helper function to calculate all Haralick features for a single window.
fn calculate_features_for_window(window: &Array2<u8>, levels: u8) -> (f64, f64, f64, f64) {
    // Offsets matching scikit-image: [dist, angle]
    // 0 deg: (0, 1), 45 deg: (-1, 1), 90 deg: (-1, 0), 135 deg: (-1, -1)
    let offsets = [(0, 1), (-1, 1), (-1, 0), (-1, -1)];

    let mut total_contrast = 0.0;
    let mut total_dissimilarity = 0.0;
    let mut total_homogeneity = 0.0;
    let mut total_entropy = 0.0;
    let count = offsets.len() as f64;

    for &(dy, dx) in &offsets {
        let mut glcm_matrix = glcm(window, levels, dx, dy);
        let t_glcm = glcm_matrix.t();
        glcm_matrix = &glcm_matrix + &t_glcm; // Symmetrize
        normalize_glcm(&mut glcm_matrix);

        total_contrast += contrast(&glcm_matrix);
        total_dissimilarity += dissimilarity(&glcm_matrix);
        total_homogeneity += homogeneity(&glcm_matrix);
        total_entropy += entropy(&glcm_matrix);
    }

    (
        total_contrast / count,
        total_dissimilarity / count,
        total_homogeneity / count,
        total_entropy / count,
    )
}

/// Applies a sliding window over a 2D array and calculates Haralick texture features.
///
/// :param arr: 2D NumPy array of unsigned 8-bit integers.
/// :param window_size: The size of the square window.
/// :param levels: The number of gray levels for quantization.
/// :return: A tuple of 4 2D NumPy arrays: (contrast, dissimilarity, homogeneity, entropy).
#[pyfunction]
#[pyo3(name = "haralick_features")]
pub fn haralick_features_py(
    py: Python<'_>,
    arr: PyReadonlyArray2<u8>,
    window_size: usize,
    levels: u8,
) -> HaralickPyResult {
    let array = arr.as_array().to_owned();
    let (height, width) = (array.shape()[0], array.shape()[1]);
    let half_window = window_size / 2;

    let (contrast_out, dissimilarity_out, homogeneity_out, entropy_out) =
        py.allow_threads(move || {
            let mut contrast_out = Array2::<f64>::zeros((height, width));
            let mut dissimilarity_out = Array2::<f64>::zeros((height, width));
            let mut homogeneity_out = Array2::<f64>::zeros((height, width));
            let mut entropy_out = Array2::<f64>::zeros((height, width));

            let pixels: Vec<(usize, usize)> = (0..height)
                .flat_map(|r| (0..width).map(move |c| (r, c)))
                .collect();

            let results: Vec<(f64, f64, f64, f64)> = pixels
                .par_iter()
                .map(|&(r, c)| {
                    let r_min = r.saturating_sub(half_window);
                    let r_max = (r + half_window).min(height - 1);
                    let c_min = c.saturating_sub(half_window);
                    let c_max = (c + half_window).min(width - 1);

                    let window = array.slice(s![r_min..=r_max, c_min..=c_max]).to_owned();
                    calculate_features_for_window(&window, levels)
                })
                .collect();

            for (i, (con, dis, hom, ent)) in results.into_iter().enumerate() {
                let r = i / width;
                let c = i % width;
                contrast_out[[r, c]] = con;
                dissimilarity_out[[r, c]] = dis;
                homogeneity_out[[r, c]] = hom;
                entropy_out[[r, c]] = ent;
            }

            (
                contrast_out,
                dissimilarity_out,
                homogeneity_out,
                entropy_out,
            )
        });

    Ok((
        contrast_out.into_pyarray(py).to_owned(),
        dissimilarity_out.into_pyarray(py).to_owned(),
        homogeneity_out.into_pyarray(py).to_owned(),
        entropy_out.into_pyarray(py).to_owned(),
    ))
}
