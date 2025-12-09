use ndarray::{Array2, ArrayView2};
use numpy::{PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

/// Perform binary dilation on a 2D boolean/int array.
///
/// # Arguments
/// * `input` - 2D input array (treated as boolean: >0 is True).
/// * `kernel_size` - Size of the square structuring element (default 3).
///
/// # Returns
/// Dilated 2D array (uint8: 0 or 1).
#[pyfunction]
#[pyo3(signature = (input, kernel_size=3))]
pub fn binary_dilation(
    py: Python<'_>,
    input: PyReadonlyArray2<u8>,
    kernel_size: usize,
) -> PyResult<PyObject> {
    let input_arr = input.as_array();
    let dilated = dilation_impl(input_arr, kernel_size);
    Ok(dilated.to_pyarray(py).into())
}

/// Perform binary erosion on a 2D boolean/int array.
#[pyfunction]
#[pyo3(signature = (input, kernel_size=3))]
pub fn binary_erosion(
    py: Python<'_>,
    input: PyReadonlyArray2<u8>,
    kernel_size: usize,
) -> PyResult<PyObject> {
    let input_arr = input.as_array();
    let eroded = erosion_impl(input_arr, kernel_size);
    Ok(eroded.to_pyarray(py).into())
}

/// Perform binary opening (erosion followed by dilation).
#[pyfunction]
#[pyo3(signature = (input, kernel_size=3))]
pub fn binary_opening(
    py: Python<'_>,
    input: PyReadonlyArray2<u8>,
    kernel_size: usize,
) -> PyResult<PyObject> {
    // We can't easily compose PyReadonlyArray2 without converting back and forth or refactoring logic.
    // Refactoring logic to pure Rust functions is better.

    let input_arr = input.as_array();
    let eroded = erosion_impl(input_arr, kernel_size);
    let dilated = dilation_impl(eroded.view(), kernel_size);

    Ok(dilated.to_pyarray(py).into())
}

/// Perform binary closing (dilation followed by erosion).
#[pyfunction]
#[pyo3(signature = (input, kernel_size=3))]
pub fn binary_closing(
    py: Python<'_>,
    input: PyReadonlyArray2<u8>,
    kernel_size: usize,
) -> PyResult<PyObject> {
    let input_arr = input.as_array();
    let dilated = dilation_impl(input_arr, kernel_size);
    let eroded = erosion_impl(dilated.view(), kernel_size);

    Ok(eroded.to_pyarray(py).into())
}

// Pure Rust implementations for composition
fn dilation_impl(input: ArrayView2<u8>, kernel_size: usize) -> Array2<u8> {
    let (rows, cols) = input.dim();
    let radius = (kernel_size / 2) as isize;
    let mut out_vec = vec![0u8; rows * cols];

    use rayon::prelude::*;
    out_vec
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(r, row_slice)| {
            for (c, out) in row_slice.iter_mut().enumerate().take(cols) {
                let mut hit = false;
                'kernel: for kr in -radius..=radius {
                    for kc in -radius..=radius {
                        let nr = r as isize + kr;
                        let nc = c as isize + kc;
                        if nr >= 0
                            && nr < rows as isize
                            && nc >= 0
                            && nc < cols as isize
                            && input[[nr as usize, nc as usize]] > 0
                        {
                            hit = true;
                            break 'kernel;
                        }
                    }
                }
                *out = if hit { 1 } else { 0 };
            }
        });
    Array2::from_shape_vec((rows, cols), out_vec).unwrap()
}

fn erosion_impl(input: ArrayView2<u8>, kernel_size: usize) -> Array2<u8> {
    let (rows, cols) = input.dim();
    let radius = (kernel_size / 2) as isize;
    let mut out_vec = vec![0u8; rows * cols];

    use rayon::prelude::*;
    out_vec
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(r, row_slice)| {
            for (c, out) in row_slice.iter_mut().enumerate().take(cols) {
                if input[[r, c]] == 0 {
                    *out = 0;
                    continue;
                }
                let mut all_hit = true;
                'kernel: for kr in -radius..=radius {
                    for kc in -radius..=radius {
                        let nr = r as isize + kr;
                        let nc = c as isize + kc;

                        if nr < 0
                            || nr >= rows as isize
                            || nc < 0
                            || nc >= cols as isize
                            || input[[nr as usize, nc as usize]] == 0
                        {
                            all_hit = false;
                            break 'kernel;
                        }
                    }
                }
                *out = if all_hit { 1 } else { 0 };
            }
        });
    Array2::from_shape_vec((rows, cols), out_vec).unwrap()
}
