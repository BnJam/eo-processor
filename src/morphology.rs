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
    let (rows, cols) = input_arr.dim();
    let radius = (kernel_size / 2) as isize;

    // Parallel iteration using Rayon if possible, but let's start with serial for simplicity/correctness first.
    // Actually, let's use Rayon immediately as it's easy with ndarray parallel iterators if we use Zip,
    // but random write is hard.
    // We can use `par_map_axis` or just iterate rows in parallel.
    // For now, serial implementation to ensure correctness of bounds logic.

    // Optimization: If kernel is 3x3, unroll?
    // General case:

    // We iterate over every pixel.
    // If input[x, y] is 1, it "hits" the kernel.
    // Dilation: Output is 1 if ANY pixel in the kernel window around it is 1.
    // Equivalent: If input[x, y] is 1, set the kernel window in output to 1.
    // BUT that requires atomic writes for parallel.
    // Better: For each output pixel, check if ANY input pixel in its window is 1.

    // Let's do the "For each output pixel" approach (Gather).
    // output[r, c] = max(input[r+dr, c+dc]) for dr, dc in kernel.

    // We can parallelize over rows.
    use rayon::prelude::*;

    // We can't easily use slices with 2D logic without stride math.
    // Let's stick to indexed iteration.

    // Naive implementation with bounds checking
    // To optimize, we can separate the "inner" part (no bounds check) from "border" part.
    // But for now, let's just implement correct logic.

    let mut out_vec = vec![0u8; rows * cols];

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

                        if nr >= 0 && nr < rows as isize && nc >= 0 && nc < cols as isize {
                            // Safe index
                            if input_arr[[nr as usize, nc as usize]] > 0 {
                                hit = true;
                                break 'kernel;
                            }
                        }
                    }
                }
                *out = if hit { 1 } else { 0 };
            }
        });

    let out_arr = Array2::from_shape_vec((rows, cols), out_vec).unwrap();
    Ok(out_arr.to_pyarray(py).into())
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
    let (rows, cols) = input_arr.dim();
    let radius = (kernel_size / 2) as isize;

    let mut out_vec = vec![0u8; rows * cols];

    // Erosion: Output is 1 if ALL pixels in the kernel window are 1.
    // If input[r,c] is 0, output is definitely 0 (assuming kernel includes center).
    // Optimization: Only check neighbors if center is 1.

    use rayon::prelude::*;
    out_vec
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(r, row_slice)| {
            for (c, out) in row_slice.iter_mut().enumerate().take(cols) {
                // If center is 0, erosion is 0 (assuming structural element contains origin)
                if input_arr[[r, c]] == 0 {
                    *out = 0;
                    continue;
                }

                let mut all_hit = true;
                'kernel: for kr in -radius..=radius {
                    for kc in -radius..=radius {
                        let nr = r as isize + kr;
                        let nc = c as isize + kc;

                        if nr < 0 || nr >= rows as isize || nc < 0 || nc >= cols as isize || input_arr[[nr as usize, nc as usize]] == 0 {
                            all_hit = false;
                            break 'kernel;
                        }
                    }
                }
                *out = if all_hit { 1 } else { 0 };
            }
        });

    let out_arr = Array2::from_shape_vec((rows, cols), out_vec).unwrap();
    Ok(out_arr.to_pyarray(py).into())
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
                        if nr >= 0 && nr < rows as isize && nc >= 0 && nc < cols as isize &&
                            input[[nr as usize, nc as usize]] > 0 {
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

                        if nr < 0 || nr >= rows as isize || nc < 0 || nc >= cols as isize || input[[nr as usize, nc as usize]] == 0 {
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
