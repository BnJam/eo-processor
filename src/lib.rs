pub mod indices;
pub mod spatial;

use pyo3::prelude::*;

/// Python module for high-performance Earth Observation processing.
///
/// This module provides Rust-accelerated functions for common EO computations
/// that can be used with XArray/Dask workflows to bypass Python's GIL.
#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    // --- Spectral Indices ---
    m.add_function(wrap_pyfunction!(indices::normalized_difference, m)?)?;
    m.add_function(wrap_pyfunction!(indices::ndvi, m)?)?;
    m.add_function(wrap_pyfunction!(indices::ndwi, m)?)?;
    m.add_function(wrap_pyfunction!(indices::enhanced_vegetation_index, m)?)?;

    // --- Spatial Functions (Future) ---
    m.add_function(wrap_pyfunction!(spatial::euclidean_distance, m)?)?;
    m.add_function(wrap_pyfunction!(spatial::manhattan_distance, m)?)?;
    m.add_function(wrap_pyfunction!(spatial::chebyshev_distance, m)?)?;
    m.add_function(wrap_pyfunction!(spatial::median, m)?)?;

    Ok(())
}
