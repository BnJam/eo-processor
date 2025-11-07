// Bring in the functionality from sub-modules
pub mod indices;
pub mod spatial;
pub mod temporal;

use pyo3::prelude::*;

/// Python module for high-performance Earth Observation processing.
///
/// This module provides Rust-accelerated functions for common EO computations
/// that can be used with XArray/Dask workflows to bypass Python's GIL.
#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    // --- Spectral Indices ---
    m.add_function(wrap_pyfunction!(indices::normalized_difference_1d, m)?)?;
    m.add_function(wrap_pyfunction!(indices::normalized_difference_2d, m)?)?;
    m.add_function(wrap_pyfunction!(indices::normalized_difference_3d, m)?)?;
    m.add_function(wrap_pyfunction!(indices::ndvi_1d, m)?)?;
    m.add_function(wrap_pyfunction!(indices::ndvi_2d, m)?)?;
    m.add_function(wrap_pyfunction!(indices::ndvi_3d, m)?)?;
    m.add_function(wrap_pyfunction!(indices::ndwi_1d, m)?)?;
    m.add_function(wrap_pyfunction!(indices::ndwi_2d, m)?)?;
    m.add_function(wrap_pyfunction!(indices::enhanced_vegetation_index_1d, m)?)?;

    // --- Spatial Functions ---
    m.add_function(wrap_pyfunction!(spatial::median_filter_2d, m)?)?;

    // --- Temporal Functions ---
    m.add_function(wrap_pyfunction!(temporal::temporal_mean_1d, m)?)?;

    Ok(())
}
