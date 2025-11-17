pub mod indices;
pub mod masking;
pub mod processes;
pub mod spatial;
pub mod temporal;
pub mod trends;

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
    m.add_function(wrap_pyfunction!(indices::savi, m)?)?;
    m.add_function(wrap_pyfunction!(indices::nbr, m)?)?;
    // Additional spectral indices
    m.add_function(wrap_pyfunction!(indices::ndmi, m)?)?;
    m.add_function(wrap_pyfunction!(indices::nbr2, m)?)?;
    m.add_function(wrap_pyfunction!(indices::gci, m)?)?;
    // --- Change Detection Indices ---
    m.add_function(wrap_pyfunction!(indices::delta_ndvi, m)?)?;
    m.add_function(wrap_pyfunction!(indices::delta_nbr, m)?)?;

    // --- Spatial Distance & Aggregation Functions ---
    m.add_function(wrap_pyfunction!(spatial::euclidean_distance, m)?)?;
    m.add_function(wrap_pyfunction!(spatial::manhattan_distance, m)?)?;
    m.add_function(wrap_pyfunction!(spatial::chebyshev_distance, m)?)?;
    m.add_function(wrap_pyfunction!(spatial::minkowski_distance, m)?)?;
    m.add_function(wrap_pyfunction!(spatial::median, m)?)?;

    // --- Temporal Functions ---
    m.add_function(wrap_pyfunction!(temporal::temporal_mean, m)?)?;
    m.add_function(wrap_pyfunction!(temporal::temporal_std, m)?)?;
    m.add_function(wrap_pyfunction!(temporal::temporal_sum, m)?)?;
    // --- Masking Functions ---
    m.add_function(wrap_pyfunction!(masking::mask_vals, m)?)?;
    m.add_function(wrap_pyfunction!(masking::replace_nans, m)?)?;
    m.add_function(wrap_pyfunction!(masking::mask_out_range, m)?)?;
    m.add_function(wrap_pyfunction!(masking::mask_invalid, m)?)?;
    m.add_function(wrap_pyfunction!(masking::mask_in_range, m)?)?;
    m.add_function(wrap_pyfunction!(masking::mask_scl, m)?)?;
    // --- Advanced Processes ---
    m.add_function(wrap_pyfunction!(processes::moving_average_temporal, m)?)?;
    m.add_function(wrap_pyfunction!(
        processes::moving_average_temporal_stride,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(processes::pixelwise_transform, m)?)?;
    m.add_function(wrap_pyfunction!(processes::temporal_composite, m)?)?;

    // --- Trend Analysis ---
    m.add_class::<trends::TrendSegment>()?;
    m.add_function(wrap_pyfunction!(trends::trend_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(trends::linear_regression, m)?)?;

    Ok(())
}
