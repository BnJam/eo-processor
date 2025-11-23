use crate::CoreError;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::prelude::*;
use std::collections::HashMap;

/// struct to hold aggregated statistics for a single zone.
#[pyclass]
#[derive(Debug, Clone, Copy)]
pub struct ZoneStats {
    #[pyo3(get)]
    pub count: usize,
    #[pyo3(get)]
    pub sum: f64,
    #[pyo3(get)]
    pub mean: f64,
    #[pyo3(get)]
    pub min: f64,
    #[pyo3(get)]
    pub max: f64,
    #[pyo3(get)]
    pub std: f64,
}

// Optimized Accumulator using Sum of Squares (faster than Welford, slightly less stable)
struct SumSqAccumulator {
    count: usize,
    sum: f64,
    sum_sq: f64,
    min: f64,
    max: f64,
}

impl SumSqAccumulator {
    fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    #[inline(always)]
    fn update(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_sq += value * value;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    fn to_stats(&self) -> ZoneStats {
        let mean = if self.count > 0 {
            self.sum / self.count as f64
        } else {
            f64::NAN
        };
        let std = if self.count < 2 {
            0.0
        } else {
            let variance = (self.sum_sq - (self.sum * self.sum) / self.count as f64)
                / (self.count as f64 - 1.0);
            variance.max(0.0).sqrt()
        };

        ZoneStats {
            count: self.count,
            sum: self.sum,
            mean,
            min: if self.count > 0 { self.min } else { f64::NAN },
            max: if self.count > 0 { self.max } else { f64::NAN },
            std,
        }
    }
}

/// Calculate zonal statistics.
///
/// # Arguments
/// * `values` - Input value array (any numeric dtype, coerced to float64).
/// * `zones` - Input zone label array (must be broadcastable to values, coerced to int64).
///
/// # Returns
/// Dictionary mapping zone ID (int) to ZoneStats object.
#[pyfunction]
pub fn zonal_stats(
    _py: Python<'_>,
    values: &PyAny,
    zones: &PyAny,
) -> PyResult<HashMap<i64, ZoneStats>> {
    // We need to handle different dimensions. To avoid code duplication and copies,
    // we can use a helper function that takes generic ArrayViews.
    // However, pyo3 extraction gives specific types.
    // The best way to avoid copy is to match on dimension and call a generic function.

    fn run_zonal_stats<'a, D>(
        values: ndarray::ArrayView<'a, f64, D>,
        zones: ndarray::ArrayView<'a, i64, D>,
    ) -> PyResult<HashMap<i64, ZoneStats>>
    where
        D: ndarray::Dimension,
    {
        if values.shape() != zones.shape() {
            return Err(CoreError::InvalidArgument(format!(
                "Shape mismatch: values {:?} vs zones {:?}",
                values.shape(),
                zones.shape()
            ))
            .into());
        }

        let mut min_zone = i64::MAX;
        let mut max_zone = i64::MIN;

        for &z in zones.iter() {
            if z < min_zone {
                min_zone = z;
            }
            if z > max_zone {
                max_zone = z;
            }
        }

        let range = if max_zone >= min_zone {
            (max_zone - min_zone) as usize
        } else {
            0
        };
        let mut result: HashMap<i64, ZoneStats> = HashMap::new();

        if range < 1_000_000 {
            // Try to get contiguous slices for Rayon
            if let (Some(v_slice), Some(z_slice)) = (values.as_slice(), zones.as_slice()) {
                // Simple serial implementation for baseline
                let mut accs: Vec<SumSqAccumulator> = Vec::with_capacity(range + 1);
                for _ in 0..=range {
                    accs.push(SumSqAccumulator::new());
                }

                for (&v, &z) in v_slice.iter().zip(z_slice.iter()) {
                    if !v.is_nan() {
                        let idx = (z - min_zone) as usize;
                        unsafe {
                            accs.get_unchecked_mut(idx).update(v);
                        }
                    }
                }

                for (i, acc) in accs.into_iter().enumerate() {
                    if acc.count > 0 {
                        result.insert(min_zone + i as i64, acc.to_stats());
                    }
                }
            } else {
                // Fallback to sequential Zip for non-contiguous arrays
                let mut accs: Vec<SumSqAccumulator> = Vec::with_capacity(range + 1);
                for _ in 0..=range {
                    accs.push(SumSqAccumulator::new());
                }

                ndarray::Zip::from(&values).and(&zones).for_each(|&v, &z| {
                    if !v.is_nan() {
                        let idx = (z - min_zone) as usize;
                        unsafe {
                            accs.get_unchecked_mut(idx).update(v);
                        }
                    }
                });

                for (i, acc) in accs.into_iter().enumerate() {
                    if acc.count > 0 {
                        result.insert(min_zone + i as i64, acc.to_stats());
                    }
                }
            }
        } else {
            let mut accumulators: HashMap<i64, SumSqAccumulator> = HashMap::new();
            ndarray::Zip::from(&values).and(&zones).for_each(|&v, &z| {
                if !v.is_nan() {
                    let entry = accumulators.entry(z).or_insert_with(SumSqAccumulator::new);
                    entry.update(v);
                }
            });
            for (z, acc) in accumulators {
                result.insert(z, acc.to_stats());
            }
        }
        Ok(result)
    }

    // Dispatch logic
    if let Ok(v_arr) = values.extract::<PyReadonlyArray1<f64>>() {
        let z_arr = zones.extract::<PyReadonlyArray1<i64>>().map_err(|_| {
            CoreError::InvalidArgument("Zones must be int64 1D array matching values.".to_string())
        })?;
        return run_zonal_stats(v_arr.as_array(), z_arr.as_array());
    } else if let Ok(v_arr) = values.extract::<PyReadonlyArray2<f64>>() {
        let z_arr = zones.extract::<PyReadonlyArray2<i64>>().map_err(|_| {
            CoreError::InvalidArgument("Zones must be int64 2D array matching values.".to_string())
        })?;
        return run_zonal_stats(v_arr.as_array(), z_arr.as_array());
    } else if let Ok(v_arr) = values.extract::<PyReadonlyArray3<f64>>() {
        let z_arr = zones.extract::<PyReadonlyArray3<i64>>().map_err(|_| {
            CoreError::InvalidArgument("Zones must be int64 3D array matching values.".to_string())
        })?;
        return run_zonal_stats(v_arr.as_array(), z_arr.as_array());
    } else if let Ok(v_arr) = values.extract::<PyReadonlyArray4<f64>>() {
        let z_arr = zones.extract::<PyReadonlyArray4<i64>>().map_err(|_| {
            CoreError::InvalidArgument("Zones must be int64 4D array matching values.".to_string())
        })?;
        return run_zonal_stats(v_arr.as_array(), z_arr.as_array());
    }

    // If we are here, direct extraction failed. Try coercion.
    // Note: Coercion creates a new array (copy), so we can't avoid that if types don't match.
    let v_coerced = values.call_method1("astype", ("float64",))?;
    let z_coerced = zones.call_method1("astype", ("int64",))?;

    if let Ok(v_arr) = v_coerced.extract::<PyReadonlyArray1<f64>>() {
        let z_arr = z_coerced.extract::<PyReadonlyArray1<i64>>()?;
        return run_zonal_stats(v_arr.as_array(), z_arr.as_array());
    } else if let Ok(v_arr) = v_coerced.extract::<PyReadonlyArray2<f64>>() {
        let z_arr = z_coerced.extract::<PyReadonlyArray2<i64>>()?;
        return run_zonal_stats(v_arr.as_array(), z_arr.as_array());
    } else if let Ok(v_arr) = v_coerced.extract::<PyReadonlyArray3<f64>>() {
        let z_arr = z_coerced.extract::<PyReadonlyArray3<i64>>()?;
        return run_zonal_stats(v_arr.as_array(), z_arr.as_array());
    } else if let Ok(v_arr) = v_coerced.extract::<PyReadonlyArray4<f64>>() {
        let z_arr = z_coerced.extract::<PyReadonlyArray4<i64>>()?;
        return run_zonal_stats(v_arr.as_array(), z_arr.as_array());
    }

    Err(CoreError::InvalidArgument(
        "Could not process inputs. Ensure they are numeric arrays.".to_string(),
    )
    .into())
}
