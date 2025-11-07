"""
High-performance Earth Observation processing library.

This library provides Rust-accelerated functions for common EO/geospatial
computations that can be used within XArray/Dask workflows to bypass Python's GIL.
"""

from ._core import (
    normalized_difference,
    ndvi,
    ndwi,
    enhanced_vegetation_index,
)

__version__ = "0.1.0"

__all__ = [
    "normalized_difference",
    "ndvi",
    "ndwi",
    "enhanced_vegetation_index",
]
