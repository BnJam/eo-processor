"""
High-performance Earth Observation processing library.

This library provides Rust-accelerated functions for common EO/geospatial
computations that can be used within XArray/Dask workflows to bypass Python's GIL.
"""

from ._core import (
    normalized_difference as _normalized_difference,
    ndvi as _ndvi,
    ndwi as _ndwi,
    enhanced_vegetation_index as _enhanced_vegetation_index,
    median as _median,
)

__version__ = "0.1.0"

__all__ = [
    "normalized_difference",
    "ndvi",
    "ndwi",
    "enhanced_vegetation_index",
    "evi",
    "median",
    "composite",
]


def normalized_difference(a, b):
    """
    Compute normalized difference (a - b) / (a + b) using the Rust core.
    Supports 1D or 2D numpy float arrays; dimensional dispatch occurs in Rust.
    """
    return _normalized_difference(a, b)


def ndvi(nir, red):
    """
    Compute NDVI = (NIR - Red) / (NIR + Red) via Rust core (1D or 2D).
    """
    return _ndvi(nir, red)


def ndwi(green, nir):
    """
    Compute NDWI = (Green - NIR) / (Green + NIR) via Rust core (1D or 2D).
    """
    return _ndwi(green, nir)


def enhanced_vegetation_index(nir, red, blue):
    """
    Compute EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1) via Rust core (1D or 2D).
    """
    return _enhanced_vegetation_index(nir, red, blue)


# Alias
evi = enhanced_vegetation_index


def median(arr, skip_na=True):
    """
    Compute median over the time axis of a 1D, 2D, 3D, or 4D array.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array.
    skip_na : bool, optional
        Whether to skip NaN values, by default True. If False, the median
        of any pixel containing a NaN will be NaN.
    """
    return _median(arr, skip_na=skip_na)


def composite(arr, method="median", **kwargs):
    """
    Compute a composite over the time axis of a 1D, 2D, 3D, or 4D array.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array.
    method : str, optional
        The compositing method to use, by default "median".
    **kwargs
        Additional keyword arguments to pass to the compositing function.
    """
    if method == "median":
        return median(arr, **kwargs)
    else:
        raise ValueError(f"Unknown composite method: {method}")
