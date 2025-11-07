"""
High-performance Earth Observation processing library.

This library provides Rust-accelerated functions for common EO/geospatial
computations that can be used within XArray/Dask workflows to bypass Python's GIL.
"""

from ._core import (
    normalized_difference_1d,
    normalized_difference_2d,
    normalized_difference_3d,
    ndvi_1d,
    ndvi_2d,
    ndvi_3d,
    ndwi_1d,
    ndwi_2d,
)

__version__ = "0.1.0"

__all__ = [
    "normalized_difference_1d",
    "normalized_difference_2d",
    "normalized_difference_3d",
    "ndvi_1d",
    "ndvi_2d",
    "ndvi_3d",
    "ndwi_1d",
    "ndwi_2d",
    "normalized_difference",
    "ndvi",
    "ndwi",
]


def normalized_difference(a, b):
    """
    Compute normalized difference between two arrays.

    Automatically handles both 1D and 2D arrays.

    Formula: (a - b) / (a + b)

    Parameters
    ----------
    a : numpy.ndarray
        First input array (e.g., NIR band for NDVI)
    b : numpy.ndarray
        Second input array (e.g., Red band for NDVI)

    Returns
    -------
    numpy.ndarray
        Array with the same shape as inputs containing normalized difference values

    Examples
    --------
    >>> import numpy as np
    >>> from eo_processor import normalized_difference
    >>> nir = np.array([0.8, 0.7, 0.6])
    >>> red = np.array([0.2, 0.1, 0.3])
    >>> ndvi = normalized_difference(nir, red)
    >>> ndvi
    array([0.6       , 0.75      , 0.33333333])
    """
    if a.ndim == 1:
        return normalized_difference_1d(a, b)
    elif a.ndim == 2:
        return normalized_difference_2d(a, b)
    elif a.ndim == 3:
        return normalized_difference_3d(a, b)
    else:
        raise ValueError(f"Unsupported array dimension: {a.ndim}. Only 1D and 2D arrays are supported.")


def ndvi(nir, red):
    """
    Compute NDVI (Normalized Difference Vegetation Index).

    NDVI = (NIR - Red) / (NIR + Red)

    Automatically handles both 1D and 2D arrays.

    Parameters
    ----------
    nir : numpy.ndarray
        Near-infrared band values
    red : numpy.ndarray
        Red band values

    Returns
    -------
    numpy.ndarray
        NDVI values ranging from -1 to 1

    Examples
    --------
    >>> import numpy as np
    >>> from eo_processor import ndvi
    >>> nir = np.array([0.8, 0.7, 0.6])
    >>> red = np.array([0.2, 0.1, 0.3])
    >>> ndvi_values = ndvi(nir, red)
    """
    if nir.ndim == 1:
        return ndvi_1d(nir, red)
    elif nir.ndim == 2:
        return ndvi_2d(nir, red)
    elif nir.ndim == 3:
        return ndvi_3d(nir, red)
    else:
        raise ValueError(f"Unsupported array dimension: {nir.ndim}. Only 1D and 2D arrays are supported.")


def ndwi(green, nir):
    """
    Compute NDWI (Normalized Difference Water Index).

    NDWI = (Green - NIR) / (Green + NIR)

    Automatically handles both 1D and 2D arrays.

    Parameters
    ----------
    green : numpy.ndarray
        Green band values
    nir : numpy.ndarray
        Near-infrared band values

    Returns
    -------
    numpy.ndarray
        NDWI values ranging from -1 to 1

    Examples
    --------
    >>> import numpy as np
    >>> from eo_processor import ndwi
    >>> green = np.array([0.3, 0.4, 0.5])
    >>> nir = np.array([0.2, 0.1, 0.3])
    >>> ndwi_values = ndwi(green, nir)
    """
    if green.ndim == 1:
        return ndwi_1d(green, nir)
    elif green.ndim == 2:
        return ndwi_2d(green, nir)
    else:
        raise ValueError(f"Unsupported array dimension: {green.ndim}. Only 1D and 2D arrays are supported.")
