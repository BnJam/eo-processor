"""Type stubs for eo_processor.

Notes:
- All spectral, temporal, processes & masking functions accept any numeric numpy dtype; Rust layer coerces to float64.
- Dimensional support:
  * Spectral indices (ndvi, ndwi, savi, nbr, ndmi, nbr2, gci, enhanced_vegetation_index/evi) + delta indices (delta_ndvi, delta_nbr): public wrappers 1D/2D (some internal paths support higher ranks).
  * normalized_difference: 1D–4D.
  * temporal_mean, temporal_std, median, composite: 1D–4D (time-first).
  * moving_average_temporal, moving_average_temporal_stride: 1D–4D (time-first).
  * pixelwise_transform: 1D–4D.
  * masking functions: 1D–4D.
  * distance functions: 2D only (N, D).
- Delta indices: pre/post inputs must have identical shapes.
"""

from typing import Literal, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

# Dimensional summary kept in sync with README & Sphinx:
#   - normalized_difference: 1D–4D
#   - ndvi, ndwi, savi, nbr, ndmi, nbr2, gci, enhanced_vegetation_index (evi), delta_ndvi, delta_nbr: primarily 1D–2D
#   - temporal_mean, temporal_std, median, composite: 1D–4D
#   - moving_average_temporal, moving_average_temporal_stride: 1D–4D
#   - pixelwise_transform: 1D–4D
#   - masking utilities: 1D–4D
#   - distance functions: 2D (N,D)
NumericArray = NDArray[np.generic]

__version__: Literal["0.6.0"]

# Spectral & change detection
def normalized_difference(a: NumericArray, b: NumericArray) -> NDArray[np.float64]: ...
def ndvi(nir: NumericArray, red: NumericArray) -> NDArray[np.float64]: ...
def ndwi(green: NumericArray, nir: NumericArray) -> NDArray[np.float64]: ...
def savi(nir: NumericArray, red: NumericArray, L: float = ...) -> NDArray[np.float64]: ...
def nbr(nir: NumericArray, swir2: NumericArray) -> NDArray[np.float64]: ...
def ndmi(nir: NumericArray, swir1: NumericArray) -> NDArray[np.float64]: ...
def nbr2(swir1: NumericArray, swir2: NumericArray) -> NDArray[np.float64]: ...
def gci(nir: NumericArray, green: NumericArray) -> NDArray[np.float64]: ...
def delta_ndvi(
    pre_nir: NumericArray,
    pre_red: NumericArray,
    post_nir: NumericArray,
    post_red: NumericArray,
) -> NDArray[np.float64]: ...
def delta_nbr(
    pre_nir: NumericArray,
    pre_swir2: NumericArray,
    post_nir: NumericArray,
    post_swir2: NumericArray,
) -> NDArray[np.float64]: ...
def enhanced_vegetation_index(nir: NumericArray, red: NumericArray, blue: NumericArray) -> NDArray[np.float64]: ...
evi = enhanced_vegetation_index

# Temporal reducers & composites
def median(arr: NumericArray, skip_na: bool = ...) -> NDArray[np.float64]: ...
def composite(arr: NumericArray, method: str = ..., **kwargs) -> NDArray[np.float64]: ...
def temporal_mean(arr: NumericArray, skip_na: bool = ...) -> NDArray[np.float64]: ...
def temporal_std(arr: NumericArray, skip_na: bool = ...) -> NDArray[np.float64]: ...

# Advanced temporal processes
def moving_average_temporal(
    arr: NumericArray,
    window: int,
    skip_na: bool = ...,
    mode: str = ...,
) -> NDArray[np.float64]: ...
def moving_average_temporal_stride(
    arr: NumericArray,
    window: int,
    stride: int,
    skip_na: bool = ...,
    mode: str = ...,
) -> NDArray[np.float64]: ...

# Pixel-wise transform
def pixelwise_transform(
    arr: NumericArray,
    scale: float = ...,
    offset: float = ...,
    clamp_min: Optional[float] = ...,
    clamp_max: Optional[float] = ...,
) -> NDArray[np.float64]: ...

# Distance functions (pairwise)
def euclidean_distance(points_a: NumericArray, points_b: NumericArray) -> NDArray[np.float64]: ...
def manhattan_distance(points_a: NumericArray, points_b: NumericArray) -> NDArray[np.float64]: ...
def chebyshev_distance(points_a: NumericArray, points_b: NumericArray) -> NDArray[np.float64]: ...
def minkowski_distance(points_a: NumericArray, points_b: NumericArray, p: float) -> NDArray[np.float64]: ...

# Masking utilities
def mask_vals(
    arr: NumericArray,
    values: Optional[Sequence[float]] = ...,
    fill_value: Optional[float] = ...,
    nan_to: Optional[float] = ...,
) -> NDArray[np.float64]: ...
def replace_nans(arr: NumericArray, value: float) -> NDArray[np.float64]: ...
def mask_out_range(
    arr: NumericArray,
    min_val: Optional[float] = ...,
    max_val: Optional[float] = ...,
    fill_value: Optional[float] = ...,
) -> NDArray[np.float64]: ...
def mask_invalid(
    arr: NumericArray,
    invalid_values: Sequence[float],
    fill_value: Optional[float] = ...,
) -> NDArray[np.float64]: ...
def mask_in_range(
    arr: NumericArray,
    min_val: Optional[float] = ...,
    max_val: Optional[float] = ...,
    fill_value: Optional[float] = ...,
) -> NDArray[np.float64]: ...
def mask_scl(
    scl: NumericArray,
    keep_codes: Optional[Sequence[float]] = ...,
    fill_value: Optional[float] = ...,
) -> NDArray[np.float64]: ...

# Raises ValueError if p < 1.0
