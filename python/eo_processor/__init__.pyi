"""Type stubs for eo_processor.

Notes:
- All spectral & masking functions accept any numeric numpy dtype; Rust layer coerces to float64.
- Dimensional support:
  * Spectral indices (ndvi, ndwi, savi, nbr, ndmi, nbr2, gci, enhanced_vegetation_index/evi) and delta indices (delta_ndvi, delta_nbr) dispatch over 1D and 2D; some additionally allow 3D/4D internally via generic normalized_difference but wrappers expose 1D/2D semantics.
  * normalized_difference supports 1D–4D; temporal_mean, temporal_std, median, composite, masking functions support 1D–4D.
  * Distance functions expect 2D inputs shaped (N, D).
- Delta indices: pre/post inputs must have identical shapes; currently limited to 1D or 2D in public Python API.
"""

from typing import Literal, Optional, Sequence
import numpy as np
from numpy.typing import NDArray

# Inputs accept any numeric dtype; implementation coerces to float64 internally for stable arithmetic.
# Dimensional summary:
#   - normalized_difference: 1D–4D
#   - ndvi, ndwi, savi, nbr, ndmi, nbr2, gci, enhanced_vegetation_index (evi): 1D–2D primary (internal may handle >2D via generic path)
#   - delta_ndvi, delta_nbr: 1D–2D (pre/post shapes must match exactly)
#   - temporal_mean, temporal_std, median, composite: 1D–4D (time-first)
#   - masking functions: 1D–4D
#   - distance functions (euclidean_distance, manhattan_distance, chebyshev_distance, minkowski_distance): 2D only (N, D)
# Keep this list in sync with README & Sphinx docs when adding new functions.
NumericArray = NDArray[np.generic]

__version__: Literal["0.6.0"]

def normalized_difference(a: NumericArray, b: NumericArray) -> NDArray[np.float64]: ...
def ndvi(nir: NumericArray, red: NumericArray) -> NDArray[np.float64]: ...
def ndwi(green: NumericArray, nir: NumericArray) -> NDArray[np.float64]: ...
def savi(
    nir: NumericArray, red: NumericArray, L: float = ...
) -> NDArray[np.float64]: ...
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
def enhanced_vegetation_index(
    nir: NumericArray, red: NumericArray, blue: NumericArray
) -> NDArray[np.float64]: ...

evi = enhanced_vegetation_index

def median(arr: NumericArray, skip_na: bool = ...) -> NDArray[np.float64]: ...
def composite(
    arr: NumericArray, method: str = ..., **kwargs
) -> NDArray[np.float64]: ...
def temporal_mean(arr: NumericArray, skip_na: bool = ...) -> NDArray[np.float64]: ...
def temporal_std(arr: NumericArray, skip_na: bool = ...) -> NDArray[np.float64]: ...
def euclidean_distance(
    points_a: NumericArray, points_b: NumericArray
) -> NDArray[np.float64]: ...
def manhattan_distance(
    points_a: NumericArray, points_b: NumericArray
) -> NDArray[np.float64]: ...
def chebyshev_distance(
    points_a: NumericArray, points_b: NumericArray
) -> NDArray[np.float64]: ...
def minkowski_distance(
    points_a: NumericArray, points_b: NumericArray, p: float
) -> NDArray[np.float64]: ...
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
