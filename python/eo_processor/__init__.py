"""
High-performance Earth Observation processing library.

This library provides Rust-accelerated functions for common EO/geospatial
computations that can be used within XArray/Dask workflows to bypass Python's GIL.

NOTE: All public spectral and temporal functions accept any numeric NumPy dtype
(int, uint, float32, float64, etc.). Inputs are automatically coerced to float64
in the Rust layer for consistent and stable computation.
"""

from ._core import (
    chebyshev_distance as _chebyshev_distance,
    delta_nbr as _delta_nbr,
    delta_ndvi as _delta_ndvi,
    enhanced_vegetation_index as _enhanced_vegetation_index,
    euclidean_distance as _euclidean_distance,
    gci as _gci,
    manhattan_distance as _manhattan_distance,
    mask_in_range as _mask_in_range,
    mask_invalid as _mask_invalid,
    mask_out_range as _mask_out_range,
    mask_scl as _mask_scl,
    mask_vals as _mask_vals,
    median as _median,
    minkowski_distance as _minkowski_distance,
    moving_average_temporal as _moving_average_temporal,
    moving_average_temporal_stride as _moving_average_temporal_stride,
    nbr as _nbr,
    nbr2 as _nbr2,
    ndmi as _ndmi,
    ndvi as _ndvi,
    ndwi as _ndwi,
    normalized_difference as _normalized_difference,
    pixelwise_transform as _pixelwise_transform,
    replace_nans as _replace_nans,
    savi as _savi,
    temporal_mean as _temporal_mean,
    temporal_std as _temporal_std,
    temporal_sum as _temporal_sum,
    temporal_composite as _temporal_composite,
)

__version__ = "0.6.0"

__all__ = [
    "chebyshev_distance",
    "composite",
    "delta_nbr",
    "delta_ndvi",
    "enhanced_vegetation_index",
    "euclidean_distance",
    "evi",
    "gci",
    "manhattan_distance",
    "mask_in_range",
    "mask_invalid",
    "mask_out_range",
    "mask_scl",
    "mask_vals",
    "median",
    "minkowski_distance",
    "moving_average_temporal",
    "moving_average_temporal_stride",
    "nbr",
    "nbr2",
    "ndmi",
    "ndvi",
    "ndwi",
    "normalized_difference",
    "pixelwise_transform",
    "replace_nans",
    "savi",
    "temporal_mean",
    "temporal_std",
    "temporal_sum",
    "temporal_composite",
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


def savi(nir, red, L=0.5, **kwargs):
    """
    Compute Soil Adjusted Vegetation Index (SAVI).

    SAVI = (NIR - Red) / (NIR + Red + L) * (1 + L)

    Parameters
    ----------
    nir : numpy.ndarray
        Near-infrared band.
    red : numpy.ndarray
        Red band.
    L : float, optional
        Soil brightness correction factor (default 0.5). Typical range 0–1.
        Larger L reduces soil background influence.
    **kwargs :
        May contain 'l' to specify the soil adjustment factor instead of 'L'.

    Returns
    -------
    numpy.ndarray
        SAVI values with same shape as inputs.

    Notes
    -----
    You can call as:
        savi(nir, red)              # uses L=0.5
        savi(nir, red, L=0.25)      # custom L
        savi(nir, red, l=0.25)      # alternative keyword
    If both L and l are provided, 'l' takes precedence.
    """
    l_val = kwargs.get("l", L)
    return _savi(nir, red, l_val)


def ndmi(nir, swir1):
    """
    Normalized Difference Moisture Index (NDMI)

    NDMI = (NIR - SWIR1) / (NIR + SWIR1)

    Parameters
    ----------
    nir : numpy.ndarray
        Near-infrared band.
    swir1 : numpy.ndarray
        Short-wave infrared 1 band.

    Returns
    -------
    numpy.ndarray
        NDMI values (-1 .. 1).
    """
    return _ndmi(nir, swir1)


def nbr2(swir1, swir2):
    """
    Normalized Burn Ratio 2 (NBR2)

    NBR2 = (SWIR1 - SWIR2) / (SWIR1 + SWIR2)

    Parameters
    ----------
    swir1 : numpy.ndarray
        Short-wave infrared 1 band.
    swir2 : numpy.ndarray
        Short-wave infrared 2 band.

    Returns
    -------
    numpy.ndarray
        NBR2 values (-1 .. 1).
    """
    return _nbr2(swir1, swir2)


def gci(nir, green):
    """
    Green Chlorophyll Index (GCI)

    GCI = (NIR / Green) - 1

    Parameters
    ----------
    nir : numpy.ndarray
        Near-infrared band (any numeric dtype; auto-coerced to float64).
    green : numpy.ndarray
        Green band (any numeric dtype).

    Returns
    -------
    numpy.ndarray
        GCI values (unbounded; typical vegetation > 0).

    Notes
    -----
    Division-by-near-zero guarded; returns 0 where Green is ~0.
    """
    return _gci(nir, green)


def delta_ndvi(pre_nir, pre_red, post_nir, post_red):
    """
    Change in NDVI (pre - post).

    Parameters
    ----------
    pre_nir, pre_red : numpy.ndarray
        Pre-event near-infrared and red bands.
    post_nir, post_red : numpy.ndarray
        Post-event near-infrared and red bands.

    Returns
    -------
    numpy.ndarray
        ΔNDVI array (same shape as inputs), positive values often indicate vegetation loss.

    Notes
    -----
    Inputs may be any numeric dtype; values are coerced to float64 internally.
    """
    return _delta_ndvi(pre_nir, pre_red, post_nir, post_red)


def delta_nbr(pre_nir, pre_swir2, post_nir, post_swir2):
    """
    Change in NBR (pre - post) for burn severity analysis.

    Parameters
    ----------
    pre_nir, pre_swir2 : numpy.ndarray
        Pre-event NIR and SWIR2 bands.
    post_nir, post_swir2 : numpy.ndarray
        Post-event NIR and SWIR2 bands.

    Returns
    -------
    numpy.ndarray
        ΔNBR array (same shape as inputs). Larger positive values generally indicate higher burn severity.

    Notes
    -----
    Inputs may be any numeric dtype; internal coercion to float64.
    """
    return _delta_nbr(pre_nir, pre_swir2, post_nir, post_swir2)


def nbr(nir, swir2):
    """
    Compute Normalized Burn Ratio (NBR).

    NBR = (NIR - SWIR2) / (NIR + SWIR2)

    Parameters
    ----------
    nir : numpy.ndarray
        Near-infrared band.
    swir2 : numpy.ndarray
        Short-wave infrared (SWIR2) band.

    Returns
    -------
    numpy.ndarray
        NBR values with same shape as inputs.
    """
    return _nbr(nir, swir2)


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
    Composite convenience wrapper over temporal aggregation functions.

    Currently only supports method="median". The composite is computed along the
    leading time axis for arrays with shape:
      - 1D: (time,)
      - 2D: (time, bands|features)
      - 3D: (time, y, x)
      - 4D: (time, band, y, x)

    Parameters
    ----------
    arr : numpy.ndarray
        Input array (1D–4D). Any numeric dtype accepted; coerced to float64 internally.
    method : str, optional
        Name of compositing method. Currently only "median" is implemented.
        Providing any other value raises ValueError.
    **kwargs
        Passed through to the underlying method. For "median" this includes
        skip_na (bool) to control NaN handling (default True).

    Returns
    -------
    numpy.ndarray
        Composite with time axis removed. Output dimensionality:
          - 1D input -> scalar (float64)
          - 2D input -> (bands,)
          - 3D input -> (y, x)
          - 4D input -> (band, y, x)

    Notes
    -----
    This wrapper centralizes dispatch so future methods (e.g. "mean", "std",
    "percentile") can be added without changing call sites. When new methods
    are introduced: update README, type stubs, Sphinx autosummary entries,
    and bump the minor version.

    Raises
    ------
    ValueError
        If method is not recognized.
    """
    if method == "median":
        return median(arr, **kwargs)
    else:
        raise ValueError(f"Unknown composite method: {method}")


def temporal_mean(arr, skip_na=True):
    """
    Compute the mean along the leading time axis of a 1D–4D time‑first array.

    Parameters
    ----------
    arr : numpy.ndarray
        Time‑first array (1D–4D). Shapes:
        (T,), (T, F), (T, Y, X), (T, B, Y, X).
    skip_na : bool, default True
        If True, NaNs are excluded per pixel/band; all‑NaN series produce NaN.
        If False, any NaN in a series propagates NaN to the output position.

    Returns
    -------
    numpy.ndarray
        Mean with time axis removed; float64 dtype. Scalar for 1D input.
    """
    return _temporal_mean(arr, skip_na=skip_na)


def temporal_std(arr, skip_na=True):
    """
    Compute the sample standard deviation (ddof=1) along the leading time axis
    of a 1D–4D time‑first array.

    Parameters
    ----------
    arr : numpy.ndarray
        Time‑first array (1D–4D). Shapes:
        (T,), (T, F), (T, Y, X), (T, B, Y, X).
    skip_na : bool, default True
        If True, NaNs are excluded before variance; fewer than 2 valid values
        yield NaN. If False, any NaN in a series propagates NaN.

    Returns
    -------
    numpy.ndarray
        Standard deviation with time axis removed; float64 dtype. Scalar for 1D input.
    """
    return _temporal_std(arr, skip_na=skip_na)


def euclidean_distance(points_a, points_b):
    """
    Compute pairwise Euclidean distances between two point sets.

    Parameters
    ----------
    points_a : numpy.ndarray (N, D)
    points_b : numpy.ndarray (M, D)

    Returns
    -------
    numpy.ndarray (N, M)
        Distance matrix where element (i, j) is distance between
        points_a[i] and points_b[j].
    """
    return _euclidean_distance(points_a, points_b)


def manhattan_distance(points_a, points_b):
    """
    Compute pairwise Manhattan (L1) distances between two point sets.
    See `euclidean_distance` for shape conventions.
    """
    return _manhattan_distance(points_a, points_b)


def chebyshev_distance(points_a, points_b):
    """
    Compute pairwise Chebyshev (L∞) distances between two point sets.
    """
    return _chebyshev_distance(points_a, points_b)


def minkowski_distance(points_a, points_b, p):
    """
    Compute pairwise Minkowski distances (order `p`) between two point sets.

    Parameters
    ----------
    points_a : numpy.ndarray (N, D)
        First point set.
    points_b : numpy.ndarray (M, D)
        Second point set.
    p : float
        Norm order (must be >= 1). p=1 → Manhattan, p=2 → Euclidean,
        large p → approximates Chebyshev (L∞).

    Returns
    -------
    numpy.ndarray (N, M)
        Distance matrix.

    Raises
    ------
    ValueError
        If p < 1.0 (propagated from the Rust implementation).
    """
    return _minkowski_distance(points_a, points_b, p)


def mask_vals(arr, values=None, fill_value=None, nan_to=None):
    """
    Mask specified values (exact equality) and optionally replace NaNs.

    Parameters
    ----------
    arr : numpy.ndarray (1D–4D)
        Input array; any numeric dtype accepted (coerced to float64 internally).
    values : sequence, optional
        Iterable of numeric codes to mask. If None, no value masking is performed.
    fill_value : float, optional
        Value to write for masked codes. Defaults to NaN when None.
    nan_to : float, optional
        If provided, all NaNs (original or created by masking) are replaced with this value
        after masking.

    Returns
    -------
    numpy.ndarray (float64)
        Masked array preserving original shape.

    Notes
    -----
    This is a thin pass-through to the Rust implementation; see README “Masking Utilities”.
    """
    return _mask_vals(arr, values=values, fill_value=fill_value, nan_to=nan_to)


def replace_nans(arr, value):
    """
    Replace all NaNs in `arr` with `value`.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array (1D–4D supported).
    value : float
        Replacement for every NaN.

    Returns
    -------
    numpy.ndarray
        Array with NaNs replaced.
    """
    return _replace_nans(arr, value)


def mask_out_range(arr, min_val=None, max_val=None, fill_value=None):
    """
    Mask values outside a specified numeric range [min_val, max_val].

    Parameters
    ----------
    arr : numpy.ndarray
        Input array (1D–4D).
    min_val : float, optional
        Minimum valid value (inclusive).
    max_val : float, optional
        Maximum valid value (inclusive).
    fill_value : float, optional
        Value for masked positions (default NaN).

    Returns
    -------
    numpy.ndarray
        Masked array.
    """
    return _mask_out_range(arr, min=min_val, max=max_val, fill_value=fill_value)


def mask_invalid(arr, invalid_values, fill_value=None):
    """
    Mask a list of common invalid sentinel values.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array.
    invalid_values : sequence
        List of numeric codes to mask.
    fill_value : float, optional
        Value for masked positions (default NaN).

    Returns
    -------
    numpy.ndarray
        Masked array.
    """
    return _mask_invalid(arr, invalid_values=invalid_values, fill_value=fill_value)


def mask_in_range(arr, min_val=None, max_val=None, fill_value=None):
    """
    Mask values inside a specified numeric range [min_val, max_val].

    Parameters
    ----------
    arr : numpy.ndarray
        Input array (1D–4D).
    min_val : float, optional
        Minimum value of range to mask (inclusive).
    max_val : float, optional
        Maximum value of range to mask (inclusive).
    fill_value : float, optional
        Value for masked positions (default NaN).

    Returns
    -------
    numpy.ndarray
        Masked array.
    """
    return _mask_in_range(arr, min=min_val, max=max_val, fill_value=fill_value)


def mask_scl(scl, keep_codes=None, fill_value=None):
    """
    Mask a Sentinel-2 Scene Classification Layer (SCL) array.

    By default, keeps vegetation, water, bare soil, and snow.

    Parameters
    ----------
    scl : numpy.ndarray
        SCL array.
    keep_codes : sequence, optional
        List of SCL codes to keep. Defaults to [4, 5, 6, 7, 11].
    fill_value : float, optional
        Value for masked positions (default NaN).

    Returns
    -------
    numpy.ndarray
        Masked SCL array.
    """
    return _mask_scl(scl, keep_codes=keep_codes, fill_value=fill_value)


def moving_average_temporal(arr, window, skip_na=True, mode="same"):
    """
    Sliding window mean along leading time axis of a 1D–4D time-first array.

    Parameters
    ----------
    arr : numpy.ndarray
        Time-first array (T,...).
    window : int
        Window size (>=1).
    skip_na : bool, default True
        Exclude NaNs from window mean; if all NaN -> NaN.
    mode : {"same","valid"}, default "same"
        "same": output length equals T (edge windows shrink).
        "valid": only full windows; output length = T - window + 1.

    Returns
    -------
    numpy.ndarray
    """
    return _moving_average_temporal(arr, window, skip_na=skip_na, mode=mode)


def moving_average_temporal_stride(arr, window, stride, skip_na=True, mode="same"):
    """
    Stride-based sliding window mean along leading time axis.

    Computes moving_average_temporal then samples every `stride` steps
    along the time axis to reduce temporal resolution.

    Parameters
    ----------
    arr : numpy.ndarray
        Time-first array (T,...).
    window : int
        Window size (>=1).
    stride : int
        Sampling interval along output time axis (>=1).
    skip_na : bool, default True
        Exclude NaNs from window mean; if all NaN -> NaN.
    mode : {"same","valid"}, default "same"
        "same": output length equals T (variable-size edges).
        "valid": only full windows; base length = T - window + 1.

    Returns
    -------
    numpy.ndarray
        Downsampled moving average with time dimension approximately
        ceil(base_length / stride).
    """
    return _moving_average_temporal_stride(
        arr, window, stride, skip_na=skip_na, mode=mode
    )


def pixelwise_transform(arr, scale=1.0, offset=0.0, clamp_min=None, clamp_max=None):
    """
    Apply linear transform scale*arr + offset with optional clamping per element.

    NaNs propagate unchanged.

    Parameters
    ----------
    arr : numpy.ndarray
    scale : float
    offset : float
    clamp_min : float or None
    clamp_max : float or None

    Returns
    -------
    numpy.ndarray
    """
    return _pixelwise_transform(
        arr,
        scale=scale,
        offset=offset,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )


def temporal_sum(arr, skip_na=True):
    """
    Compute the sum along the leading time axis of a 1D–4D time‑first array.

    Parameters
    ----------
    arr : numpy.ndarray
        Time‑first array (1D–4D).
    skip_na : bool, default True
        If True, NaNs are excluded.

    Returns
    -------
    numpy.ndarray
        Sum with time axis removed; float64 dtype. Scalar for 1D input.
    """
    return _temporal_sum(arr, skip_na=skip_na)


def temporal_composite(arr, weights, skip_na=True):
    """
    Compute a temporal composite of a 4D array using a weighted median.

    Parameters
    ----------
    arr : numpy.ndarray
        4D array with shape (time, bands, y, x).
    weights : numpy.ndarray
        1D array of weights with the same length as the time dimension of arr.
    skip_na : bool, default True
        If True, NaNs are excluded.

    Returns
    -------
    numpy.ndarray
        Composited 3D array with shape (bands, y, x).
    """
    return _temporal_composite(arr, weights, skip_na=skip_na)
