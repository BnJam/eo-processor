import numpy as np
import pytest
import xarray as xr
from skimage.feature import graycomatrix, graycoprops
from eo_processor import haralick_features

# Create a sample 2D array for testing
np.random.seed(42)
SAMPLE_ARRAY = np.random.randint(0, 8, size=(100, 100), dtype=np.uint8)
SAMPLE_XR = xr.DataArray(SAMPLE_ARRAY, dims=("y", "x"))

# Define parameters for tests
WINDOW_SIZE = 5
LEVELS = 8
FEATURES = ["contrast", "dissimilarity", "homogeneity", "entropy"]


def _calculate_skimage_props_for_pixel(window, levels):
    """
    Helper to calculate GLCM properties for a single window using scikit-image,
    matching the eo-processor implementation (symmetric, normed, averaged).
    """
    # Distances and angles must match the Rust implementation
    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    glcm = graycomatrix(
        window,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    results = {}
    # skimage calls it 'ASM' for homogeneity, we use the common name
    results["contrast"] = graycoprops(glcm, "contrast").mean()
    results["dissimilarity"] = graycoprops(glcm, "dissimilarity").mean()
    results["homogeneity"] = graycoprops(glcm, "homogeneity").mean()

    # Calculate entropy manually per-matrix, then average, to match graycoprops behavior
    entropy = np.mean([-np.sum(p[p > 0] * np.log2(p[p > 0])) for p in glcm])
    results["entropy"] = entropy

    return results


@pytest.fixture(scope="module")
def skimage_results():
    """
    Pre-calculate the expected results using scikit-image for the center pixel
    of the sample array. This is slow, so we do it once.
    """
    half_window = WINDOW_SIZE // 2
    center_y, center_x = 50, 50

    window = SAMPLE_ARRAY[
        center_y - half_window : center_y + half_window + 1,
        center_x - half_window : center_x + half_window + 1,
    ]

    return _calculate_skimage_props_for_pixel(window, LEVELS)


def test_haralick_features_correctness(skimage_results):
    """
    Compare the output of the Rust implementation with scikit-image for a single pixel.
    """
    # Run eo-processor implementation
    result_xr = haralick_features(
        SAMPLE_XR,
        window_size=WINDOW_SIZE,
        levels=LEVELS,
        features=FEATURES,
    )

    # Extract the values for the center pixel
    center_y, center_x = 50, 50
    eo_processor_results = result_xr.sel(y=center_y, x=center_x).values

    # Compare results
    for i, feature in enumerate(FEATURES):
        expected = skimage_results[feature]
        actual = eo_processor_results[i]
        # Using a tolerance because of potential minor floating point differences
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_dask_integration():
    """
    Test that the function works correctly with a Dask-backed xarray DataArray.
    """
    # Chunk the sample array to trigger Dask execution
    dask_xr = SAMPLE_XR.chunk({"y": 50, "x": 50})

    # Run haralick_features
    result_dask = haralick_features(
        dask_xr,
        window_size=WINDOW_SIZE,
        levels=LEVELS,
        features=FEATURES,
    )

    # Compute the result
    computed_result = result_dask.compute()

    # For simplicity, we compare against a non-dask run.
    # The correctness test already validates against scikit-image.
    result_numpy = haralick_features(
        SAMPLE_XR,
        window_size=WINDOW_SIZE,
        levels=LEVELS,
        features=FEATURES,
    )

    xr.testing.assert_allclose(computed_result, result_numpy)


def test_edge_case_small_array():
    """
    Test that the function handles arrays smaller than the window size gracefully.
    The Rust implementation should produce NaNs.
    """
    small_array = xr.DataArray(
        np.random.randint(0, 8, size=(2, 2), dtype=np.uint8), dims=("y", "x")
    )

    result = haralick_features(
        small_array,
        window_size=3,
        levels=8,
    )

    # The output should be all NaNs because no full window can be formed
    assert np.all(np.isnan(result.values))


def test_quantization():
    """
    Test that the auto-quantization logic works as expected.
    """
    # Create an array with values outside the [0, levels-1] range
    high_value_array = xr.DataArray(np.arange(0, 100).reshape(10, 10), dims=("y", "x"))
    levels = 16

    # This should run without error and produce a valid result
    result = haralick_features(high_value_array, window_size=3, levels=levels)

    # Check that the output is not all NaNs (which would indicate an error)
    assert not np.all(np.isnan(result.values))

    # Check shape
    assert result.shape == (len(FEATURES), 10, 10)
