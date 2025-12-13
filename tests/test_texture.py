import numpy as np
from eo_processor import texture_entropy

def test_texture_entropy():
    # Create a sample 2D array
    arr = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ], dtype=np.float64)

    # Compute the texture entropy with a 3x3 window
    entropy = texture_entropy(arr, 3)

    # Check that the output is a 2D array with the same shape as the input
    assert entropy.shape == arr.shape

    # Check that the entropy values are within a reasonable range (0 to log2(window_size^2))
    assert np.all(entropy >= 0)
    assert np.all(entropy <= np.log2(3**2))

def test_texture_entropy_small_image():
    # Create a sample 2D array smaller than the window size
    arr = np.array([
        [1, 2],
        [3, 4]
    ], dtype=np.float64)

    # Compute the texture entropy with a 3x3 window
    entropy = texture_entropy(arr, 3)

    # Check that the output is a 2D array with the same shape as the input
    assert entropy.shape == arr.shape

    # Check that the entropy values are within a reasonable range (0 to log2(window_size^2))
    assert np.all(entropy >= 0)
    assert np.all(entropy <= np.log2(3**2))

def test_texture_entropy_uniform_image():
    # Create a uniform 2D array
    arr = np.ones((10, 10), dtype=np.float64)

    # Compute the texture entropy with a 3x3 window
    entropy = texture_entropy(arr, 3)

    # Check that the output is a 2D array with the same shape as the input
    assert entropy.shape == arr.shape

    # Check that the entropy is close to zero for a uniform image
    assert np.allclose(entropy, 0)
