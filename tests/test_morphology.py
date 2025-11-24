import numpy as np
import pytest
from eo_processor import binary_dilation, binary_erosion, binary_opening, binary_closing

def test_binary_dilation_basic():
    # Single pixel in 3x3
    input_arr = np.zeros((5, 5), dtype=np.uint8)
    input_arr[2, 2] = 1
    
    # Dilate with 3x3 kernel -> 3x3 block of 1s
    dilated = binary_dilation(input_arr, kernel_size=3)
    
    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[1:4, 1:4] = 1
    
    np.testing.assert_array_equal(dilated, expected)

def test_binary_erosion_basic():
    # 3x3 block of 1s
    input_arr = np.zeros((5, 5), dtype=np.uint8)
    input_arr[1:4, 1:4] = 1
    
    # Erode with 3x3 kernel -> single pixel at center
    eroded = binary_erosion(input_arr, kernel_size=3)
    
    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[2, 2] = 1
    
    np.testing.assert_array_equal(eroded, expected)

def test_binary_opening_noise_removal():
    # 3x3 block + noise pixel
    input_arr = np.zeros((7, 7), dtype=np.uint8)
    input_arr[2:5, 2:5] = 1
    input_arr[0, 0] = 1 # Noise
    
    # Opening should remove the isolated noise pixel but keep the block roughly same
    # (Erosion removes noise and shrinks block, Dilation restores block)
    opened = binary_opening(input_arr, kernel_size=3)
    
    expected = np.zeros((7, 7), dtype=np.uint8)
    expected[2:5, 2:5] = 1
    
    np.testing.assert_array_equal(opened, expected)

def test_binary_closing_hole_filling():
    # 3x3 block with hole in center
    input_arr = np.zeros((5, 5), dtype=np.uint8)
    input_arr[1:4, 1:4] = 1
    input_arr[2, 2] = 0 # Hole
    
    # Closing should fill the hole
    closed = binary_closing(input_arr, kernel_size=3)
    
    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[1:4, 1:4] = 1
    
    np.testing.assert_array_equal(closed, expected)

def test_kernel_size_5():
    input_arr = np.zeros((7, 7), dtype=np.uint8)
    input_arr[3, 3] = 1
    
    # Dilate with 5x5 kernel -> 5x5 block
    dilated = binary_dilation(input_arr, kernel_size=5)
    
    expected = np.zeros((7, 7), dtype=np.uint8)
    expected[1:6, 1:6] = 1
    
    np.testing.assert_array_equal(dilated, expected)

def test_boundary_handling():
    # Pixel at edge
    input_arr = np.zeros((5, 5), dtype=np.uint8)
    input_arr[0, 0] = 1
    
    # Dilate 3x3 -> 2x2 block at corner
    dilated = binary_dilation(input_arr, kernel_size=3)
    
    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[0:2, 0:2] = 1
    
    np.testing.assert_array_equal(dilated, expected)
