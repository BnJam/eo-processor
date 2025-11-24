import numpy as np
import eo_processor
from eo_processor import binary_dilation, binary_erosion, binary_opening, binary_closing

def print_grid(arr):
    for row in arr:
        print(" ".join(str(x) for x in row))

def main():
    print("EO Processor - Morphology Example")
    print("=================================")

    # 1. Create a simple binary pattern (7x7)
    # ---------------------------------------
    # A 3x3 block in the center with some noise
    input_arr = np.zeros((7, 7), dtype=np.uint8)
    input_arr[2:5, 2:5] = 1
    input_arr[0, 0] = 1 # Noise pixel
    input_arr[3, 3] = 0 # Hole in the center

    print("\nInput Pattern:")
    print_grid(input_arr)

    # 2. Dilation
    # -----------
    print("\n--- Dilation (Kernel=3) ---")
    # Expands the white regions
    dilated = binary_dilation(input_arr, kernel_size=3)
    print_grid(dilated)

    # 3. Erosion
    # ----------
    print("\n--- Erosion (Kernel=3) ---")
    # Shrinks the white regions (removes noise)
    eroded = binary_erosion(input_arr, kernel_size=3)
    print_grid(eroded)

    # 4. Opening (Erosion -> Dilation)
    # --------------------------------
    print("\n--- Opening (Kernel=3) ---")
    # Removes small objects (noise) but preserves shape of larger objects
    opened = binary_opening(input_arr, kernel_size=3)
    print_grid(opened)
    print("(Note: Noise at [0,0] is gone, hole at [3,3] remains)")

    # 5. Closing (Dilation -> Erosion)
    # --------------------------------
    print("\n--- Closing (Kernel=3) ---")
    # Fills small holes
    closed = binary_closing(input_arr, kernel_size=3)
    print_grid(closed)
    print("(Note: Hole at [3,3] is filled, noise at [0,0] remains)")

if __name__ == "__main__":
    main()
