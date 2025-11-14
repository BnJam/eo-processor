import numpy as np
from eo_processor import temporal_sum, temporal_composite

def run_temporal_examples():
    """Demonstrate temporal_sum and temporal_composite."""
    print("--- Temporal Sum and Composite Example ---")

    # Create a synthetic 4D array (time, bands, y, x)
    t, b, h, w = 12, 4, 64, 64
    data = np.random.rand(t, b, h, w).astype(np.float64)
    data[0, 0, 0, 0] = np.nan  # Add a NaN for testing skip_na

    # --- temporal_sum ---
    print("\n1. temporal_sum")
    sum_all = temporal_sum(data, skip_na=False)
    sum_skip = temporal_sum(data, skip_na=True)
    print(f"Input shape: {data.shape}")
    print(f"Sum shape (skip_na=False): {sum_all.shape}")
    print(f"Sum shape (skip_na=True): {sum_skip.shape}")
    # Verify the NaN was propagated correctly
    assert np.isnan(sum_all[0, 0, 0]), "NaN should propagate when skip_na=False"
    assert not np.isnan(sum_skip[0, 0, 0]), "NaN should be skipped when skip_na=True"
    print("temporal_sum with skip_na=False correctly propagates NaNs.")
    print("temporal_sum with skip_na=True correctly skips NaNs.")

    # --- temporal_composite ---
    print("\n2. temporal_composite (weighted median)")
    weights = np.linspace(0.5, 1.5, t)
    composite = temporal_composite(data, weights=weights, skip_na=True)
    print(f"Input shape: {data.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Composite shape: {composite.shape}")

    # Verify a known weighted median case
    pixel_series = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    pixel_weights = np.array([1, 1, 10, 1, 1], dtype=np.float64)

    # Create a 4D array with this pixel series
    test_stack = np.zeros((5, 1, 1, 1), dtype=np.float64)
    test_stack[:, 0, 0, 0] = pixel_series

    weighted_median = temporal_composite(test_stack, weights=pixel_weights)
    assert weighted_median.shape == (1, 1, 1)
    # The median should be 3 because its weight (10) is > half the total weight (14 / 2 = 7)
    assert abs(weighted_median[0, 0, 0] - 3.0) < 1e-9, "Weighted median calculation is incorrect"
    print("temporal_composite correctly computes the weighted median.")

    print("\n--- Example Complete ---")

if __name__ == "__main__":
    run_temporal_examples()
