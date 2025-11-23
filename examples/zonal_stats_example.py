import numpy as np
import eo_processor
from eo_processor import zonal_stats

def main():
    print("EO Processor - Zonal Statistics Example")
    print("=======================================")

    # 1. Create dummy data
    # --------------------
    shape = (100, 100)
    print(f"\nGenerating {shape[0]}x{shape[1]} random data...")
    
    # Values: Random float data (e.g., NDVI or reflectance)
    values = np.random.uniform(0.0, 1.0, size=shape)
    
    # Add some NaNs to demonstrate handling
    values[10:20, 10:20] = np.nan
    print("Added NaNs to a 10x10 region.")

    # Zones: Random integer labels (e.g., field IDs, classification classes)
    # 5 distinct zones (0 to 4)
    zones = np.random.randint(0, 5, size=shape, dtype=np.int64)
    
    print(f"Zones: {np.unique(zones)}")

    # 2. Run Zonal Statistics
    # -----------------------
    print("\nCalculating zonal statistics...")
    # This calls the optimized Rust implementation
    stats = zonal_stats(values, zones)

    # 3. Display Results
    # ------------------
    print("\nResults:")
    print(f"{'Zone ID':<10} {'Count':<10} {'Mean':<10} {'Min':<10} {'Max':<10} {'Std':<10}")
    print("-" * 65)

    # Sort by zone ID for display
    for zone_id in sorted(stats.keys()):
        zs = stats[zone_id]
        print(f"{zone_id:<10} {zs.count:<10} {zs.mean:<10.4f} {zs.min:<10.4f} {zs.max:<10.4f} {zs.std:<10.4f}")

    # 4. Accessing specific stats
    # ---------------------------
    if 1 in stats:
        z1 = stats[1]
        print(f"\nDetailed stats for Zone 1:")
        print(f"  Sum:   {z1.sum:.2f}")
        print(f"  Count: {z1.count}")
        print(f"  Mean:  {z1.mean:.4f}")

if __name__ == "__main__":
    main()
