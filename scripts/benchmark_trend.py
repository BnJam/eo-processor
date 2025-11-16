
import numpy as np
import time
from eo_processor._core import trend_analysis

def run_benchmark(series_length):
    """
    Benchmarks the trend_analysis function with a time series of a given length.
    """
    print(f"--- Benchmarking trend_analysis with series length: {series_length} ---")

    # Generate a sample time series with a break
    y = np.concatenate([
        np.linspace(0, 10, series_length // 2),
        np.linspace(10, 0, series_length // 2)
    ]) + np.random.normal(0, 0.5, series_length)

    # Time the execution
    t0 = time.time()
    try:
        segments = trend_analysis(y.tolist(), threshold=5.0)
        t_rust = time.time() - t0
        print(f"Execution time: {t_rust:.3f}s")
        print(f"Found {len(segments)} segments.")
    except Exception as e:
        print(f"An error occurred: {e}")
        t_rust = float('inf') # Indicate failure

    return t_rust

if __name__ == "__main__":
    lengths_to_test = [1000, 5000, 10000, 20000]
    results = {}

    for length in lengths_to_test:
        exec_time = run_benchmark(length)
        results[length] = exec_time
        print("-" * 20)

    print("\n--- Benchmark Summary ---")
    for length, exec_time in results.items():
        if exec_time == float('inf'):
            print(f"Series length: {length}, Time: FAILED")
        else:
            print(f"Series length: {length}, Time: {exec_time:.3f}s")
