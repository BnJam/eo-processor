# scripts/benchmark_trends.py

import timeit
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.eo_processor._core import trend_analysis

def run_benchmark():
    """
    Measures the performance of the trend_analysis UDF with varying data lengths.
    """
    data_lengths = [100, 500, 1000, 5000, 10000]
    results = {}

    for length in data_lengths:
        # Generate sample data
        y = np.random.rand(length).tolist()

        # Measure execution time
        execution_time = timeit.timeit(
            lambda: trend_analysis(y, threshold=0.1),
            number=10
        )
        results[length] = execution_time / 10

    return results

if __name__ == "__main__":
    benchmark_results = run_benchmark()
    print("Trend Analysis Benchmark Results:")
    for length, avg_time in benchmark_results.items():
        print(f"  Data Length: {length}, Average Time: {avg_time:.6f} seconds")
