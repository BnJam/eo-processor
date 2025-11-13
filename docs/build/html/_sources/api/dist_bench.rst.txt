Benchmark Report
================

Meta
----
Python: 3.10.18
Platform: macOS-15.6.1-arm64-arm-64bit
Group: distances
Functions: euclidean_distance, manhattan_distance, chebyshev_distance, minkowski_distance
Loops: 2
Warmups: 1
Seed: 42
Compare NumPy: True
Height: 2048
Width: 2048
Time: 12
Points A: 800
Points B: 600
Point Dim: 32

Results
-------
+====================+===========+============+==========+==========+==========+=============================+==============================+==================+====================+
| Function           | Mean (ms) | StDev (ms) | Min (ms) | Max (ms) | Elements | Rust Throughput (M elems/s) | NumPy Throughput (M elems/s) | Speedup vs NumPy | Shape              |
+====================+===========+============+==========+==========+==========+=============================+==============================+==================+====================+
| euclidean_distance | 1.85      | 0.00       | 1.85     | 1.85     | 480,000  | 259.41                      | 325.24                       | 0.80x            | N=800, M=600, D=32 |
| manhattan_distance | 2.57      | 0.14       | 2.43     | 2.71     | 480,000  | 186.61                      | 13.93                        | 13.40x           | N=800, M=600, D=32 |
| chebyshev_distance | 2.67      | 0.09       | 2.58     | 2.76     | 480,000  | 179.79                      | 11.01                        | 16.33x           | N=800, M=600, D=32 |
| minkowski_distance | 22.62     | 0.07       | 22.55    | 22.69    | 480,000  | 21.22                       | 1.97                         | 10.78x           | N=800, M=600, D=32 |
+====================+===========+============+==========+==========+==========+=============================+==============================+==================+====================+

Speedup vs NumPy = (NumPy mean time / Rust mean time); values > 1 indicate Rust is faster.
