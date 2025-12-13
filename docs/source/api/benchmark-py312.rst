Benchmark Report
================

Meta
----
Python: 3.12.3
Platform: Linux-6.8.0-x86_64-with-glibc2.39
Group: all
Functions: ndvi, ndwi, evi, savi, nbr, ndmi, nbr2, gci, delta_ndvi, delta_nbr, normalized_difference, temporal_mean, temporal_std, median, trend_analysis, euclidean_distance, manhattan_distance, chebyshev_distance, minkowski_distance, moving_average_temporal, moving_average_temporal_stride, pixelwise_transform, zonal_stats, binary_dilation, binary_erosion, binary_opening, binary_closing
Distance Baseline: broadcast
Stress Mode: False
Loops: 3
Warmups: 1
Seed: 42
Compare NumPy: True
Height: 1000
Width: 1000
Time: 24
Points A: 1000
Points B: 1000
Point Dim: 32
Size Sweep: None
MA Window: 5
MA Stride: 4
MA Baseline: naive
Zones Count: 100

Results
-------
+================================+===========+============+==========+==========+============+=============================+==============================+==================+===============================+
| Function                       | Mean (ms) | StDev (ms) | Min (ms) | Max (ms) | Elements   | Rust Throughput (M elems/s) | NumPy Throughput (M elems/s) | Speedup vs NumPy | Shape                         |
+================================+===========+============+==========+==========+============+=============================+==============================+==================+===============================+
| ndvi                           | 92.47     | 0.10       | 92.37    | 92.61    | 1,000,000  | 10.81                       | 109.79                       | 0.10x            | 1000x1000                     |
| ndwi                           | 86.41     | 0.13       | 86.24    | 86.53    | 1,000,000  | 11.57                       | 167.42                       | 0.07x            | 1000x1000                     |
| evi                            | 105.95    | 0.29       | 105.69   | 106.35   | 1,000,000  | 9.44                        | 67.84                        | 0.14x            | 1000x1000                     |
| savi                           | 87.94     | 1.58       | 86.43    | 90.12    | 1,000,000  | 11.37                       | 130.02                       | 0.09x            | 1000x1000                     |
| nbr                            | 88.65     | 0.56       | 87.88    | 89.16    | 1,000,000  | 11.28                       | 153.39                       | 0.07x            | 1000x1000                     |
| ndmi                           | 87.21     | 0.69       | 86.59    | 88.16    | 1,000,000  | 11.47                       | 164.73                       | 0.07x            | 1000x1000                     |
| nbr2                           | 87.36     | 0.48       | 86.73    | 87.88    | 1,000,000  | 11.45                       | 161.99                       | 0.07x            | 1000x1000                     |
| gci                            | 86.43     | 0.27       | 86.22    | 86.81    | 1,000,000  | 11.57                       | 309.19                       | 0.04x            | 1000x1000                     |
| delta_ndvi                     | 260.39    | 0.68       | 259.66   | 261.30   | 1,000,000  | 3.84                        | 84.06                        | 0.05x            | 1000x1000                     |
| delta_nbr                      | 259.57    | 3.37       | 255.35   | 263.60   | 1,000,000  | 3.85                        | 70.82                        | 0.05x            | 1000x1000                     |
| normalized_difference          | 87.88     | 0.69       | 87.21    | 88.83    | 1,000,000  | 11.38                       | 141.94                       | 0.08x            | 1000x1000                     |
| temporal_mean                  | 829.64    | 32.62      | 795.78   | 873.70   | 24,000,000 | 28.93                       | 606.81                       | 0.05x            | 24x1000x1000                  |
| temporal_std                   | 1183.65   | 16.06      | 1162.08  | 1200.58  | 24,000,000 | 20.28                       | 155.39                       | 0.13x            | 24x1000x1000                  |
| median                         | 2411.87   | 19.91      | 2387.04  | 2435.77  | 24,000,000 | 9.95                        | 37.21                        | 0.27x            | 24x1000x1000                  |
| trend_analysis                 | 0.10      | 0.00       | 0.10     | 0.10     | 24         | 0.25                        | -                            | -                | T=24                          |
| euclidean_distance             | 818.72    | 12.19      | 803.60   | 833.45   | 32,000,000 | 39.09                       | 3418.68                      | 0.01x            | N=1000, M=1000, D=32          |
| manhattan_distance             | 847.10    | 73.90      | 746.51   | 921.94   | 32,000,000 | 37.78                       | 141.43                       | 0.27x            | N=1000, M=1000, D=32          |
| chebyshev_distance             | 768.66    | 1.14       | 767.39   | 770.15   | 32,000,000 | 41.63                       | 114.04                       | 0.37x            | N=1000, M=1000, D=32          |
| minkowski_distance             | 1077.96   | 1.84       | 1075.89  | 1080.36  | 32,000,000 | 29.69                       | 29.13                        | 1.02x            | N=1000, M=1000, D=32          |
| moving_average_temporal        | 2730.50   | 10.89      | 2722.45  | 2745.89  | 24,000,000 | 8.79                        | 38.56                        | 0.23x            | 24x1000x1000(win=5)           |
| moving_average_temporal_stride | 3038.88   | 31.02      | 3010.54  | 3082.06  | 24,000,000 | 7.90                        | 41.57                        | 0.19x            | 24x1000x1000(win=5, stride=4) |
| pixelwise_transform            | 1079.51   | 16.15      | 1058.23  | 1097.32  | 24,000,000 | 22.23                       | 155.23                       | 0.14x            | 24x1000x1000                  |
| zonal_stats                    | 60.45     | 0.08       | 60.34    | 60.53    | -          | -                           | -                            | 3.63x            | 1000x1000 (Zones=100)         |
| binary_dilation                | 55.43     | 3.18       | 53.03    | 59.93    | -          | -                           | -                            | 0.03x            | 1000x1000 (Kernel=3)          |
| binary_erosion                 | 47.86     | 0.96       | 47.14    | 49.21    | -          | -                           | -                            | 0.04x            | 1000x1000 (Kernel=3)          |
| binary_opening                 | 240.76    | 3.13       | 236.38   | 243.48   | -          | -                           | -                            | 0.01x            | 1000x1000 (Kernel=3)          |
| binary_closing                 | 249.54    | 1.33       | 247.72   | 250.86   | -          | -                           | -                            | 0.01x            | 1000x1000 (Kernel=3)          |
+================================+===========+============+==========+==========+============+=============================+==============================+==================+===============================+

Speedup vs NumPy = (NumPy mean time / Rust mean time); values > 1 indicate Rust is faster.
