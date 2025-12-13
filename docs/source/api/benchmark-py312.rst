Benchmark Report
================

Meta
----
Python: 3.12.3
Platform: Linux-6.8.0-x86_64-with-glibc2.39
Group: all
Functions: ndvi, ndwi, evi, savi, nbr, ndmi, nbr2, gci, delta_ndvi, delta_nbr, normalized_difference, temporal_mean, temporal_std, median, trend_analysis, euclidean_distance, manhattan_distance, chebyshev_distance, minkowski_distance, moving_average_temporal, moving_average_temporal_stride, pixelwise_transform, zonal_stats, binary_dilation, binary_erosion, binary_opening, binary_closing, texture_entropy
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
| ndvi                           | 93.61     | 0.60       | 92.89    | 94.36    | 1,000,000  | 10.68                       | 111.15                       | 0.10x            | 1000x1000                     |
| ndwi                           | 87.87     | 0.41       | 87.30    | 88.29    | 1,000,000  | 11.38                       | 189.66                       | 0.06x            | 1000x1000                     |
| evi                            | 108.42    | 0.57       | 107.62   | 108.91   | 1,000,000  | 9.22                        | 69.04                        | 0.13x            | 1000x1000                     |
| savi                           | 88.00     | 0.96       | 87.06    | 89.31    | 1,000,000  | 11.36                       | 156.33                       | 0.07x            | 1000x1000                     |
| nbr                            | 98.35     | 8.91       | 89.37    | 110.50   | 1,000,000  | 10.17                       | 183.38                       | 0.06x            | 1000x1000                     |
| ndmi                           | 86.85     | 0.23       | 86.54    | 87.10    | 1,000,000  | 11.51                       | 188.11                       | 0.06x            | 1000x1000                     |
| nbr2                           | 113.07    | 36.67      | 87.04    | 164.93   | 1,000,000  | 8.84                        | 180.74                       | 0.05x            | 1000x1000                     |
| gci                            | 86.24     | 2.03       | 84.76    | 89.10    | 1,000,000  | 11.60                       | 395.62                       | 0.03x            | 1000x1000                     |
| delta_ndvi                     | 262.70    | 1.67       | 260.78   | 264.85   | 1,000,000  | 3.81                        | 76.93                        | 0.05x            | 1000x1000                     |
| delta_nbr                      | 270.77    | 16.45      | 258.52   | 294.03   | 1,000,000  | 3.69                        | 84.24                        | 0.04x            | 1000x1000                     |
| normalized_difference          | 88.14     | 0.25       | 87.87    | 88.47    | 1,000,000  | 11.35                       | 188.40                       | 0.06x            | 1000x1000                     |
| temporal_mean                  | 610.57    | 27.96      | 589.19   | 650.07   | 24,000,000 | 39.31                       | 910.57                       | 0.04x            | 24x1000x1000                  |
| temporal_std                   | 1223.60   | 16.51      | 1200.47  | 1237.89  | 24,000,000 | 19.61                       | 149.15                       | 0.13x            | 24x1000x1000                  |
| median                         | 2462.81   | 4.95       | 2458.15  | 2469.66  | 24,000,000 | 9.74                        | 36.56                        | 0.27x            | 24x1000x1000                  |
| trend_analysis                 | 0.09      | 0.04       | 0.06     | 0.15     | 24         | 0.25                        | -                            | -                | T=24                          |
| euclidean_distance             | 817.66    | 6.09       | 810.85   | 825.64   | 32,000,000 | 39.14                       | 3347.13                      | 0.01x            | N=1000, M=1000, D=32          |
| manhattan_distance             | 792.98    | 4.42       | 786.77   | 796.71   | 32,000,000 | 40.35                       | 120.13                       | 0.34x            | N=1000, M=1000, D=32          |
| chebyshev_distance             | 820.30    | 12.74      | 803.05   | 833.42   | 32,000,000 | 39.01                       | 106.89                       | 0.36x            | N=1000, M=1000, D=32          |
| minkowski_distance             | 1113.61   | 8.38       | 1101.76  | 1119.67  | 32,000,000 | 28.74                       | 23.52                        | 1.22x            | N=1000, M=1000, D=32          |
| moving_average_temporal        | 3157.49   | 464.50     | 2801.94  | 3813.63  | 24,000,000 | 7.60                        | 35.68                        | 0.21x            | 24x1000x1000(win=5)           |
| moving_average_temporal_stride | 3209.63   | 150.25     | 3102.05  | 3422.10  | 24,000,000 | 7.48                        | 40.31                        | 0.19x            | 24x1000x1000(win=5, stride=4) |
| pixelwise_transform            | 1052.10   | 1.60       | 1050.42  | 1054.26  | 24,000,000 | 22.81                       | 150.83                       | 0.15x            | 24x1000x1000                  |
| zonal_stats                    | 58.60     | 0.35       | 58.30    | 59.10    | -          | -                           | -                            | 3.57x            | 1000x1000 (Zones=100)         |
| binary_dilation                | 55.65     | 1.81       | 53.17    | 57.41    | -          | -                           | -                            | 0.03x            | 1000x1000 (Kernel=3)          |
| binary_erosion                 | 45.19     | 0.19       | 44.98    | 45.45    | -          | -                           | -                            | 0.04x            | 1000x1000 (Kernel=3)          |
| binary_opening                 | 229.28    | 1.33       | 228.16   | 231.14   | -          | -                           | -                            | 0.01x            | 1000x1000 (Kernel=3)          |
| binary_closing                 | 248.73    | 0.76       | 247.70   | 249.49   | -          | -                           | -                            | 0.01x            | 1000x1000 (Kernel=3)          |
| texture_entropy                | 2618.82   | 38.66      | 2585.20  | 2672.96  | 1,000,000  | 0.38                        | 0.03                         | 12.05x           | 1000x1000 (Window=3)          |
+================================+===========+============+==========+==========+============+=============================+==============================+==================+===============================+

Speedup vs NumPy = (NumPy mean time / Rust mean time); values > 1 indicate Rust is faster.
