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
| ndvi                           | 95.84     | 2.32       | 92.65    | 98.11    | 1,000,000  | 10.43                       | 97.80                        | 0.11x            | 1000x1000                     |
| ndwi                           | 92.49     | 4.17       | 88.68    | 98.29    | 1,000,000  | 10.81                       | 135.58                       | 0.08x            | 1000x1000                     |
| evi                            | 111.23    | 2.24       | 108.09   | 113.23   | 1,000,000  | 8.99                        | 68.32                        | 0.13x            | 1000x1000                     |
| savi                           | 89.81     | 2.13       | 87.87    | 92.78    | 1,000,000  | 11.13                       | 114.90                       | 0.10x            | 1000x1000                     |
| nbr                            | 90.91     | 2.32       | 87.83    | 93.41    | 1,000,000  | 11.00                       | 144.62                       | 0.08x            | 1000x1000                     |
| ndmi                           | 87.84     | 0.14       | 87.65    | 88.01    | 1,000,000  | 11.38                       | 144.47                       | 0.08x            | 1000x1000                     |
| nbr2                           | 87.66     | 0.32       | 87.39    | 88.11    | 1,000,000  | 11.41                       | 115.87                       | 0.10x            | 1000x1000                     |
| gci                            | 84.81     | 0.26       | 84.55    | 85.17    | 1,000,000  | 11.79                       | 283.07                       | 0.04x            | 1000x1000                     |
| delta_ndvi                     | 256.91    | 1.11       | 255.37   | 257.93   | 1,000,000  | 3.89                        | 74.40                        | 0.05x            | 1000x1000                     |
| delta_nbr                      | 267.28    | 12.69      | 257.27   | 285.18   | 1,000,000  | 3.74                        | 66.87                        | 0.06x            | 1000x1000                     |
| normalized_difference          | 88.01     | 0.51       | 87.29    | 88.43    | 1,000,000  | 11.36                       | 144.57                       | 0.08x            | 1000x1000                     |
| temporal_mean                  | 599.29    | 5.36       | 592.64   | 605.76   | 24,000,000 | 40.05                       | 889.98                       | 0.04x            | 24x1000x1000                  |
| temporal_std                   | 1210.81   | 25.58      | 1178.59  | 1241.17  | 24,000,000 | 19.82                       | 139.12                       | 0.14x            | 24x1000x1000                  |
| median                         | 2440.02   | 17.41      | 2423.46  | 2464.08  | 24,000,000 | 9.84                        | 36.74                        | 0.27x            | 24x1000x1000                  |
| trend_analysis                 | 0.09      | 0.00       | 0.09     | 0.10     | 24         | 0.27                        | -                            | -                | T=24                          |
| euclidean_distance             | 779.56    | 24.10      | 758.90   | 813.36   | 32,000,000 | 41.05                       | 3467.98                      | 0.01x            | N=1000, M=1000, D=32          |
| manhattan_distance             | 748.22    | 12.81      | 738.37   | 766.32   | 32,000,000 | 42.77                       | 129.95                       | 0.33x            | N=1000, M=1000, D=32          |
| chebyshev_distance             | 778.15    | 8.14       | 769.20   | 788.89   | 32,000,000 | 41.12                       | 109.15                       | 0.38x            | N=1000, M=1000, D=32          |
| minkowski_distance             | 1105.87   | 20.01      | 1081.32  | 1130.33  | 32,000,000 | 28.94                       | 28.62                        | 1.01x            | N=1000, M=1000, D=32          |
| moving_average_temporal        | 3452.81   | 698.09     | 2736.66  | 4399.39  | 24,000,000 | 6.95                        | 36.49                        | 0.19x            | 24x1000x1000(win=5)           |
| moving_average_temporal_stride | 3196.56   | 169.91     | 3075.22  | 3436.85  | 24,000,000 | 7.51                        | 41.36                        | 0.18x            | 24x1000x1000(win=5, stride=4) |
| pixelwise_transform            | 1230.16   | 141.54     | 1052.95  | 1399.37  | 24,000,000 | 19.51                       | 127.92                       | 0.15x            | 24x1000x1000                  |
| zonal_stats                    | 96.76     | 3.76       | 91.47    | 99.88    | -          | -                           | -                            | 3.66x            | 1000x1000 (Zones=100)         |
| binary_dilation                | 75.27     | 4.62       | 68.80    | 79.32    | -          | -                           | -                            | 0.04x            | 1000x1000 (Kernel=3)          |
| binary_erosion                 | 62.81     | 1.21       | 61.10    | 63.80    | -          | -                           | -                            | 0.04x            | 1000x1000 (Kernel=3)          |
| binary_opening                 | 331.47    | 10.86      | 320.76   | 346.36   | -          | -                           | -                            | 0.01x            | 1000x1000 (Kernel=3)          |
| binary_closing                 | 358.51    | 26.76      | 322.84   | 387.30   | -          | -                           | -                            | 0.01x            | 1000x1000 (Kernel=3)          |
| texture_entropy                | 3170.49   | 304.54     | 2836.78  | 3573.13  | 1,000,000  | 0.32                        | 0.03                         | 10.71x           | 1000x1000 (Window=3)          |
+================================+===========+============+==========+==========+============+=============================+==============================+==================+===============================+

Speedup vs NumPy = (NumPy mean time / Rust mean time); values > 1 indicate Rust is faster.
