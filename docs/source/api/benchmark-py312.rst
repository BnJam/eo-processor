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
| ndvi                           | 93.00     | 0.34       | 92.58    | 93.41    | 1,000,000  | 10.75                       | 104.55                       | 0.10x            | 1000x1000                     |
| ndwi                           | 87.82     | 0.31       | 87.39    | 88.10    | 1,000,000  | 11.39                       | 144.60                       | 0.08x            | 1000x1000                     |
| evi                            | 107.97    | 0.84       | 107.17   | 109.14   | 1,000,000  | 9.26                        | 64.16                        | 0.14x            | 1000x1000                     |
| savi                           | 88.58     | 0.45       | 87.99    | 89.06    | 1,000,000  | 11.29                       | 119.46                       | 0.09x            | 1000x1000                     |
| nbr                            | 86.97     | 0.35       | 86.47    | 87.24    | 1,000,000  | 11.50                       | 145.26                       | 0.08x            | 1000x1000                     |
| ndmi                           | 87.91     | 0.31       | 87.67    | 88.35    | 1,000,000  | 11.37                       | 142.50                       | 0.08x            | 1000x1000                     |
| nbr2                           | 87.54     | 0.16       | 87.38    | 87.77    | 1,000,000  | 11.42                       | 146.88                       | 0.08x            | 1000x1000                     |
| gci                            | 85.03     | 0.18       | 84.80    | 85.23    | 1,000,000  | 11.76                       | 276.46                       | 0.04x            | 1000x1000                     |
| delta_ndvi                     | 258.07    | 0.92       | 257.00   | 259.25   | 1,000,000  | 3.87                        | 68.37                        | 0.06x            | 1000x1000                     |
| delta_nbr                      | 257.99    | 0.87       | 256.83   | 258.92   | 1,000,000  | 3.88                        | 66.71                        | 0.06x            | 1000x1000                     |
| normalized_difference          | 87.37     | 0.22       | 87.09    | 87.64    | 1,000,000  | 11.45                       | 143.35                       | 0.08x            | 1000x1000                     |
| temporal_mean                  | 603.45    | 7.57       | 594.36   | 612.90   | 24,000,000 | 39.77                       | 886.67                       | 0.04x            | 24x1000x1000                  |
| temporal_std                   | 1216.98   | 13.06      | 1200.52  | 1232.47  | 24,000,000 | 19.72                       | 153.47                       | 0.13x            | 24x1000x1000                  |
| median                         | 2428.33   | 6.48       | 2421.03  | 2436.78  | 24,000,000 | 9.88                        | 37.02                        | 0.27x            | 24x1000x1000                  |
| trend_analysis                 | 0.08      | 0.01       | 0.07     | 0.09     | 24         | 0.29                        | -                            | -                | T=24                          |
| euclidean_distance             | 774.87    | 3.27       | 770.44   | 778.23   | 32,000,000 | 41.30                       | 3350.48                      | 0.01x            | N=1000, M=1000, D=32          |
| manhattan_distance             | 779.00    | 16.01      | 759.35   | 798.58   | 32,000,000 | 41.08                       | 138.18                       | 0.30x            | N=1000, M=1000, D=32          |
| chebyshev_distance             | 786.92    | 3.82       | 782.35   | 791.69   | 32,000,000 | 40.66                       | 111.72                       | 0.36x            | N=1000, M=1000, D=32          |
| minkowski_distance             | 1093.02   | 1.21       | 1091.31  | 1093.97  | 32,000,000 | 29.28                       | 28.95                        | 1.01x            | N=1000, M=1000, D=32          |
| moving_average_temporal        | 2705.50   | 4.29       | 2700.15  | 2710.66  | 24,000,000 | 8.87                        | 38.68                        | 0.23x            | 24x1000x1000(win=5)           |
| moving_average_temporal_stride | 3020.06   | 5.16       | 3012.91  | 3024.89  | 24,000,000 | 7.95                        | 41.99                        | 0.19x            | 24x1000x1000(win=5, stride=4) |
| pixelwise_transform            | 1042.18   | 1.29       | 1041.22  | 1044.01  | 24,000,000 | 23.03                       | 153.79                       | 0.15x            | 24x1000x1000                  |
| zonal_stats                    | 57.66     | 0.27       | 57.38    | 58.02    | -          | -                           | -                            | 6.45x            | 1000x1000 (Zones=100)         |
| binary_dilation                | 66.68     | 3.41       | 62.89    | 71.15    | -          | -                           | -                            | 0.03x            | 1000x1000 (Kernel=3)          |
| binary_erosion                 | 53.42     | 3.60       | 50.19    | 58.44    | -          | -                           | -                            | 0.04x            | 1000x1000 (Kernel=3)          |
| binary_opening                 | 241.83    | 7.48       | 235.43   | 252.32   | -          | -                           | -                            | 0.02x            | 1000x1000 (Kernel=3)          |
| binary_closing                 | 254.91    | 1.17       | 253.32   | 256.11   | -          | -                           | -                            | 0.01x            | 1000x1000 (Kernel=3)          |
+================================+===========+============+==========+==========+============+=============================+==============================+==================+===============================+

Speedup vs NumPy = (NumPy mean time / Rust mean time); values > 1 indicate Rust is faster.
