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
| ndvi                           | 106.34    | 2.25       | 103.16   | 108.04   | 1,000,000  | 9.40                        | 79.18                        | 0.12x            | 1000x1000                     |
| ndwi                           | 96.53     | 1.15       | 95.40    | 98.12    | 1,000,000  | 10.36                       | 133.09                       | 0.08x            | 1000x1000                     |
| evi                            | 118.93    | 0.82       | 118.04   | 120.02   | 1,000,000  | 8.41                        | 54.38                        | 0.15x            | 1000x1000                     |
| savi                           | 97.48     | 0.66       | 96.59    | 98.17    | 1,000,000  | 10.26                       | 116.65                       | 0.09x            | 1000x1000                     |
| nbr                            | 96.60     | 1.78       | 95.33    | 99.11    | 1,000,000  | 10.35                       | 135.38                       | 0.08x            | 1000x1000                     |
| ndmi                           | 99.34     | 7.32       | 94.00    | 109.70   | 1,000,000  | 10.07                       | 156.41                       | 0.06x            | 1000x1000                     |
| nbr2                           | 108.66    | 4.03       | 103.20   | 112.81   | 1,000,000  | 9.20                        | 147.37                       | 0.06x            | 1000x1000                     |
| gci                            | 103.95    | 0.48       | 103.49   | 104.62   | 1,000,000  | 9.62                        | 278.87                       | 0.03x            | 1000x1000                     |
| delta_ndvi                     | 315.72    | 17.59      | 296.28   | 338.87   | 1,000,000  | 3.17                        | 67.28                        | 0.05x            | 1000x1000                     |
| delta_nbr                      | 308.20    | 16.86      | 284.38   | 321.09   | 1,000,000  | 3.24                        | 64.56                        | 0.05x            | 1000x1000                     |
| normalized_difference          | 95.47     | 2.15       | 93.91    | 98.51    | 1,000,000  | 10.47                       | 139.82                       | 0.07x            | 1000x1000                     |
| temporal_mean                  | 584.41    | 0.47       | 583.76   | 584.87   | 24,000,000 | 41.07                       | 920.30                       | 0.04x            | 24x1000x1000                  |
| temporal_std                   | 1211.24   | 17.17      | 1187.35  | 1226.93  | 24,000,000 | 19.81                       | 150.16                       | 0.13x            | 24x1000x1000                  |
| median                         | 2505.13   | 133.31     | 2410.77  | 2693.65  | 24,000,000 | 9.58                        | 36.67                        | 0.26x            | 24x1000x1000                  |
| trend_analysis                 | 0.10      | 0.00       | 0.09     | 0.10     | 24         | 0.25                        | -                            | -                | T=24                          |
| euclidean_distance             | 780.82    | 10.61      | 766.37   | 791.53   | 32,000,000 | 40.98                       | 3387.91                      | 0.01x            | N=1000, M=1000, D=32          |
| manhattan_distance             | 747.29    | 5.94       | 739.15   | 753.14   | 32,000,000 | 42.82                       | 132.89                       | 0.32x            | N=1000, M=1000, D=32          |
| chebyshev_distance             | 843.60    | 44.74      | 780.81   | 881.71   | 32,000,000 | 37.93                       | 108.83                       | 0.35x            | N=1000, M=1000, D=32          |
| minkowski_distance             | 1086.48   | 4.62       | 1080.03  | 1090.57  | 32,000,000 | 29.45                       | 24.65                        | 1.19x            | N=1000, M=1000, D=32          |
| moving_average_temporal        | 2828.82   | 126.76     | 2703.84  | 3002.61  | 24,000,000 | 8.48                        | 38.21                        | 0.22x            | 24x1000x1000(win=5)           |
| moving_average_temporal_stride | 3096.06   | 80.06      | 3007.90  | 3201.66  | 24,000,000 | 7.75                        | 42.60                        | 0.18x            | 24x1000x1000(win=5, stride=4) |
| pixelwise_transform            | 1044.64   | 10.14      | 1036.77  | 1058.95  | 24,000,000 | 22.97                       | 149.80                       | 0.15x            | 24x1000x1000                  |
| zonal_stats                    | 57.98     | 0.49       | 57.47    | 58.63    | -          | -                           | -                            | 3.43x            | 1000x1000 (Zones=100)         |
| binary_dilation                | 52.52     | 0.22       | 52.21    | 52.70    | -          | -                           | -                            | 0.03x            | 1000x1000 (Kernel=3)          |
| binary_erosion                 | 46.39     | 2.21       | 44.74    | 49.52    | -          | -                           | -                            | 0.03x            | 1000x1000 (Kernel=3)          |
| binary_opening                 | 229.41    | 1.34       | 228.11   | 231.26   | -          | -                           | -                            | 0.01x            | 1000x1000 (Kernel=3)          |
| binary_closing                 | 248.33    | 0.76       | 247.27   | 249.00   | -          | -                           | -                            | 0.01x            | 1000x1000 (Kernel=3)          |
+================================+===========+============+==========+==========+============+=============================+==============================+==================+===============================+

Speedup vs NumPy = (NumPy mean time / Rust mean time); values > 1 indicate Rust is faster.
