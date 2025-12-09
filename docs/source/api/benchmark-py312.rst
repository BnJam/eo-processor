Benchmark Report
================

Meta
----
Python: 3.12.3
Platform: Linux-6.8.0-x86_64-with-glibc2.39
Group: all
Functions: ndvi, ndwi, evi, savi, nbr, ndmi, nbr2, gci, delta_ndvi, delta_nbr, normalized_difference, temporal_mean, temporal_std, median, trend_analysis, euclidean_distance, manhattan_distance, chebyshev_distance, minkowski_distance, moving_average_temporal, moving_average_temporal_stride, pixelwise_transform, binary_dilation, binary_erosion, binary_opening, binary_closing
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
| ndvi                           | 92.54     | 0.24       | 92.20    | 92.76    | 1,000,000  | 10.81                       | 95.31                        | 0.11x            | 1000x1000                     |
| ndwi                           | 88.99     | 1.79       | 86.51    | 90.66    | 1,000,000  | 11.24                       | 153.77                       | 0.07x            | 1000x1000                     |
| evi                            | 111.56    | 3.29       | 108.07   | 115.97   | 1,000,000  | 8.96                        | 62.88                        | 0.14x            | 1000x1000                     |
| savi                           | 88.66     | 0.47       | 88.10    | 89.24    | 1,000,000  | 11.28                       | 122.76                       | 0.09x            | 1000x1000                     |
| nbr                            | 88.78     | 1.57       | 86.57    | 89.95    | 1,000,000  | 11.26                       | 151.55                       | 0.07x            | 1000x1000                     |
| ndmi                           | 87.71     | 1.40       | 86.35    | 89.64    | 1,000,000  | 11.40                       | 153.33                       | 0.07x            | 1000x1000                     |
| nbr2                           | 88.17     | 1.29       | 86.39    | 89.39    | 1,000,000  | 11.34                       | 154.39                       | 0.07x            | 1000x1000                     |
| gci                            | 86.65     | 1.57       | 84.43    | 87.81    | 1,000,000  | 11.54                       | 311.88                       | 0.04x            | 1000x1000                     |
| delta_ndvi                     | 262.66    | 1.81       | 261.13   | 265.21   | 1,000,000  | 3.81                        | 77.23                        | 0.05x            | 1000x1000                     |
| delta_nbr                      | 255.20    | 0.66       | 254.68   | 256.12   | 1,000,000  | 3.92                        | 72.97                        | 0.05x            | 1000x1000                     |
| normalized_difference          | 87.62     | 0.17       | 87.42    | 87.84    | 1,000,000  | 11.41                       | 146.67                       | 0.08x            | 1000x1000                     |
| temporal_mean                  | 585.60    | 1.80       | 583.52   | 587.90   | 24,000,000 | 40.98                       | 937.31                       | 0.04x            | 24x1000x1000                  |
| temporal_std                   | 1207.76   | 24.05      | 1177.12  | 1235.85  | 24,000,000 | 19.87                       | 154.47                       | 0.13x            | 24x1000x1000                  |
| median                         | 2403.10   | 4.03       | 2399.03  | 2408.59  | 24,000,000 | 9.99                        | 37.15                        | 0.27x            | 24x1000x1000                  |
| trend_analysis                 | 0.09      | 0.01       | 0.08     | 0.10     | 24         | 0.26                        | -                            | -                | T=24                          |
| euclidean_distance             | 772.33    | 1.38       | 770.63   | 774.02   | 32,000,000 | 41.43                       | 3523.72                      | 0.01x            | N=1000, M=1000, D=32          |
| manhattan_distance             | 791.04    | 44.42      | 759.06   | 853.85   | 32,000,000 | 40.45                       | 134.37                       | 0.30x            | N=1000, M=1000, D=32          |
| chebyshev_distance             | 798.27    | 4.27       | 794.31   | 804.20   | 32,000,000 | 40.09                       | 112.40                       | 0.36x            | N=1000, M=1000, D=32          |
| minkowski_distance             | 1103.49   | 9.04       | 1093.48  | 1115.37  | 32,000,000 | 29.00                       | 27.47                        | 1.06x            | N=1000, M=1000, D=32          |
| moving_average_temporal        | 2920.71   | 93.45      | 2789.44  | 2999.57  | 24,000,000 | 8.22                        | 39.07                        | 0.21x            | 24x1000x1000(win=5)           |
| moving_average_temporal_stride | 3047.34   | 24.89      | 3024.69  | 3082.00  | 24,000,000 | 7.88                        | 42.86                        | 0.18x            | 24x1000x1000(win=5, stride=4) |
| pixelwise_transform            | 1042.96   | 2.51       | 1039.57  | 1045.56  | 24,000,000 | 23.01                       | 156.98                       | 0.15x            | 24x1000x1000                  |
| binary_dilation                | 53.41     | 0.45       | 53.03    | 54.05    | -          | -                           | -                            | 0.04x            | 1000x1000 (Kernel=3)          |
| binary_erosion                 | 46.18     | 0.78       | 45.31    | 47.20    | -          | -                           | -                            | 0.04x            | 1000x1000 (Kernel=3)          |
| binary_opening                 | 240.41    | 11.06      | 231.68   | 256.01   | -          | -                           | -                            | 0.02x            | 1000x1000 (Kernel=3)          |
| binary_closing                 | 255.16    | 2.35       | 252.60   | 258.27   | -          | -                           | -                            | 0.01x            | 1000x1000 (Kernel=3)          |
+================================+===========+============+==========+==========+============+=============================+==============================+==================+===============================+

Speedup vs NumPy = (NumPy mean time / Rust mean time); values > 1 indicate Rust is faster.
