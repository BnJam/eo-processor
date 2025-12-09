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
| ndvi                           | 92.37     | 0.44       | 91.75    | 92.73    | 1,000,000  | 10.83                       | 106.91                       | 0.10x            | 1000x1000                     |
| ndwi                           | 87.89     | 0.10       | 87.78    | 88.02    | 1,000,000  | 11.38                       | 150.60                       | 0.08x            | 1000x1000                     |
| evi                            | 109.07    | 0.41       | 108.49   | 109.39   | 1,000,000  | 9.17                        | 61.02                        | 0.15x            | 1000x1000                     |
| savi                           | 90.24     | 1.42       | 88.84    | 92.18    | 1,000,000  | 11.08                       | 145.92                       | 0.08x            | 1000x1000                     |
| nbr                            | 87.43     | 0.45       | 87.00    | 88.06    | 1,000,000  | 11.44                       | 148.68                       | 0.08x            | 1000x1000                     |
| ndmi                           | 89.49     | 2.07       | 86.60    | 91.32    | 1,000,000  | 11.17                       | 165.37                       | 0.07x            | 1000x1000                     |
| nbr2                           | 87.52     | 0.72       | 86.56    | 88.27    | 1,000,000  | 11.43                       | 177.42                       | 0.06x            | 1000x1000                     |
| gci                            | 85.93     | 0.60       | 85.21    | 86.69    | 1,000,000  | 11.64                       | 368.89                       | 0.03x            | 1000x1000                     |
| delta_ndvi                     | 259.36    | 0.17       | 259.13   | 259.52   | 1,000,000  | 3.86                        | 85.83                        | 0.04x            | 1000x1000                     |
| delta_nbr                      | 257.04    | 0.97       | 255.83   | 258.20   | 1,000,000  | 3.89                        | 85.50                        | 0.05x            | 1000x1000                     |
| normalized_difference          | 113.25    | 13.25      | 94.51    | 122.79   | 1,000,000  | 8.83                        | 149.15                       | 0.06x            | 1000x1000                     |
| temporal_mean                  | 581.00    | 7.37       | 575.38   | 591.41   | 24,000,000 | 41.31                       | 912.71                       | 0.05x            | 24x1000x1000                  |
| temporal_std                   | 1207.37   | 4.70       | 1200.73  | 1210.95  | 24,000,000 | 19.88                       | 146.81                       | 0.14x            | 24x1000x1000                  |
| median                         | 2508.63   | 52.24      | 2455.92  | 2579.82  | 24,000,000 | 9.57                        | 36.77                        | 0.26x            | 24x1000x1000                  |
| trend_analysis                 | 0.09      | 0.03       | 0.07     | 0.13     | 24         | 0.25                        | -                            | -                | T=24                          |
| euclidean_distance             | 773.18    | 5.73       | 765.74   | 779.68   | 32,000,000 | 41.39                       | 3497.08                      | 0.01x            | N=1000, M=1000, D=32          |
| manhattan_distance             | 739.58    | 1.79       | 738.17   | 742.12   | 32,000,000 | 43.27                       | 131.82                       | 0.33x            | N=1000, M=1000, D=32          |
| chebyshev_distance             | 769.71    | 1.62       | 767.70   | 771.68   | 32,000,000 | 41.57                       | 108.47                       | 0.38x            | N=1000, M=1000, D=32          |
| minkowski_distance             | 1075.58   | 5.09       | 1070.92  | 1082.66  | 32,000,000 | 29.75                       | 19.68                        | 1.51x            | N=1000, M=1000, D=32          |
| moving_average_temporal        | 2730.74   | 9.74       | 2722.42  | 2744.40  | 24,000,000 | 8.79                        | 39.29                        | 0.22x            | 24x1000x1000(win=5)           |
| moving_average_temporal_stride | 3016.04   | 15.34      | 3003.71  | 3037.67  | 24,000,000 | 7.96                        | 43.54                        | 0.18x            | 24x1000x1000(win=5, stride=4) |
| pixelwise_transform            | 1037.92   | 2.58       | 1035.00  | 1041.28  | 24,000,000 | 23.12                       | 155.71                       | 0.15x            | 24x1000x1000                  |
| zonal_stats                    | 57.48     | 0.09       | 57.37    | 57.58    | -          | -                           | -                            | 3.14x            | 1000x1000 (Zones=100)         |
| binary_dilation                | 53.88     | 1.52       | 51.81    | 55.40    | -          | -                           | -                            | 0.03x            | 1000x1000 (Kernel=3)          |
| binary_erosion                 | 44.45     | 0.45       | 44.07    | 45.09    | -          | -                           | -                            | 0.04x            | 1000x1000 (Kernel=3)          |
| binary_opening                 | 228.92    | 0.81       | 227.79   | 229.62   | -          | -                           | -                            | 0.01x            | 1000x1000 (Kernel=3)          |
| binary_closing                 | 247.51    | 0.18       | 247.28   | 247.70   | -          | -                           | -                            | 0.01x            | 1000x1000 (Kernel=3)          |
+================================+===========+============+==========+==========+============+=============================+==============================+==================+===============================+

Speedup vs NumPy = (NumPy mean time / Rust mean time); values > 1 indicate Rust is faster.
