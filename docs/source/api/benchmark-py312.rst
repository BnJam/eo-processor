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
| ndvi                           | 93.88     | 1.31       | 92.90    | 95.73    | 1,000,000  | 10.65                       | 96.35                        | 0.11x            | 1000x1000                     |
| ndwi                           | 89.11     | 1.12       | 87.52    | 90.02    | 1,000,000  | 11.22                       | 185.94                       | 0.06x            | 1000x1000                     |
| evi                            | 108.37    | 0.46       | 107.96   | 109.02   | 1,000,000  | 9.23                        | 71.90                        | 0.13x            | 1000x1000                     |
| savi                           | 88.40     | 0.25       | 88.05    | 88.63    | 1,000,000  | 11.31                       | 150.39                       | 0.08x            | 1000x1000                     |
| nbr                            | 86.49     | 0.32       | 86.21    | 86.93    | 1,000,000  | 11.56                       | 189.41                       | 0.06x            | 1000x1000                     |
| ndmi                           | 87.07     | 1.12       | 86.09    | 88.64    | 1,000,000  | 11.48                       | 189.86                       | 0.06x            | 1000x1000                     |
| nbr2                           | 86.84     | 0.21       | 86.55    | 87.04    | 1,000,000  | 11.51                       | 187.17                       | 0.06x            | 1000x1000                     |
| gci                            | 84.61     | 0.18       | 84.41    | 84.84    | 1,000,000  | 11.82                       | 360.20                       | 0.03x            | 1000x1000                     |
| delta_ndvi                     | 258.19    | 0.88       | 256.98   | 259.05   | 1,000,000  | 3.87                        | 76.17                        | 0.05x            | 1000x1000                     |
| delta_nbr                      | 256.50    | 1.41       | 255.33   | 258.47   | 1,000,000  | 3.90                        | 79.07                        | 0.05x            | 1000x1000                     |
| normalized_difference          | 86.78     | 0.49       | 86.26    | 87.44    | 1,000,000  | 11.52                       | 184.22                       | 0.06x            | 1000x1000                     |
| temporal_mean                  | 588.94    | 0.82       | 588.15   | 590.07   | 24,000,000 | 40.75                       | 858.77                       | 0.05x            | 24x1000x1000                  |
| temporal_std                   | 1201.67   | 17.09      | 1177.73  | 1216.55  | 24,000,000 | 19.97                       | 149.49                       | 0.13x            | 24x1000x1000                  |
| median                         | 2416.20   | 6.72       | 2408.93  | 2425.13  | 24,000,000 | 9.93                        | 37.06                        | 0.27x            | 24x1000x1000                  |
| trend_analysis                 | 0.10      | 0.03       | 0.07     | 0.13     | 24         | 0.25                        | -                            | -                | T=24                          |
| euclidean_distance             | 793.71    | 8.60       | 782.49   | 803.38   | 32,000,000 | 40.32                       | 3404.46                      | 0.01x            | N=1000, M=1000, D=32          |
| manhattan_distance             | 750.15    | 2.62       | 746.97   | 753.38   | 32,000,000 | 42.66                       | 136.27                       | 0.31x            | N=1000, M=1000, D=32          |
| chebyshev_distance             | 771.37    | 1.97       | 769.23   | 773.99   | 32,000,000 | 41.48                       | 113.12                       | 0.37x            | N=1000, M=1000, D=32          |
| minkowski_distance             | 1075.32   | 0.89       | 1074.07  | 1076.08  | 32,000,000 | 29.76                       | 28.94                        | 1.03x            | N=1000, M=1000, D=32          |
| moving_average_temporal        | 2737.41   | 14.86      | 2720.59  | 2756.72  | 24,000,000 | 8.77                        | 38.01                        | 0.23x            | 24x1000x1000(win=5)           |
| moving_average_temporal_stride | 3061.85   | 29.22      | 3039.48  | 3103.12  | 24,000,000 | 7.84                        | 41.68                        | 0.19x            | 24x1000x1000(win=5, stride=4) |
| pixelwise_transform            | 1044.50   | 4.17       | 1039.01  | 1049.11  | 24,000,000 | 22.98                       | 155.50                       | 0.15x            | 24x1000x1000                  |
| zonal_stats                    | 57.69     | 0.12       | 57.57    | 57.86    | -          | -                           | -                            | 4.47x            | 1000x1000 (Zones=100)         |
| binary_dilation                | 54.60     | 1.46       | 52.77    | 56.34    | -          | -                           | -                            | 0.04x            | 1000x1000 (Kernel=3)          |
| binary_erosion                 | 46.72     | 0.66       | 45.81    | 47.33    | -          | -                           | -                            | 0.05x            | 1000x1000 (Kernel=3)          |
| binary_opening                 | 232.04    | 0.69       | 231.31   | 232.97   | -          | -                           | -                            | 0.02x            | 1000x1000 (Kernel=3)          |
| binary_closing                 | 251.64    | 0.64       | 250.74   | 252.19   | -          | -                           | -                            | 0.01x            | 1000x1000 (Kernel=3)          |
+================================+===========+============+==========+==========+============+=============================+==============================+==================+===============================+

Speedup vs NumPy = (NumPy mean time / Rust mean time); values > 1 indicate Rust is faster.
