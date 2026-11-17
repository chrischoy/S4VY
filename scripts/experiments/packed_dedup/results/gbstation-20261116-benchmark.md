# Packed dedup CPU pilot

synthetic moving rectangles; 25% exact duplicate masks. No real predictions, GPU, or end-to-end timing.

CPU dedup only; input Boolean arrays and weights ready; packing included each call; warmup excluded.

Exact keep, keeper, and IoU equality required before reporting each case.

| Case | Float64 ms | Packed ms | Speedup | Float64 peak RSS MiB | Packed peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| frame64 | 10.032 | 0.711 | 14.11x | 55.8 | 36.1 |
| frame128 | 28.901 | 1.084 | 26.66x | 79.6 | 37.3 |
| tube32x8 | 26.377 | 3.115 | 8.47x | 127.8 | 44.1 |
| frame400 | 237.267 | 3.927 | 60.42x | 183.2 | 44.4 |
| tube64x8 | 76.584 | 3.973 | 19.28x | 220.7 | 49.6 |

Measured peak RSS is isolated whole-process high-water RSS. Array byte counts are estimates, not measured peaks. Single bitplane excludes weight partitions and scratch.

Python 3.12.3; NumPy 2.1.0; CPU aarch64; host 2f2ff427cc62; one BLAS thread; 5 timed calls per case.
