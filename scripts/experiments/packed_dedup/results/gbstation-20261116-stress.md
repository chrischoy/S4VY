# Packed dedup CPU pilot

synthetic moving rectangles; 25% exact duplicate masks. No real predictions, GPU, or end-to-end timing.

CPU dedup only; input Boolean arrays and weights ready; packing included each call; warmup excluded.

No float64 reference run or speedup for packed-only stress. Parity established only on smaller paired cases, reported separately.

| Case | Float64 ms | Packed ms | Speedup | Float64 peak RSS MiB | Packed peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| tube400x100 | not run | 355.920 | n/a | not measured | 1048.3 |

Measured peak RSS is isolated whole-process high-water RSS. Array byte counts are estimates, not measured peaks. Single bitplane excludes weight partitions and scratch.

Python 3.12.3; NumPy 2.1.0; CPU aarch64; host 2f2ff427cc62; one BLAS thread; 1 timed calls per case.
