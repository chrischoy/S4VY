| Cohort | Metric | Scored / selected clips | Owned, dedup off | Owned + tube 0.7 | Delta (pp) |
|---|---|---:|---:|---:|---:|
| ScanNet (IGGT) | J&F | 10 / 10 | 79.776 | 79.776 | +0.000 |
| DAVIS 2017 | J&F | 30 / 30 | 84.001 | 83.861 | -0.139 |
| DAVIS 2017 | IoU4D | 30 / 30 | 79.141 | 78.886 | -0.254 |
| LVOS | J&F | 50 / 50 | 67.880 | 67.873 | -0.007 |
| LVOS | IoU4D | 50 / 50 | 63.152 | 63.171 | +0.019 |
| VIPSeg | STQ | 343 / 343 | 82.419 | 82.446 | +0.027 |
| VIPSeg | mAP | 343 / 343 | 44.749 | 44.755 | +0.006 |

Warm forward + decode, milliseconds per clip (mean of per-clip medians of three repeats):

| Cohort | Warm clips | Dedup off (ms) | Tube 0.7 (ms) | Tube/off |
|---|---:|---:|---:|---:|
| ScanNet (IGGT) | 10 | 120.927 | 123.602 | 1.022 |
| DAVIS 2017 | 30 | 1897.794 | 1915.178 | 1.009 |
| LVOS | 50 | 3228.816 | 3264.525 | 1.011 |
| VIPSeg | 343 | 477.340 | 480.950 | 1.008 |

Measured 2026-11-16–17 UTC on one NVIDIA GB300: 433 clips / 15,327 frames, public checkpoint, native input, full acceleration.
Accuracy is reported in percent; deltas are percentage points. VIPSeg mAP is pooled.
Timing includes dedup preparation, GPU-to-CPU transfer, packing, suppression and ownership; it excludes input loading, output writing, initial compilation and graph capture.
These timing scopes differ from the end-to-end speed table. Confidence intervals were not computed.
CPU scoring ran concurrently with six workers; the GPU also retained the idle demo and another resident service. Timings reflect this shared workstation.
Helper parity passed seven unit tests, five paired synthetic benchmark cases and a bounded real-tensor subset; this does not establish full-size W8 framewise decoder parity.
