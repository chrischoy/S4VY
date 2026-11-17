| Cohort | Metric | Clips | Off | Tube 0.7 | Delta (pp) |
|---|---|---:|---:|---:|---:|
| iggt_scannet | F | 1 | 85.360 | 85.360 | +0.000 |
| iggt_scannet | J | 1 | 91.181 | 91.181 | +0.000 |
| iggt_scannet | JF | 1 | 88.270 | 88.270 | +0.000 |
| iggt_scannet | t_miou | 1 | 90.970 | 90.970 | +0.000 |
| iggt_scannet | t_sr | 1 | 100.000 | 100.000 | +0.000 |
| vipseg | AQ | 1 | 64.530 | 64.774 | +0.243 |
| vipseg | SQ | 1 | 96.444 | 96.442 | -0.002 |
| vipseg | STQ | 1 | 78.889 | 79.037 | +0.148 |
| vipseg | mAP | 1 | 43.568 | 43.568 | +0.000 |
| vipseg | AP50 | 1 | 74.519 | 74.519 | +0.000 |
| vipseg | AP75 | 1 | 52.415 | 52.415 | +0.000 |
| vipseg | AP25 | 1 | 81.250 | 81.250 | +0.000 |
| vipseg | matched_mIoU | 1 | 61.510 | 61.510 | +0.000 |
| vipseg | matched_mAcc | 1 | 64.427 | 64.427 | +0.000 |
| davis | F | 1 | 77.557 | 77.557 | +0.000 |
| davis | IoU4D | 1 | 70.640 | 70.640 | +0.000 |
| davis | J | 1 | 79.765 | 79.765 | +0.000 |
| davis | JF | 1 | 78.661 | 78.661 | +0.000 |
| davis | SR4D | 1 | 100.000 | 100.000 | +0.000 |
| davis | t_miou | 1 | 79.135 | 79.135 | +0.000 |
| davis | t_sr | 1 | 100.000 | 100.000 | +0.000 |
| lvos | F | 1 | 88.697 | 88.697 | +0.000 |
| lvos | IoU4D | 1 | 86.212 | 86.212 | +0.000 |
| lvos | J | 1 | 71.503 | 71.503 | +0.000 |
| lvos | JF | 1 | 80.100 | 80.100 | +0.000 |
| lvos | SR4D | 1 | 100.000 | 100.000 | +0.000 |
| lvos | t_miou | 1 | 81.841 | 81.841 | +0.000 |
| lvos | t_sr | 1 | 0.000 | 0.000 | +0.000 |

Warm forward + decode (mean of per-clip repetition medians):

| Cohort | Warm clips | Off (ms) | Tube 0.7 (ms) | Tube/off |
|---|---:|---:|---:|---:|
| iggt_scannet | 1 | 110.563 | 113.692 | 1.028 |
| vipseg | 1 | 241.480 | 245.170 | 1.015 |
| davis | 1 | 1829.156 | 1850.943 | 1.012 |
| lvos | 1 | 3690.111 | 3697.257 | 1.002 |

Confidence intervals not computed. Timing excludes input/output IO; not comparable to the published end-to-end speed table.
