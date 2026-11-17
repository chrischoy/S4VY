| Cohort | Metric | Clips | Off | Tube 0.7 | Delta (pp) |
|---|---|---:|---:|---:|---:|
| vipseg | AQ | 343 | 70.469 | 70.522 | +0.052 |
| vipseg | SQ | 343 | 97.403 | 97.392 | -0.012 |
| vipseg | STQ | 343 | 82.419 | 82.446 | +0.027 |
| vipseg | mAP | 343 | 44.749 | 44.755 | +0.006 |
| vipseg | AP50 | 343 | 70.540 | 70.541 | +0.000 |
| vipseg | AP75 | 343 | 46.355 | 46.355 | +0.000 |
| vipseg | AP25 | 343 | 80.677 | 80.639 | -0.039 |
| vipseg | matched_mIoU | 343 | 64.579 | 64.603 | +0.025 |
| vipseg | matched_mAcc | 343 | 65.779 | 65.781 | +0.002 |
| iggt_scannet | F | 10 | 78.680 | 78.680 | +0.000 |
| iggt_scannet | J | 10 | 80.872 | 80.872 | +0.000 |
| iggt_scannet | JF | 10 | 79.776 | 79.776 | +0.000 |
| iggt_scannet | t_miou | 10 | 82.124 | 82.124 | +0.000 |
| iggt_scannet | t_sr | 10 | 80.833 | 80.833 | +0.000 |
| davis | F | 30 | 83.943 | 83.670 | -0.273 |
| davis | IoU4D | 30 | 79.141 | 78.886 | -0.254 |
| davis | J | 30 | 84.058 | 84.052 | -0.006 |
| davis | JF | 30 | 84.001 | 83.861 | -0.139 |
| davis | SR4D | 30 | 95.278 | 94.167 | -1.111 |
| davis | t_miou | 30 | 85.887 | 85.902 | +0.015 |
| davis | t_sr | 30 | 79.778 | 79.778 | +0.000 |
| lvos | F | 50 | 71.326 | 71.321 | -0.005 |
| lvos | IoU4D | 50 | 63.152 | 63.171 | +0.019 |
| lvos | J | 50 | 64.433 | 64.426 | -0.008 |
| lvos | JF | 50 | 67.880 | 67.873 | -0.007 |
| lvos | SR4D | 50 | 75.333 | 75.333 | +0.000 |
| lvos | t_miou | 50 | 68.181 | 68.182 | +0.001 |
| lvos | t_sr | 50 | 7.500 | 7.500 | +0.000 |

Warm forward + decode (mean of per-clip repetition medians):

| Cohort | Warm clips | Off (ms) | Tube 0.7 (ms) | Tube/off |
|---|---:|---:|---:|---:|
| vipseg | 343 | 477.340 | 480.950 | 1.008 |
| iggt_scannet | 10 | 120.927 | 123.602 | 1.022 |
| davis | 30 | 1897.794 | 1915.178 | 1.009 |
| lvos | 50 | 3228.816 | 3264.525 | 1.011 |

Confidence intervals not computed. Timing excludes input/output IO; not comparable to the published end-to-end speed table.
