## Architecture ablation under both assignment rules (clean test split)

| Change | Rule | mAP50 | mAP50-95 | Bird AP50 | Drone AP50 | Small-drone AP50 |
|---|---|---|---|---|---|---|
| CSP + FPN, P3-P5, CIoU | original (A0_plain) | 12.9 | 3.9 | 4.0 | 21.9 | 52.9 |
| | scale-aware (B0_plain_sa) | 82.7 | 37.3 | 88.8 | 76.5 | 81.3 |
| + P2 level | original (A1_p2) | 16.0 | 5.6 | 2.7 | 29.4 | 62.1 |
| | scale-aware (B1_p2_sa) | 77.2 | 34.6 | 82.9 | 71.4 | 65.5 |
| + MSFE | original (A2_p2_msfe) | 23.1 | 7.5 | 10.8 | 35.4 | 54.3 |
| | scale-aware (B2_p2_msfe_sa) | 63.9 | 29.5 | 79.7 | 48.1 | 53.5 |
| + CBAM | original (A3_p2_msfe_cbam) | 24.2 | 7.7 | 6.3 | 42.1 | 72.0 |
| | scale-aware (B3_p2_msfe_cbam_sa) | 82.5 | 37.4 | 86.7 | 78.2 | 77.9 |
| + BFF | original (A4_p2_msfe_cbam_bff) | 30.3 | 9.7 | 13.0 | 47.7 | 88.8 |
| | scale-aware (B4_p2_msfe_cbam_bff_sa) | 84.0 | 39.7 | 87.1 | 80.9 | 93.8 |
| + P5 transformer | original (A5_full_ciou) | 30.9 | 11.5 | 13.1 | 48.8 | 86.6 |
| | scale-aware (B5_full_ciou_sa) | 79.5 | 38.5 | 76.6 | 82.4 | 71.4 |
| + Wise-IoU | original (A6_full_wiou) | 28.0 | 8.8 | 12.2 | 43.8 | 85.1 |
| | scale-aware (A7_full_wiou_scaleaware) | 75.6 | 33.5 | 70.2 | 81.0 | 84.4 |

## Decomposition of the scale-aware assignment (full architecture, Wise-IoU)

| Variant | mAP50 | mAP50-95 | Bird AP50 | Drone AP50 | Small-drone AP50 |
|---|---|---|---|---|---|
| Original assigner (A6) | 28.0 | 8.8 | 12.2 | 43.8 | 85.1 |
| Scale gating only: one cell per object (C2) | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 |
| Multi-positive only: all levels (C3) | 82.8 | 38.9 | 78.0 | 87.5 | 96.7 |
| Gating + multi-positive, objectness target 1.0 (C1) | 65.2 | 28.9 | 58.1 | 72.3 | 85.7 |
| Full scale-aware assignment (A7) | 75.6 | 33.5 | 70.2 | 81.0 | 84.4 |
