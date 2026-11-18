## Multi-seed results (test split, mean ± sample std over runs)

| Config | Runs | mAP50 | mAP50-95 | Bird AP50 | Drone AP50 |
|---|---|---|---|---|---|
| A5_full_ciou | 3 | 29.0 ± 2.2 | 9.9 ± 1.5 | 11.3 ± 2.3 | 46.6 ± 2.2 |
| A6_full_wiou | 3 | 29.0 ± 9.6 | 9.8 ± 3.5 | 14.1 ± 11.4 | 43.9 ± 7.8 |
| A7_full_wiou_scaleaware | 3 | 79.2 ± 3.4 | 35.3 ± 1.7 | 77.6 ± 6.7 | 80.8 ± 4.2 |
| final_full_budget | 1 | 87.0 (n=1) | 53.8 (n=1) | 89.7 (n=1) | 84.4 (n=1) |
| yolov8s-scratch | 3 | 97.3 ± 0.2 | 66.3 ± 0.7 | 97.6 ± 0.5 | 97.0 ± 0.6 |

A7 - A6 (assignment effect), mAP50: +50.2 points (min A7 75.6 vs max A6 39.1)
A6 - A5 (Wise-IoU vs CIoU), mAP50: +0.1 points
