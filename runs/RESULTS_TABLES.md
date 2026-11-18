## LCMS-DB variants (clean test split, evaluate_v2.py)

| Run | Params (M) | GFLOPs | mAP50 | mAP50-95 | Bird AP50 | Drone AP50 | Drone R@0.25 | FP@0.25 | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| A0_plain | 8.66 | 23.8 | 12.9 | 3.9 | 4.0 | 21.9 | 29.1 | 930 | 20.3 (cuda) |
| A1_p2 | 9.10 | 33.2 | 16.0 | 5.6 | 2.7 | 29.4 | 29.3 | 1103 | 11.5 (cuda) |
| A2_p2_msfe | 11.23 | 42.8 | 23.1 | 7.5 | 10.8 | 35.4 | 41.1 | 317 | 20.1 (cuda) |
| A3_p2_msfe_cbam | 11.25 | 42.8 | 24.2 | 7.7 | 6.3 | 42.1 | 33.3 | 124 | 15.6 (cuda) |
| A4_p2_msfe_cbam_bff | 11.25 | 50.9 | 30.3 | 9.7 | 13.0 | 47.7 | 41.6 | 94 | 17.0 (cuda) |
| A5_full_ciou | 11.78 | 51.4 | 30.9 | 11.5 | 13.1 | 48.8 | 41.4 | 169 | 18.1 (cuda) |
| A6_full_wiou | 11.78 | 51.4 | 28.0 | 8.8 | 12.2 | 43.8 | 37.5 | 193 | 19.4 (cuda) |
| A7_full_wiou_scaleaware | 11.78 | 51.4 | 75.6 | 33.5 | 70.2 | 81.0 | 76.0 | 596 | 17.9 (cuda) |
| final_full_budget | 11.78 | 51.4 | 87.0 | 53.8 | 89.7 | 84.4 | 83.3 | 65 | 34.9 (cuda) |
| v2_full | 11.78 | 51.4 | 34.4 | 11.0 | 21.0 | 47.7 | 34.0 | 86 | 16.7 (cuda) |

## Baselines scored with evaluate_v2.py --yolo (same protocol and code as the LCMS-DB rows)

| Run | Params (M) | GFLOPs | mAP50 | mAP50-95 | Bird AP50 | Drone AP50 | Drone R@0.25 | FP@0.25 | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| yolov8n (COCO-pretrained) | 3.14 | 8.6 | 96.7 | 65.2 | 97.4 | 96.0 | 92.2 | 83 | 8.6 (cuda) |
| yolov8s (COCO-pretrained) | 11.14 | 28.4 | 96.8 | 65.3 | 95.9 | 97.7 | 89.5 | 39 | 9.7 (cuda) |
| yolov8s-p2 (COCO-pretrained) | 10.86 | 38.9 | 97.0 | 62.9 | 96.3 | 97.7 | 90.9 | 50 | 13.6 (cuda) |
| yolo11s (COCO-pretrained) | 9.43 | 21.4 | 95.7 | 59.7 | 94.0 | 97.4 | 93.0 | 69 | 10.4 (cuda) |
| yolov8s-scratch (from scratch) | 11.14 | 28.4 | 97.2 | 65.8 | 98.1 | 96.4 | 88.4 | 51 | 9.7 (cuda) |

## Leakage-free test subset (25 test images with no same-scene training image; tools/scene_overlap.py)

| Run | Params (M) | GFLOPs | mAP50 | mAP50-95 | Bird AP50 | Drone AP50 | Drone R@0.25 | FP@0.25 | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| final_full_budget | 11.78 | 51.4 | nan | nan | 67.1 | nan | 0.0 | 4 | 29.3 (cuda) |
| A7_full_wiou_scaleaware | 11.78 | 51.4 | nan | nan | 58.6 | nan | 0.0 | 15 | 39.6 (cuda) |
| yolov8n | 3.14 | 8.6 | nan | nan | 84.5 | nan | 0.0 | 9 | 8.6 (cuda) |
| yolov8s | 11.14 | 28.4 | nan | nan | 87.5 | nan | 0.0 | 7 | 9.4 (cuda) |
| yolov8s-p2 | 10.86 | 38.9 | nan | nan | 84.8 | nan | 0.0 | 3 | 12.9 (cuda) |
| yolo11s | 9.43 | 21.4 | nan | nan | 58.8 | nan | 0.0 | 13 | 19.8 (cuda) |
| yolov8s-scratch | 11.14 | 28.4 | nan | nan | 86.4 | nan | 0.0 | 6 | 9.5 (cuda) |

## Baselines (Ultralytics val, conf 0.001, IoU 0.6, clean test split)

| Model | Params (M) | mAP50 | mAP50-95 | Bird AP50 | Drone AP50 | P | R | ms/img |
|---|---|---|---|---|---|---|---|---|
| yolov8n | 3.14 | 97.03 | 64.11 | 97.69 | 96.37 | 94.99 | 91.31 | 8.01 |
| yolov8s | 11.14 | 96.87 | 65.39 | 95.88 | 97.85 | 96.58 | 91.09 | 10.59 |
| yolov8s-p2 | 10.86 | 96.99 | 62.49 | 96.32 | 97.65 | 94.32 | 94.0 | 14.1 |
| yolo11s | 9.43 | 96.79 | 62.55 | 95.83 | 97.75 | 94.94 | 93.86 | 11.34 |
| yolov8s-scratch | 11.14 | 97.52 | 65.59 | 98.03 | 97.01 | 96.66 | 91.96 | 8.6 |
| yolov8s-scratch_s1 | 11.14 | 98.5 | 66.86 | 98.63 | 98.37 | 95.5 | 97.02 | 9.04 |
| yolov8s-scratch_s2 | 11.14 | 97.55 | 65.79 | 97.74 | 97.35 | 96.11 | 92.39 | 12.46 |
