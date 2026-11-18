## Kaggle test split (scenes also present in training) vs external test set (other source)

| Model | Test set | Images | mAP50 | mAP50-95 | Bird AP50 | Drone AP50 | Bird R@0.25 | Drone R@0.25 | FP@0.25 |
|---|---|---|---|---|---|---|---|---|---|
| LCMS-DB (final) | Kaggle test | 889 | 87.0 | 53.8 | 89.7 | 84.4 | 84.7 | 83.3 | 65 |
| LCMS-DB (final) | External | 2773 | 0.2 | 0.1 | 0.4 | 0.0 | 0.3 | 0.0 | 512 |
| LCMS-DB A7 | Kaggle test | 889 | 75.6 | 33.5 | 70.2 | 81.0 | 80.0 | 76.0 | 596 |
| LCMS-DB A7 | External | 2773 | 0.0 | 0.0 | 0.0 | 0.0 | 0.2 | 0.0 | 2774 |
| YOLOv8s (scratch) | Kaggle test | 889 | 97.2 | 65.8 | 98.1 | 96.4 | 96.1 | 88.4 | 51 |
| YOLOv8s (scratch) | External | 2773 | 3.8 | 1.1 | 7.3 | 0.3 | 11.6 | 0.0 | 1660 |
| YOLOv8s (COCO) | Kaggle test | 889 | 96.8 | 65.3 | 95.9 | 97.7 | 93.9 | 89.5 | 39 |
| YOLOv8s (COCO) | External | 2773 | 3.6 | 1.1 | 7.1 | 0.2 | 7.9 | 0.0 | 839 |
| YOLOv8n (COCO) | Kaggle test | 889 | 96.7 | 65.2 | 97.4 | 96.0 | 96.1 | 92.2 | 83 |
| YOLOv8n (COCO) | External | 2773 | 2.9 | 0.9 | 5.8 | 0.1 | 7.6 | 0.0 | 1021 |
| YOLOv8s-P2 (COCO) | Kaggle test | 889 | 97.0 | 62.9 | 96.3 | 97.7 | 92.9 | 90.9 | 50 |
| YOLOv8s-P2 (COCO) | External | 2773 | 5.2 | 1.6 | 10.0 | 0.5 | 12.5 | 0.0 | 1152 |
| YOLO11s (COCO) | Kaggle test | 889 | 95.7 | 59.7 | 94.0 | 97.4 | 92.6 | 93.0 | 69 |
| YOLO11s (COCO) | External | 2773 | 2.2 | 0.7 | 4.2 | 0.1 | 5.2 | 0.0 | 722 |

## Change in mAP50 from the Kaggle test split to the external set

| Model | Kaggle test | External | Change |
|---|---|---|---|
| LCMS-DB (final) | 87.0 | 0.2 | -86.8 |
| LCMS-DB A7 | 75.6 | 0.0 | -75.5 |
| YOLOv8s (scratch) | 97.2 | 3.8 | -93.5 |
| YOLOv8s (COCO) | 96.8 | 3.6 | -93.2 |
| YOLOv8n (COCO) | 96.7 | 2.9 | -93.8 |
| YOLOv8s-P2 (COCO) | 97.0 | 5.2 | -91.8 |
| YOLO11s (COCO) | 95.7 | 2.2 | -93.5 |
