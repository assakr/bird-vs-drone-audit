# LCMS-DB: Lightweight Cross-Modal Sensing for Drone–Bird Discrimination

LCMS-DB is a real-time aerial object detection framework designed to accurately distinguish birds from drones in challenging low-altitude surveillance scenarios. Built on a YOLO-family backbone, it incorporates three complementary modules — **Multi-Scale Feature Enhancement (MSFE)**, **Context-Aware Attention Module (CAAM)**, and **Bidirectional Feature Fusion (BFF)** — to address the core difficulty of separating visually similar, fast-moving targets at long range. The system is benchmarked on the BirDrone dataset at both 640×640 and 1280×1280 input resolutions, with a rigorous ablation study confirming the independent and combined contribution of each module.

---

## Architecture

```
                        ┌─────────────────────────────────────────────────┐
                        │                  INPUT IMAGE                     │
                        │           (640×640 or 1280×1280)                 │
                        └──────────────────────┬──────────────────────────┘
                                               │
                        ┌──────────────────────▼──────────────────────────┐
                        │              BACKBONE (CSP-DarkNet)              │
                        │        P3 (large)  P4 (medium)  P5 (small)      │
                        └──────┬─────────────────┬──────────────┬─────────┘
                               │                 │              │
                        ┌──────▼─────────────────▼──────────────▼─────────┐
                        │        MSFE — Multi-Scale Feature Enhancement    │
                        │   (aggregates P3/P4/P5 context; Sec 5.2)        │
                        └──────────────────────┬──────────────────────────┘
                                               │
                        ┌──────────────────────▼──────────────────────────┐
                        │        CAAM — Context-Aware Attention Module     │
                        │   (spatial + channel attention weighting; 5.3)  │
                        └──────────────────────┬──────────────────────────┘
                                               │
                        ┌──────────────────────▼──────────────────────────┐
                        │          BFF — Bidirectional Feature Fusion      │
                        │   (top-down + bottom-up feature pyramid; 5.4)   │
                        └──────────────────────┬──────────────────────────┘
                                               │
                        ┌──────────────────────▼──────────────────────────┐
                        │                DETECTION HEAD                    │
                        │     (decoupled cls / box / obj branches)         │
                        └──────────────────────┬──────────────────────────┘
                                               │
                        ┌──────────────────────▼──────────────────────────┐
                        │                  PREDICTIONS                     │
                        │     [Bird, Drone] — (cx, cy, w, h, conf, cls)   │
                        └─────────────────────────────────────────────────┘
```

---

## Installation

```bash
git clone https://github.com/<your-org>/lcms_db.git
cd lcms_db
pip install -r requirements.txt
```

> **Python 3.9+** and **CUDA 11.8+** are recommended. CPU-only inference is supported but slow.

---

## Dataset Setup

LCMS-DB uses the **BirDrone** dataset. Organise it as follows:

```
lcms_db/
└── data/
    ├── images/
    │   ├── train/      ← training images (.jpg / .png)
    │   ├── val/        ← validation images
    │   └── test/       ← test images
    └── labels/
        ├── train/      ← YOLO .txt labels (one per image, same stem)
        ├── val/
        └── test/
```

Each `.txt` label file contains one bounding box per line in normalised YOLO format:

```
<class_id>  <x_center>  <y_center>  <width>  <height>
```

- `class_id`: `0` = Bird, `1` = Drone  
- All coordinates are normalised to `[0, 1]`.

Dataset metadata is defined in [`data/data.yaml`](data/data.yaml).

---

## Quick Start

### 1 — Validate annotations

```bash
# Validate training split
python data/validate_annotations.py \
    --images-dir data/images/train \
    --labels-dir data/labels/train \
    --log-file data/validation_train.log

# Validate and remove broken pairs automatically
python data/validate_annotations.py \
    --images-dir data/images/train \
    --labels-dir data/labels/train \
    --remove-invalid \
    --log-file data/validation_train.log
```

### 2 — Train (ablation 1 — baseline)

```bash
python train.py --config configs/ablation_1_baseline.yaml
```

### 3 — Train (full LCMS-DB model)

```bash
python train.py --config configs/ablation_4_full.yaml
```

### 4 — Evaluate

```bash
python evaluate.py \
    --config configs/ablation_4_full.yaml \
    --checkpoint runs/ablation_4_full/best.pt
```

### 5 — Visualise predictions

```bash
python visualize.py \
    --config configs/ablation_4_full.yaml \
    --checkpoint runs/ablation_4_full/best.pt
```

---

## Ablation Study (Section 7)

The table below reproduces the four-condition ablation from SOP Section 7.  
Run each config and fill in your measured values.

| Config | MSFE | CAAM | BFF | mAP@0.5 | mAP@0.5:0.95 | FPS |
|--------|:----:|:----:|:---:|:-------:|:------------:|:---:|
| Baseline (YOLO only) | ✗ | ✗ | ✗ | — | — | — |
| + MSFE | ✓ | ✗ | ✗ | — | — | — |
| + CAAM | ✗ | ✓ | ✗ | — | — | — |
| Full LCMS-DB | ✓ | ✓ | ✓ | — | — | — |

**Commands:**

```bash
python train.py --config configs/ablation_1_baseline.yaml   # Baseline
python train.py --config configs/ablation_2_multiscale.yaml # + MSFE
python train.py --config configs/ablation_3_context.yaml    # + CAAM
python train.py --config configs/ablation_4_full.yaml       # Full
```

---

## Resolution Comparison (Section 8)

| Resolution | mAP@0.5 | mAP@0.5:0.95 | Tiny AP | FPS | GPU Mem (GB) |
|:----------:|:-------:|:------------:|:-------:|:---:|:------------:|
| 640×640 | — | — | — | — | — |
| 1280×1280 | — | — | — | — | — |

**Commands:**

```bash
python train.py --config configs/res_640.yaml
python train.py --config configs/res_1280.yaml
```

---

## File Structure

```
lcms_db/
├── configs/
│   ├── base.yaml                   # Master config; all ablations extend this
│   ├── ablation_1_baseline.yaml    # No custom modules (pure YOLO baseline)
│   ├── ablation_2_multiscale.yaml  # MSFE only
│   ├── ablation_3_context.yaml     # CAAM only
│   ├── ablation_4_full.yaml        # Full LCMS-DB (MSFE + CAAM + BFF)
│   ├── res_640.yaml                # Full model at 640×640
│   └── res_1280.yaml               # Full model at 1280×1280
├── data/
│   ├── data.yaml                   # Dataset split paths and class names
│   ├── validate_annotations.py     # YOLO annotation QA / cleanup tool
│   ├── images/                     # Raw images (train / val / test)
│   └── labels/                     # YOLO .txt labels (train / val / test)
├── models/
│   ├── backbone.py                 # CSP-DarkNet backbone
│   ├── msfe.py                     # Multi-Scale Feature Enhancement module
│   ├── caam.py                     # Context-Aware Attention Module
│   ├── bff.py                      # Bidirectional Feature Fusion module
│   └── lcms_db.py                  # Top-level model assembly
├── train.py                        # Training entry-point
├── evaluate.py                     # Evaluation / mAP computation
├── visualize.py                    # Inference + result visualisation
├── requirements.txt                # Python package dependencies
└── README.md                       # This file
```

---

## Results

> **Note:** Fill in the tables below after running experiments.

### 11.1 — Overall Detection Performance

| Method | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 |
|--------|:-------:|:------------:|:---------:|:------:|:--:|
| Baseline | — | — | — | — | — |
| LCMS-DB (ours) | — | — | — | — | — |

### 11.2 — Per-Class Performance

| Class | AP@0.5 | AP@0.5:0.95 | Precision | Recall |
|-------|:------:|:-----------:|:---------:|:------:|
| Bird | — | — | — | — |
| Drone | — | — | — | — |

### 11.3 — Size-Based Performance (COCO Thresholds)

| Size Bucket | AP@0.5 | AP@0.5:0.95 |
|-------------|:------:|:-----------:|
| Tiny (< 32²) | — | — |
| Small (32²–96²) | — | — |
| Medium (96²–256²) | — | — |
| Large (> 256²) | — | — |

### 11.4 — Inference Speed & Efficiency

| Config | FPS (GPU) | FPS (CPU) | Params (M) | GFLOPs |
|--------|:---------:|:---------:|:----------:|:------:|
| Baseline | — | — | — | — |
| Full LCMS-DB | — | — | — | — |

### 11.5 — Confusion Matrix (Full Model)

```
              Predicted
              Bird   Drone   Background
Actual Bird  [    ] [     ] [          ]
Actual Drone [    ] [     ] [          ]
```

---

## Citation

If you use LCMS-DB in your research, please cite:

```bibtex
@article{lcmsdb2024,
  title   = {LCMS-DB: Lightweight Cross-Modal Sensing for Drone–Bird Discrimination},
  author  = {<Authors>},
  journal = {<Journal / Conference>},
  year    = {2024},
}
```

## Acknowledgements

- BirDrone dataset contributors  
- YOLOv8 / Ultralytics open-source framework  
- This work was supported by <funding agency / institution>.
