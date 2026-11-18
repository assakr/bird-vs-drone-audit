# Bird-vs-Drone Benchmark Audit

Code, configurations and results for the paper
**"What Really Drives Bird-vs-Drone Detection Results? Label Errors, Scene Leakage, Evaluation Pitfalls, and a Controlled Study of Target Assignment"** (under review).

The paper examines what a near-97% mAP@0.5 on a widely used Kaggle bird-vs-drone dataset actually measures. Its main findings:

| Finding | Result |
|---|---|
| Validation labels | At least 65.8% of validation **bird** images carry misaligned boxes; test labels are reliable (0 of 150 reviewed misaligned) |
| Scene leakage | An estimated 98.4% of test images share a scene with a training image |
| Evaluation pitfalls | Errors in an evaluation script raised mAP@0.5 by 17.7 points and mAP@0.5:0.95 by 41.0 points on identical detections |
| Assignment vs architecture | The target-assignment rule (several positive cells per object) raised mAP@0.5 by about 50 points; five added modules gave no gain over a plain detector once the assignment worked |
| Cross-dataset test | Detectors that reach 95.7–97.3% on the benchmark reach at most 5.2% on an external set of small, distant targets |

The detector studied in the paper, **LCMS-DB**, is included as a case study. It is not presented as an improvement over standard detectors: YOLOv8s trained from scratch beats it by 10.2 points of mAP@0.5.

---

## Repository layout

```
configs/      Training configurations for every run in the paper (A0–A7, B0–B5, C1–C3, seed repeats, final model)
data/         Dataset loading and augmentation; data_*.yaml hold the dataset paths
models/       LCMS-DB backbone, MSFE, CBAM (caam.py), BFF and detection head
losses/       Loss with both assignment rules (original and multi-positive, scale-aware)
train.py      Training entry point for LCMS-DB
tools/        Audit, evaluation and reporting scripts (see below)
runs/         Saved results for every run: eval_v2.json, per-image detections, training logs, result tables
sanity_checks*/  Label overlays from an early visual check (dataset images, see "Data sources")
```

Model checkpoints (`*.pt`) are not stored in the repository because of their size.

### Scripts in `tools/`

| Script | Purpose |
|---|---|
| `make_clean_split.py` | Builds the clean split: 90% of train for training; 10% of train plus validation drone images for model selection; original test unchanged |
| `label_audit.py` | Model-assisted audit of every label (`scan`), then summary after human review (`summarize`) |
| `scene_overlap.py` | Measures test/train scene overlap with RootSIFT voting and RANSAC verification (`scan`, `summarize`) |
| `evaluate_v2.py` | Corrected evaluator used for every number in the paper (LCMS-DB checkpoints, or Ultralytics models with `--yolo`) |
| `pitfalls.py` | Re-scores saved detections under each setting of the original, faulty evaluation script |
| `bootstrap_ci.py` | Bootstrap confidence intervals and paired differences between models |
| `train_baselines.py` | Trains the Ultralytics baselines (YOLOv8n/s, YOLOv8s-P2, YOLO11s, YOLOv8s from scratch) |
| `external_test.py` | Prepares the external test set and checks its overlap with the training split |
| `external_size_subset.py` | Recall on the external set for all objects and for objects of at least 32×32 px |
| `collect_*.py` | Build the result tables in `runs/*.md` |
| `run_*.bat` | Windows batch files that run each stage end to end |

---

## Setup

```bash
git clone https://github.com/assakr/bird-vs-drone-audit.git
cd bird-vs-drone-audit
pip install -r requirements.txt
```

The experiments were run with Python 3.12 and PyTorch 2 on an NVIDIA RTX A3000 Laptop GPU.

### Dataset paths

The scripts were written on Windows, and several defaults point to `D:/PROGRAMMING/dr shimaa/...`. Before running anything, set your own paths:

1. `data/data_clean.yaml`: the clean split used for training (written by `make_clean_split.py`).
2. The `set DS=`, `set CLEAN=` and `set EXT=` lines at the top of each `tools/run_*.bat` file.
3. For single scripts, pass the paths on the command line (`--dataset`, `--src`, `--images`, `--labels`, `--out`, `--kaggle`) instead of relying on the defaults.

---

## Data sources

Neither dataset is redistributed here, apart from the small overlay images in `sanity_checks*/`.

- **Main dataset:** YOLO-based Segmented Dataset for Drone vs. Bird Detection, Kaggle version (18,323 train, 1,740 validation, 889 test images, from Pexels stock videos). Shandilya, S.K. et al., *Data in Brief* 50 (2023) 109355, https://doi.org/10.1016/j.dib.2023.109355. See the dataset page for its licence.
- **External test set:** validation split of the Roboflow "birdrone dataset", version 10, by Fatin Zamri, licence CC BY 4.0: https://universe.roboflow.com/fatin-zamri/birdrone-dataset

---

## Reproducing the paper

All steps below are run from the repository root. Each step writes to `runs/`, and finished runs are skipped when a batch file is restarted.

| Paper result | How to reproduce | Output |
|---|---|---|
| Clean split | `python tools/make_clean_split.py --src <Kaggle Dataset folder>` | `<src>_clean/` |
| Architecture ablation, original assignment (A0–A7), final model, baselines | `tools\run_all_experiments.bat` | `runs/RESULTS_TABLES.md` |
| Label audit (Table 1) | `python tools/label_audit.py scan --dataset <Dataset>`, review `runs/label_audit/review/review.html`, then `python tools/label_audit.py summarize` | `runs/label_audit/` |
| Scene overlap (Table 2) | `python tools/scene_overlap.py scan --dataset <Dataset>`, review `runs/scene_overlap/review/review.html`, then `python tools/scene_overlap.py summarize` | `runs/scene_overlap/` |
| Evaluation pitfalls (Table 3) | `evaluate_v2.py ... --keep-all-dets`, then `python tools/pitfalls.py <name> <per_image.json>` | `pitfalls_<name>.json` |
| Repeated runs (Table 6) and bootstrap intervals | `tools\run_strengthening.bat` (or `tools\run_remaining_seeds.bat` to resume) | `runs/SEEDS_TABLE.md`, `runs/bootstrap_ci.json` |
| Ablation under the new assignment and its decomposition (Tables 7–8) | `tools\run_rev5_ablation.bat` | `runs/REV5_TABLES.md` |
| Comparison with standard detectors (Table 9) | `python tools/train_baselines.py --data data/data_clean.yaml`, then `evaluate_v2.py --yolo` | `runs/baselines/` |
| Cross-dataset test (Table 10) | `tools\run_external_test.bat` | `runs/EXTERNAL_TABLE.md` |
| External recall for objects of at least 32×32 px (Table 11) | `python tools/external_size_subset.py` (after the cross-dataset test) | `runs/external_ge32.json` |

Example: scoring a model with the corrected protocol (confidence floor 0.001, NMS IoU 0.6, matching at IoU 0.5, Ultralytics-style 101-point AP):

```bash
python tools/evaluate_v2.py --config configs/final_full_budget.yaml \
    --checkpoint runs/final_full_budget/best.pt \
    --images <Dataset_clean>/test/images --labels <Dataset_clean>/test/labels \
    --out runs/final_full_budget/eval_v2_test --device cuda
```

Add `--yolo` (and drop `--config`) to score an Ultralytics checkpoint with the same protocol.

`README_EXPERIMENTS.md` records the order in which the experiments were run and why each change was made.

---

## Citation

The paper is under review. Until it is published, please cite it as:

```bibtex
@unpublished{shaban2026birddrone,
  title  = {What Really Drives Bird-vs-Drone Detection Results? Label Errors, Scene Leakage,
            Evaluation Pitfalls, and a Controlled Study of Target Assignment},
  author = {Shaban, Shymaa S. and Sakr, Ahmed S.},
  note   = {Under review},
  year   = {2026}
}
```

## Licence

The code is released under the MIT Licence (see `LICENSE`). The datasets keep their own licences.

## Acknowledgements

We thank the authors of the YOLO-based Segmented Dataset for Drone vs. Bird Detection and of the Roboflow birdrone dataset, and the Ultralytics developers.
