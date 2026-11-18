# LCMS-DB: corrected evaluation and remaining experiments

Copy the `tools/`, `configs/` and `data/` folders of this kit into `code/lcms_db/` (nothing existing is overwritten).

## What was wrong in the original pipeline

| # | Problem | Effect on the paper | Fixed in |
|---|---|---|---|
| 1 | `valid/` bird labels are misaligned with the images (24 of 30 audited BV files; test split clean) | Bird AP ≈ 0 on validation; model selection driven by noisy labels | `tools/make_clean_split.py` |
| 2 | TP matching used the NMS threshold (IoU 0.45), not 0.50 | "mAP@0.5" was really mAP@0.45 | `tools/evaluate_v2.py` |
| 3 | mAP@0.5:0.95 ignored the IoU threshold | mAP@0.5:0.95 = mAP@0.5 always | `tools/evaluate_v2.py` |
| 4 | AP computed only from detections with conf ≥ 0.25 | AP truncated, under-estimated | `tools/evaluate_v2.py` (conf floor 0.001) |
| 5 | Size-bucket records stored only TPs | precision forced to 1.0/0.5, AP = recall | `tools/evaluate_v2.py` |
| 6 | Confusion matrix had no background row/column | ~2,500 false alarms hidden, "zero confusion" claim | `tools/evaluate_v2.py` |
| 7 | Tables from `evaluate.py` on `valid/`, figures from `visualize.py` on `test/` | Figures contradicted tables | single run → all tables & figures |
| 8 | FPS measured at batch 8, AMP, without NMS | 146.6 FPS not a per-frame latency | batch-1 timing incl. decode + NMS |

## Run order (Windows, from `code/lcms_db`)

```
pip install ultralytics
tools\run_all_experiments.bat
```

or step by step:

```
python tools\make_clean_split.py --src "D:/PROGRAMMING/dr shimaa/Bird vs Drone kaggel/Dataset"
python train.py --config configs\A0_plain.yaml --device cuda:0          (repeat for A1..A6)
python tools\evaluate_v2.py --config configs\A0_plain.yaml --checkpoint runs\A0_plain\best.pt ^
       --images ".../Dataset_clean/test/images" --labels ".../Dataset_clean/test/labels" ^
       --out runs\A0_plain\eval_v2_test --device cuda
python tools\train_baselines.py --data ".../Dataset_clean/data_clean.yaml"
python train.py --config configs\final_full_budget.yaml --device cuda:0
python tools\collect_results.py          ->  runs\RESULTS_TABLES.md
```

`collect_results.py` produces the numbers for Tables V and VI of the corrected paper (ablation and baseline comparison).

## Code change made in rev. 2: scale-aware target assignment

**Problem.** `losses/yolo_loss.py::_assign_targets` gave every ground-truth box exactly **one** positive cell on **every** scale (P2 to P5), whatever its size. A 400-px bird was a positive at one P2 cell, and a 10-px drone was a positive at one P5 cell. Almost every location was a negative, which pushes objectness (and so confidence) down. This is consistent with the low recall and the large-object failures seen at test time.

**Change** (opt-in; with the default settings the code behaves exactly as before):

| Setting | Where | Default (original) | Rev. 2 (A7, final) |
|---|---|---|---|
| `loss.assigner` | loss | `center` | `scale_aware` |
| `loss.level_ranges` | loss | — | P2 (0, 64], P3 (32, 128], P4 (96, 256], P5 > 192 px, on max(w, h) |
| `loss.center_radius` | loss | — | 2.5 strides |
| `loss.obj_iou_target` | loss | `false` | `true` (objectness target = IoU of the predicted box) |
| `model.center_decode` | head | `sigmoid` | `linear`: centre = (t + grid + 0.5)·stride |

With `scale_aware`, an object is a positive only on the level(s) whose size range covers it, at every cell of that level whose centre is inside the box and within 2.5 strides of the box centre (the centre cell is always positive). A cell claimed by two objects goes to the smaller one. The linear centre decode is needed so that those off-centre cells can reach the object centre. `train.py` validation, `evaluate.py`, `visualize.py` and `tools/evaluate_v2.py` all read the decode mode from the checkpoint's config, so old checkpoints evaluate exactly as before.

**What was verified on CPU:**
- with default settings the loss equals the original file's to the last digit;
- the old `v2_full/best.pt` re-evaluated through the modified code gives identical test results;
- a 150-iteration overfitting run (4 images, 320 px, same seed) trains with both assigners; the new one lowers its loss steadily (18.5 to 3.7) and starts detecting the training objects earlier. This only checks that it trains; it is not evidence of accuracy.

The real evidence has to come from GPU training. The comparison that matters is **A7 vs A6** on the clean test split (same architecture, data, budget and seed). If A7 is not better, train `final_full_budget` with the original assigner (instructions are in the config's header).

Originals of every changed file are in `_backup_before_assigner/`.

## Rev. 3: next steps after the first full set of results

First results (clean test split): the new assigner raised mAP@0.5 from 28.0 (A6) to 75.6 (A7), with nothing else changed. The COCO-pretrained YOLO baselines, however, reach about 97%. Run `tools\run_next_steps.bat`, which does the following:

1. **Finishes `final_full_budget`**, which stopped at epoch 16 of 100 while still improving. `train.py --resume` was fixed for this. Previously it reset the optimizer and restarted the cosine learning-rate schedule, so a resumed run would have ended at a learning rate about 100× too high. It also took the "best so far" score from the last epoch instead of `best.pt`. A resumed run now follows exactly the same learning-rate curve as an uninterrupted one (checked by a test).
2. **Trains YOLOv8s from scratch** (`--models yolov8s-scratch`: random initialisation, no COCO weights, 100 epochs). It is trained the same way as LCMS-DB, so the gap between LCMS-DB and YOLOv8s can be split into the effect of pretraining and the effect of architecture.
3. **Scores every YOLO baseline with `tools/evaluate_v2.py --yolo`**, the same matching, AP, size-bucket and confusion-matrix code used for LCMS-DB. As a check, it gives the trained YOLOv8n 96.7 mAP@0.5 / 65.2 mAP@0.5:0.95; Ultralytics' own scorer gives 97.0 / 64.1. The two evaluators agree, so the gap is real.
4. **Rebuilds `runs\RESULTS_TABLES.md`**, adding a baseline table on the identical protocol.

Timing in `evaluate_v2.py` now includes input normalisation for every model (letterbox resizing excluded), so LCMS-DB and the YOLO models are timed on the same basis.

`train_baselines.py` now merges its results into `summary.csv` instead of overwriting it.

## Rev. 4: making the evidence solid (multi-seed, full audit, leakage, confidence intervals)

**Correction.** The rev. 1–3 training runs were *not* seeded: `train.py` never set a random seed, although the configs said "same seed". `train.py` now accepts `--seed` (or `train.seed` in a config) and seeds Python, NumPy and PyTorch. The DataLoader workers inherit it, so the class-balanced sampler and every augmentation are reproducible. This was checked: the same seed gives identical batches, and different seeds give different ones. cuDNN is not forced into deterministic mode, so runs are seeded but not bit-identical.

Run `tools\run_strengthening.bat` (about 1–2 h of analyses, then about 25 h of training). Then review two web pages and run `tools\run_after_review.bat`.

| Tool | What it measures | Your part |
|---|---|---|
| `label_audit.py` | Checks every label in train, valid and test against the trained YOLOv8s. Each label is classed as ok, suspect (the detector confidently sees the object elsewhere) or unconfirmed. | Open `runs\label_audit\review\review.html`, give a verdict for each image and press **Save verdicts**. Put `verdicts.csv` in the same folder. |
| `scene_overlap.py` | For each test frame, finds the best-matching training mosaic using grayscale SIFT voting plus RANSAC geometric verification. This is robust to the mosaics' colour jitter, which defeats CNN embeddings. | Open `runs\scene_overlap\review\review.html`, mark each pair same_scene or different, then save. |
| `bootstrap_ci.py` | 95% intervals for mAP from 1,000 image-level resamples, and paired intervals with p-values for differences between models. | — |
| `collect_seeds.py` | Mean ± std over 3 runs: the original unseeded run plus seeds 1 and 2. | — |
| `evaluate_v2.py --image-list` | Scores models only on test images with no same-scene training image. | — |

Checks done before release:
- **Label audit** (run on the staged validation and test images with the trained YOLOv8n): it flags 28 of the 30 validation bird images (19 suspect, 9 unconfirmed) and **0 of 889 test images as suspect**. This matches the earlier visual audit.
- **Scene overlap:** it re-finds the known duplicate BT (10) ↔ BTR (297) with 193 geometric matches; unrelated images get 32 or fewer. Even with only 12 training images available, more than 100 test frames had a same-scene match with over 100 geometric matches. So expect substantial leakage in the full run.
- **Bootstrap:** with no resampling it reproduces the evaluator exactly (87.04 / 53.82 and 34.34 / 10.96). The 95% interval for the final model is 84.0–89.5 mAP@0.5.

**Caveat for the audit:** the YOLOv8s used as the second opinion was trained on about 90% of the training split, so it tends to agree with training labels. The training-split error rate it reports is therefore a lower bound. The validation bird images and the test split were never used to train it.
