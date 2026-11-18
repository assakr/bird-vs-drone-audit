"""
visualize.py
============
LCMS-DB Visualization Script.

Generates all figures required by SOP Section 11:
  1. Training curves (box/cls/obj loss, precision, recall, mAP@0.5, mAP@0.5:0.95)
  2. Confusion matrix heatmap (Bird ↔ Drone)
  3. Precision–Recall curves (per class)
  4. Qualitative detection overlays on sample test images
  5. Side-by-side comparison: Baseline vs. LCMS-DB on the same hard images

Usage:
    # Single model: curves + confusion + PR + qualitative samples
    python visualize.py \\
        --config  configs/ablation_4_full.yaml \\
        --checkpoint runs/ablation_4_full/best.pt \\
        --output  runs/ablation_4_full/figures/

    # Side-by-side comparison (needs both configs + checkpoints)
    python visualize.py \\
        --config       configs/ablation_4_full.yaml \\
        --checkpoint   runs/ablation_4_full/best.pt \\
        --base-config  configs/ablation_1_baseline.yaml \\
        --base-checkpoint runs/ablation_1_baseline/best.pt \\
        --output       runs/comparison/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for server / headless use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import Evaluator, decode_and_postprocess, compute_ap_from_records
from models.lcms_db import build_model, load_checkpoint
from train import load_config

# -- Colours -------------------------------------------------------------------
CLASS_COLORS = {0: (0, 200, 0), 1: (0, 0, 220)}     # BGR: Bird=green, Drone=blue
CLASS_NAMES  = ['Bird', 'Drone']

PALETTE = plt.cm.tab10.colors


# ══════════════════════════════════════════════════════════════════════════════
# 1. Training Curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(csv_path: str, out_dir: Path):
    """
    Parse the training CSV log and produce loss + metric curves.

    Args:
        csv_path: path to results.csv written by train.py
        out_dir:  directory to save the figure PNG
    """
    import csv

    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[vis] Warning: CSV log not found at {csv_path}. Skipping training curves.")
        return

    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if v.replace('.', '', 1).lstrip('-').isdigit() else v
                         for k, v in row.items()})

    if not rows:
        print("[vis] Warning: empty CSV log. Skipping training curves.")
        return

    epochs = [r['epoch'] for r in rows]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle('Training Curves — LCMS-DB', fontsize=16, fontweight='bold')

    metrics_map = {
        (0, 0): ('total',     'Total Loss',        'Train Loss'),
        (0, 1): ('box',       'Box Loss',           'Box Loss (CIoU)'),
        (0, 2): ('cls',       'Cls Loss',           'Classification Loss'),
        (0, 3): ('obj',       'Obj Loss',           'Objectness Loss'),
        (1, 0): ('val_loss',  'Validation Loss',    'Val Loss'),
        (1, 1): ('val_map50', 'mAP@0.5',           'Val mAP@0.5'),
    }

    for (ri, ci), (key, label, title) in metrics_map.items():
        ax = axes[ri][ci]
        vals = [r.get(key, None) for r in rows]
        vals_clean = [v for v in vals if v is not None]
        epochs_clean = [epochs[i] for i, v in enumerate(vals) if v is not None]
        if vals_clean:
            ax.plot(epochs_clean, vals_clean, linewidth=2, color=PALETTE[ci % len(PALETTE)])
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(label)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Learning rate plot (last subplot)
    ax = axes[1][2]
    lr_vals = [r.get('lr', None) for r in rows]
    lr_vals_clean = [v for v in lr_vals if v is not None]
    eps_clean = [epochs[i] for i, v in enumerate(lr_vals) if v is not None]
    if lr_vals_clean:
        ax.plot(eps_clean, lr_vals_clean, linewidth=2, color=PALETTE[4])
    ax.set_title('Learning Rate', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Hide last unused subplot
    axes[1][3].axis('off')

    plt.tight_layout()
    save_path = out_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[vis] Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    confusion: List[List[int]],
    class_names: List[str],
    out_dir: Path,
    title: str = 'Confusion Matrix',
    filename: str = 'confusion_matrix.png',
):
    """
    Plot and save a normalized confusion matrix heatmap.

    Args:
        confusion:    2D list (pred × gt) of raw counts
        class_names:  list of class name strings
        out_dir:      output directory
        title:        figure title
        filename:     output PNG filename
    """
    import seaborn as sns

    cm = np.array(confusion, dtype=float)
    # Normalize over true positives (columns = GT class)
    col_sums = cm.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    cm_norm = cm / col_sums

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=15, fontweight='bold')

    # Raw counts
    sns.heatmap(
        cm.astype(int), annot=True, fmt='d',
        xticklabels=class_names, yticklabels=class_names,
        cmap='Blues', ax=axes[0], linewidths=0.5,
    )
    axes[0].set_title('Raw Counts')
    axes[0].set_xlabel('Ground Truth')
    axes[0].set_ylabel('Predicted')

    # Normalized
    sns.heatmap(
        cm_norm, annot=True, fmt='.3f',
        xticklabels=class_names, yticklabels=class_names,
        cmap='Blues', ax=axes[1], vmin=0, vmax=1, linewidths=0.5,
    )
    axes[1].set_title('Normalized (per GT class)')
    axes[1].set_xlabel('Ground Truth')
    axes[1].set_ylabel('Predicted')

    plt.tight_layout()
    save_path = out_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[vis] Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Precision–Recall Curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_pr_curves(
    evaluator: Evaluator,
    out_dir: Path,
    filename: str = 'pr_curves.png',
):
    """
    Plot interpolated Precision–Recall curves for each class.

    Args:
        evaluator: Evaluator instance with accumulated records
        out_dir:   output directory
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.set_title('Precision–Recall Curves (IoU@0.5)', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

    for cls_id, rec in evaluator.global_records.items():
        cls_name = evaluator.CLASS_NAMES[cls_id] if cls_id < len(evaluator.CLASS_NAMES) else str(cls_id)
        n_gt = rec['n_gt']
        if n_gt == 0 or len(rec['scores']) == 0:
            continue

        order   = np.argsort(-np.array(rec['scores']))
        tp_sort = np.array(rec['tp'])[order]
        cum_tp  = np.cumsum(tp_sort)
        cum_fp  = np.cumsum(1 - tp_sort)
        prec    = cum_tp / (cum_tp + cum_fp + 1e-9)
        recall  = cum_tp / (n_gt + 1e-9)

        # Monotone envelope
        prec_env = prec.copy()
        for i in range(len(prec_env) - 2, -1, -1):
            prec_env[i] = max(prec_env[i], prec_env[i + 1])

        ap = compute_ap_from_records(rec['tp'], rec['scores'], n_gt)
        color = [c / 255 for c in CLASS_COLORS.get(cls_id, (128, 128, 128))][::-1]   # BGR->RGB

        ax.plot(recall, prec_env, linewidth=2, color=color, label=f'{cls_name} (AP={ap:.3f})')
        ax.fill_between(recall, prec_env, alpha=0.1, color=color)

    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_path = out_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[vis] Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Qualitative detection overlays
# ══════════════════════════════════════════════════════════════════════════════

def draw_boxes(
    image: np.ndarray,
    detections: Optional[torch.Tensor],
    targets: Optional[torch.Tensor],
    resolution: int,
    orig_shape: Tuple[int, int],
    conf_thresh: float = 0.25,
) -> np.ndarray:
    """
    Draw GT boxes (dashed) and predicted boxes (solid) on image.

    Args:
        image:       HWC BGR uint8 image at `resolution` size (letterboxed)
        detections:  (N, 6) [x1y1x2y2, score, cls] in resolution pixel space
        targets:     (M, 6) [batch_idx, cls, x,y,w,h] normalized for this image
        resolution:  inference resolution
        orig_shape:  original (H, W) before letterbox
        conf_thresh: minimum score to display a detection

    Returns:
        Annotated image copy (BGR uint8).
    """
    vis = image.copy()
    H, W = vis.shape[:2]

    def norm_xywh_to_px(xywh_norm):
        """Convert normalized xywh -> xyxy pixel coords on letterboxed image."""
        x_c, y_c, bw, bh = xywh_norm
        x1 = int((x_c - bw / 2) * W)
        y1 = int((y_c - bh / 2) * H)
        x2 = int((x_c + bw / 2) * W)
        y2 = int((y_c + bh / 2) * H)
        return x1, y1, x2, y2

    # Draw GT boxes (dashed yellow)
    if targets is not None and len(targets) > 0:
        for tgt in targets:
            cls_id = int(tgt[1].item())
            xywh   = tgt[2:].cpu().numpy()
            x1, y1, x2, y2 = norm_xywh_to_px(xywh)
            # Simulate dashed line by alternating segments
            color = (0, 220, 220)   # yellow-ish
            for i in range(x1, x2, 8):
                cv2.line(vis, (min(i+4, x2), y1), (min(i+4, x2)+4, y1), color, 2)
                cv2.line(vis, (min(i+4, x2), y2), (min(i+4, x2)+4, y2), color, 2)
            for j in range(y1, y2, 8):
                cv2.line(vis, (x1, min(j+4, y2)), (x1, min(j+4, y2)+4), color, 2)
                cv2.line(vis, (x2, min(j+4, y2)), (x2, min(j+4, y2)+4), color, 2)
            label = f"GT:{CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else cls_id}"
            cv2.putText(vis, label, (x1, max(y1 - 4, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Draw predicted boxes (solid)
    if detections is not None and len(detections) > 0:
        for det in detections:
            x1, y1, x2, y2 = [int(v.item()) for v in det[:4]]
            score   = det[4].item()
            cls_id  = int(det[5].item())
            if score < conf_thresh:
                continue
            color = CLASS_COLORS.get(cls_id, (200, 200, 200))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            label = f"{cls_name} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


def save_qualitative_samples(
    model: torch.nn.Module,
    loader,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
    n_samples: int = 12,
    model_tag: str = 'model',
):
    """
    Save individual qualitative detection images from the test set.

    Covers: correct small drones, correct birds, complex backgrounds,
    tiny targets, blurred/low-light cases.

    Args:
        model:     evaluated model
        loader:    test DataLoader
        cfg:       config dict
        device:    torch device
        out_dir:   output directory
        n_samples: maximum number of samples to save
        model_tag: prefix for saved filenames
    """
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    resolution = cfg['resolution']
    conf_thresh = cfg['eval']['conf_threshold']
    iou_thresh  = cfg['eval']['iou_threshold']
    nc = cfg['model']['num_classes']

    saved = 0
    pbar = tqdm(loader, desc=f'[vis] Qualitative ({model_tag})', leave=False)
    for batch in pbar:
        if saved >= n_samples:
            break

        images      = batch['images'].to(device)
        targets_raw = batch['labels'].to(device)
        img_paths   = batch['img_paths']
        orig_shapes = batch['orig_shapes']

        with torch.no_grad():
            raw_preds = model(images)

        strides = getattr(model, 'strides', None)
        detections = decode_and_postprocess(
            raw_preds, strides=strides, center_decode=getattr(model.head, 'center_decode', 'sigmoid'),
            resolution=resolution, conf_thresh=conf_thresh,
            iou_thresh=iou_thresh, num_classes=nc,
        )

        for b_idx in range(images.shape[0]):
            if saved >= n_samples:
                break

            # Convert tensor image back to uint8 BGR
            img_t = images[b_idx].cpu().numpy()   # (3, H, W) float
            # Un-normalize (approximate — for visualization only)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_np = (img_t.transpose(1, 2, 0) * std + mean) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_bgr = img_np[:, :, ::-1].copy()

            batch_tgts = targets_raw[targets_raw[:, 0] == b_idx]

            vis = draw_boxes(
                img_bgr, detections[b_idx], batch_tgts,
                resolution, orig_shapes[b_idx], conf_thresh,
            )

            stem     = Path(img_paths[b_idx]).stem
            out_path = out_dir / f"{model_tag}_{stem}.jpg"
            cv2.imwrite(str(out_path), vis)
            saved += 1

    print(f"[vis] Saved {saved} qualitative samples to {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Side-by-side comparison
# ══════════════════════════════════════════════════════════════════════════════

def save_comparison_grid(
    baseline_model: torch.nn.Module,
    lcms_model: torch.nn.Module,
    loader,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
    n_pairs: int = 6,
):
    """
    Save side-by-side comparison images: Baseline (left) vs LCMS-DB (right).

    Args:
        baseline_model: the ablation_1 baseline model
        lcms_model:     the full LCMS-DB model
        loader:         test DataLoader
        cfg:            config dict (shared)
        device:         torch device
        out_dir:        output directory
        n_pairs:        number of comparison pairs to save
    """
    baseline_model.eval()
    lcms_model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    resolution  = cfg['resolution']
    conf_thresh = cfg['eval']['conf_threshold']
    iou_thresh  = cfg['eval']['iou_threshold']
    nc = cfg['model']['num_classes']

    saved = 0
    pbar  = tqdm(loader, desc='[vis] Comparison grid', leave=False)
    for batch in pbar:
        if saved >= n_pairs:
            break

        images      = batch['images'].to(device)
        targets_raw = batch['labels'].to(device)
        img_paths   = batch['img_paths']
        orig_shapes = batch['orig_shapes']

        with torch.no_grad():
            raw_base = baseline_model(images)
            raw_lcms = lcms_model(images)

        dets_base = decode_and_postprocess(
            raw_base, strides=getattr(baseline_model, 'strides', None), center_decode=getattr(baseline_model.head, 'center_decode', 'sigmoid'), resolution=resolution,
            conf_thresh=conf_thresh, iou_thresh=iou_thresh, num_classes=nc,
        )
        dets_lcms = decode_and_postprocess(
            raw_lcms, strides=getattr(lcms_model, 'strides', None), center_decode=getattr(lcms_model.head, 'center_decode', 'sigmoid'), resolution=resolution,
            conf_thresh=conf_thresh, iou_thresh=iou_thresh, num_classes=nc,
        )

        for b_idx in range(images.shape[0]):
            if saved >= n_pairs:
                break

            img_t = images[b_idx].cpu().numpy()
            mean  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std   = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_np = (img_t.transpose(1, 2, 0) * std + mean) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_bgr = img_np[:, :, ::-1].copy()

            batch_tgts = targets_raw[targets_raw[:, 0] == b_idx]

            vis_base = draw_boxes(img_bgr.copy(), dets_base[b_idx], batch_tgts,
                                  resolution, orig_shapes[b_idx], conf_thresh)
            vis_lcms = draw_boxes(img_bgr.copy(), dets_lcms[b_idx], batch_tgts,
                                  resolution, orig_shapes[b_idx], conf_thresh)

            # Add labels
            h, w = vis_base.shape[:2]
            label_h = 30
            lbl_base = np.zeros((label_h, w, 3), dtype=np.uint8)
            lbl_lcms = np.zeros((label_h, w, 3), dtype=np.uint8)
            cv2.putText(lbl_base, 'Baseline YOLO', (5, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(lbl_lcms, 'LCMS-DB (Full)', (5, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

            vis_base = np.vstack([lbl_base, vis_base])
            vis_lcms = np.vstack([lbl_lcms, vis_lcms])

            # Separator
            sep = np.full((vis_base.shape[0], 4, 3), 200, dtype=np.uint8)
            grid = np.hstack([vis_base, sep, vis_lcms])

            stem     = Path(img_paths[b_idx]).stem
            out_path = out_dir / f"compare_{stem}.jpg"
            cv2.imwrite(str(out_path), grid)
            saved += 1

    print(f"[vis] Saved {saved} comparison grids to {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Ablation summary bar chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_ablation_summary(results_paths: Dict[str, str], out_dir: Path):
    """
    Plot mAP@0.5 and mAP@0.5:0.95 bar charts across ablation configs.

    Args:
        results_paths: dict mapping config label -> path to eval_results.json
        out_dir:       output directory
    """
    labels, map50_vals, map5095_vals = [], [], []
    for label, path in results_paths.items():
        p = Path(path)
        if not p.exists():
            print(f"[vis] Warning: {p} not found, skipping in ablation chart.")
            continue
        with open(p) as f:
            res = json.load(f)
        labels.append(label)
        map50_vals.append(res['overall'].get('mAP@0.5', 0.0))
        map5095_vals.append(res['overall'].get('mAP@0.5:0.95', 0.0))

    if not labels:
        print("[vis] No ablation results found. Skipping ablation chart.")
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 5))
    bars1 = ax.bar(x - width / 2, map50_vals,   width, label='mAP@0.5',      color=PALETTE[0], alpha=0.85)
    bars2 = ax.bar(x + width / 2, map5095_vals, width, label='mAP@0.5:0.95', color=PALETTE[1], alpha=0.85)

    ax.set_ylabel('mAP')
    ax.set_title('Ablation Study — mAP Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_path = out_dir / 'ablation_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[vis] Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Size breakdown bar chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_size_breakdown(eval_json_path: str, out_dir: Path):
    """
    Bar chart of Precision, Recall, AP@0.5 per object-size bucket.

    Args:
        eval_json_path: path to eval_results.json
        out_dir:        output directory
    """
    p = Path(eval_json_path)
    if not p.exists():
        print(f"[vis] Warning: {p} not found. Skipping size breakdown chart.")
        return

    with open(p) as f:
        res = json.load(f)

    size_data = res.get('size_breakdown', {})
    if not size_data:
        print("[vis] No size_breakdown data. Skipping.")
        return

    buckets = ['Tiny', 'Small', 'Medium', 'Large']
    pvals   = [size_data.get(b, {}).get('Precision', 0.0) for b in buckets]
    rvals   = [size_data.get(b, {}).get('Recall',    0.0) for b in buckets]
    apvals  = [size_data.get(b, {}).get('AP@0.5',    0.0) for b in buckets]

    x = np.arange(len(buckets))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w,  pvals,  w, label='Precision', color=PALETTE[0], alpha=0.85)
    ax.bar(x,      rvals,  w, label='Recall',    color=PALETTE[1], alpha=0.85)
    ax.bar(x + w,  apvals, w, label='AP@0.5',    color=PALETTE[2], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Object-Size Breakdown — Detection Performance', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_path = out_dir / 'size_breakdown.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[vis] Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Generate LCMS-DB visualizations')
    parser.add_argument('--config',            required=True, help='YAML config for LCMS-DB model')
    parser.add_argument('--checkpoint',        required=True, help='LCMS-DB checkpoint .pt')
    parser.add_argument('--output',            default=None,  help='Output directory for figures')
    parser.add_argument('--device',            default='',    help='cuda:0 / cpu (auto if empty)')
    parser.add_argument('--n-samples',         type=int, default=12,
                        help='Number of qualitative sample images to save')
    parser.add_argument('--csv-log',           default=None,
                        help='Path to results.csv from training (for curves)')
    parser.add_argument('--eval-json',         default=None,
                        help='Path to eval_results.json (for size breakdown)')
    # Side-by-side comparison (optional)
    parser.add_argument('--base-config',       default=None,
                        help='Config for baseline model (comparison mode)')
    parser.add_argument('--base-checkpoint',   default=None,
                        help='Checkpoint for baseline model (comparison mode)')
    # Ablation summary (optional)
    parser.add_argument('--ablation-jsons',    nargs='*', default=None,
                        help='label:path pairs e.g. "Baseline:path.json" for ablation chart')
    parser.add_argument('--conf-thresh',       type=float, default=None,
                        help='Confidence threshold override')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.conf_thresh is not None:
        cfg['eval']['conf_threshold'] = args.conf_thresh

    # -- Device --
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # -- Output dir --
    out_dir = Path(args.output) if args.output else Path(args.checkpoint).parent / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[vis] Output directory: {out_dir}")

    # -- 1. Training curves --
    csv_log = args.csv_log or str(Path(args.checkpoint).parent / 'results.csv')
    plot_training_curves(csv_log, out_dir)

    # -- Load LCMS-DB model --
    model = build_model(cfg).to(device)
    model, _ = load_checkpoint(model, args.checkpoint, str(device))
    model.eval()

    # -- DataLoader --
    from data.dataset import build_dataloaders
    _, _, test_loader = build_dataloaders(cfg)

    # -- 2–3. Run inference to populate evaluator (for confusion + PR) --
    evaluator = Evaluator(cfg, num_classes=cfg['model']['num_classes'])
    resolution  = cfg['resolution']
    conf_thresh = cfg['eval']['conf_threshold']
    iou_thresh  = cfg['eval']['iou_threshold']
    nc = cfg['model']['num_classes']

    print("[vis] Running inference for confusion matrix + PR curves...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='[vis] Inference', unit='batch'):
            images      = batch['images'].to(device)
            targets_raw = batch['labels'].to(device)
            img_paths   = batch['img_paths']
            orig_shapes = batch['orig_shapes']

            raw_preds  = model(images)
            strides    = getattr(model, 'strides', None)
            detections = decode_and_postprocess(
                raw_preds, strides=strides, center_decode=getattr(model.head, 'center_decode', 'sigmoid'),
                resolution=resolution, conf_thresh=conf_thresh,
                iou_thresh=iou_thresh, num_classes=nc,
            )
            evaluator.update(detections, targets_raw, img_paths, orig_shapes)

    summary = evaluator.summarize()

    # 2. Confusion matrix
    plot_confusion_matrix(
        summary['confusion_matrix'], evaluator.CLASS_NAMES, out_dir
    )

    # 3. PR curves
    plot_pr_curves(evaluator, out_dir)

    # -- 4. Qualitative samples --
    save_qualitative_samples(
        model, test_loader, cfg, device, out_dir / 'qualitative',
        n_samples=args.n_samples, model_tag='lcms_db'
    )

    # -- 5. Side-by-side comparison --
    if args.base_config and args.base_checkpoint:
        base_cfg   = load_config(args.base_config)
        base_model = build_model(base_cfg).to(device)
        base_model, _ = load_checkpoint(base_model, args.base_checkpoint, str(device))
        base_model.eval()
        _, _, base_test_loader = build_dataloaders(base_cfg)
        save_comparison_grid(
            base_model, model, base_test_loader, cfg, device,
            out_dir / 'comparison', n_pairs=6,
        )

    # -- Ablation summary chart --
    if args.ablation_jsons:
        ablation_map = {}
        for item in args.ablation_jsons:
            parts = item.split(':', 1)
            if len(parts) == 2:
                ablation_map[parts[0]] = parts[1]
        if ablation_map:
            plot_ablation_summary(ablation_map, out_dir)

    # -- Size breakdown --
    eval_json = args.eval_json
    if eval_json is None:
        candidate = Path(args.checkpoint).parent / 'eval_results' / 'eval_results.json'
        eval_json = str(candidate) if candidate.exists() else None
    if eval_json:
        plot_size_breakdown(eval_json, out_dir)

    print(f"\n[vis] All figures saved to: {out_dir}")


if __name__ == '__main__':
    main()
