"""
evaluate.py
===========
LCMS-DB Full Evaluation Script.

Computes all metrics required by SOP Sections 9 and 10:
  - Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95
  - FPS and inference time per image
  - Model complexity: parameters, FLOPs, model file size
  - Object-size breakdown: Tiny / Small / Medium / Large
  - Per-class (Bird / Drone) confusion matrix + class-wise P/R/F1/AP
  - Robustness subset analysis (reads subset tag from filename convention)

Usage:
    python evaluate.py \\
        --config  configs/ablation_4_full.yaml \\
        --checkpoint runs/ablation_4_full/best.pt \\
        --output  runs/ablation_4_full/eval_results/

Robustness subsets: tag image filenames with one of
    _bg_complex, _illumination, _motion_blur, _occlusion
to automatically group them into robustness buckets.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_dataloaders
from data.preprocess import undo_letterbox
from models.lcms_db import LCMSDB, build_model, load_checkpoint
from train import load_config


# -- NMS ------------------------------------------------------------------------

def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.45) -> torch.Tensor:
    """
    Non-Maximum Suppression (class-agnostic).
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    from torchvision.ops import nms as tv_nms
    return tv_nms(boxes.float(), scores.float(), iou_threshold)


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.45,
    sigma: float = 0.5,
    score_threshold: float = 0.05,
    method: str = "gaussian",
) -> torch.Tensor:
    """
    Soft-NMS implementation for overlapping small objects.
    Decays scores of overlapping boxes instead of abruptly removing them.
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    boxes = boxes.float()
    scores = scores.float().clone()
    N = boxes.shape[0]
    indices = torch.arange(N, device=boxes.device)

    from torchvision.ops import box_iou
    kept = []

    for _ in range(N):
        max_idx = torch.argmax(scores)
        if scores[max_idx] < score_threshold:
            break

        max_box = boxes[max_idx:max_idx+1]
        kept.append(indices[max_idx].item())

        ious = box_iou(max_box, boxes)[0]
        if method == "gaussian":
            decay = torch.exp(-(ious ** 2) / sigma)
        else:
            decay = torch.where(ious > iou_threshold, 1.0 - ious, 1.0)

        scores = scores * decay
        scores[max_idx] = 0.0

    if not kept:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    return torch.tensor(kept, dtype=torch.long, device=boxes.device)


def decode_and_postprocess(
    raw_preds: List[torch.Tensor],
    strides: List[int] | None = None,
    resolution: int = 640,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    num_classes: int = 2,
    use_soft_nms: bool = False,
    center_decode: str = "sigmoid",
) -> List[torch.Tensor]:
    """
    Decode raw YOLO predictions and apply NMS or Soft-NMS.

    Args:
        raw_preds: list of (B, Hi*Wi, 4+1+C) tensors per scale
        strides:   [8, 16, 32] or [4, 8, 16, 32]
        resolution: input image size (square)
        conf_thresh: minimum objectness * class score
        iou_thresh: NMS IoU threshold
        num_classes: number of object classes
        use_soft_nms: whether to apply soft-NMS

    Returns:
        List of (N_det, 6) tensors per image: [x1,y1,x2,y2,score,class_id]
    """
    if strides is None:
        strides = [4, 8, 16, 32] if len(raw_preds) == 4 else [8, 16, 32]
    device = raw_preds[0].device
    batch_size = raw_preds[0].shape[0]

    all_boxes   = []   # per image
    all_scores  = []
    all_classes = []

    for scale_idx, (pred, stride) in enumerate(zip(raw_preds, strides)):
        # pred: (B, H*W, 4+1+num_classes)
        B, HW, _ = pred.shape
        grid_size = int(HW ** 0.5)

        # Build grid
        gy, gx = torch.meshgrid(
            torch.arange(grid_size, device=device, dtype=torch.float32),
            torch.arange(grid_size, device=device, dtype=torch.float32),
            indexing='ij',
        )
        grid = torch.stack([gx, gy], dim=-1).view(1, -1, 2)   # (1, H*W, 2)

        # Decode box: xy via sigmoid + grid, wh via exp
        if center_decode == "linear":
            xy = (pred[..., :2] + grid + 0.5) * stride   # pixel coords (multi-positive models)
        else:
            xy = (pred[..., :2].sigmoid() + grid) * stride   # pixel coords
        wh = pred[..., 2:4].exp() * stride
        x1 = xy[..., 0] - wh[..., 0] / 2
        y1 = xy[..., 1] - wh[..., 1] / 2
        x2 = xy[..., 0] + wh[..., 0] / 2
        y2 = xy[..., 1] + wh[..., 1] / 2
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)   # (B, HW, 4)

        obj   = pred[..., 4].sigmoid()
        cls   = pred[..., 5:].sigmoid()

        # Combined confidence: obj * cls_score
        scores, cls_ids = (obj.unsqueeze(-1) * cls).max(dim=-1)

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_classes.append(cls_ids)

    # Concatenate across scales: (B, total_anchors, *)
    all_boxes   = torch.cat(all_boxes,   dim=1)
    all_scores  = torch.cat(all_scores,  dim=1)
    all_classes = torch.cat(all_classes, dim=1)

    results = []
    for b in range(batch_size):
        mask = all_scores[b] >= conf_thresh
        b_boxes   = all_boxes[b][mask]
        b_scores  = all_scores[b][mask]
        b_classes = all_classes[b][mask].float()

        if b_boxes.shape[0] == 0:
            results.append(torch.empty((0, 6), device=device))
            continue

        # Clip to image bounds
        b_boxes[:, [0, 2]] = b_boxes[:, [0, 2]].clamp(0, resolution)
        b_boxes[:, [1, 3]] = b_boxes[:, [1, 3]].clamp(0, resolution)

        if use_soft_nms:
            keep = soft_nms(b_boxes, b_scores, iou_thresh)
        else:
            keep = nms(b_boxes, b_scores, iou_thresh)
        det  = torch.cat([
            b_boxes[keep],
            b_scores[keep].unsqueeze(1),
            b_classes[keep].unsqueeze(1),
        ], dim=1)
        results.append(det)

    return results


# -- IoU ------------------------------------------------------------------------

def box_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Pairwise IoU between two sets of boxes.

    Args:
        box1: (N, 4) xyxy
        box2: (M, 4) xyxy

    Returns:
        (N, M) IoU matrix
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter   = inter_w * inter_h

    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-9)


# -- Size bucket helpers --------------------------------------------------------

SIZE_THRESHOLDS = {
    'Tiny':   (0,        32 * 32),
    'Small':  (32 * 32,  96 * 96),
    'Medium': (96 * 96,  256 * 256),
    'Large':  (256 * 256, float('inf')),
}

ROBUSTNESS_TAGS = ['bg_complex', 'illumination', 'motion_blur', 'occlusion']


def get_size_bucket(w_px: float, h_px: float) -> str:
    """Return size bucket name for a box given pixel width and height."""
    area = w_px * h_px
    for name, (lo, hi) in SIZE_THRESHOLDS.items():
        if lo <= area < hi:
            return name
    return 'Large'


def get_robustness_tag(img_path: str) -> Optional[str]:
    """Extract robustness tag from filename if present."""
    stem = Path(img_path).stem
    for tag in ROBUSTNESS_TAGS:
        if f'_{tag}' in stem:
            return tag
    return None


# -- AP computation -------------------------------------------------------------

def compute_ap_from_records(
    tp_list: List[int],
    score_list: List[float],
    n_gt: int,
    iou_thresh: float = 0.5,
) -> float:
    """
    Compute AP@iou_thresh via interpolated precision-recall curve.

    Args:
        tp_list:    binary TP indicator per detection (already sorted by score)
        score_list: confidence scores (used for sorting)
        n_gt:       total number of ground-truth objects for this class

    Returns:
        AP as a float in [0, 1]
    """
    if n_gt == 0 or len(score_list) == 0:
        return 0.0

    order = np.argsort(-np.array(score_list))
    tp_sorted = np.array(tp_list)[order]

    cum_tp  = np.cumsum(tp_sorted)
    cum_fp  = np.cumsum(1 - tp_sorted)
    rec     = cum_tp / (n_gt + 1e-9)
    prec    = cum_tp / (cum_tp + cum_fp + 1e-9)

    # Append sentinel values
    rec  = np.concatenate([[0.0], rec,  [1.0]])
    prec = np.concatenate([[1.0], prec, [0.0]])

    # Monotone envelope
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])

    # Area under curve
    idx = np.where(rec[1:] != rec[:-1])[0]
    ap  = float(np.sum((rec[idx + 1] - rec[idx]) * prec[idx + 1]))
    return ap


def compute_map_range(
    det_records: Dict,        # {class_id: {'tp': [], 'scores': [], 'n_gt': int}}
    iou_start: float = 0.5,
    iou_end:   float = 0.95,
    iou_step:  float = 0.05,
) -> float:
    """Compute mAP@0.5:0.95 (COCO-style) averaged over IoU thresholds."""
    thresholds = np.arange(iou_start, iou_end + 1e-8, iou_step)
    aps_all = []
    for iou_t in thresholds:
        aps = []
        for cls_id, rec in det_records.items():
            ap = compute_ap_from_records(rec['tp'], rec['scores'], rec['n_gt'], iou_t)
            aps.append(ap)
        if aps:
            aps_all.append(float(np.mean(aps)))
    return float(np.mean(aps_all)) if aps_all else 0.0


# -- Main evaluation ------------------------------------------------------------

class Evaluator:
    """
    Full evaluation engine.

    Accumulates detections across the test set, then computes all
    SOP Section 9 and 10 metrics in one pass.
    """

    CLASS_NAMES = ['Bird', 'Drone']

    def __init__(self, cfg: dict, num_classes: int = 2, conf_thresh: float = None, iou_thresh: float = None):
        self.cfg         = cfg
        self.nc          = num_classes
        self.conf_thresh = conf_thresh if conf_thresh is not None else cfg['eval']['conf_threshold']
        self.iou_thresh  = iou_thresh if iou_thresh is not None else cfg['eval']['iou_threshold']
        self.resolution  = cfg['resolution']

        # Global records keyed by class_id
        self.global_records: Dict[int, Dict] = {
            c: {'tp': [], 'scores': [], 'n_gt': 0} for c in range(num_classes)
        }

        # Per-size-bucket records
        self.size_records: Dict[str, Dict[int, Dict]] = {
            bkt: {c: {'tp': [], 'scores': [], 'n_gt': 0} for c in range(num_classes)}
            for bkt in SIZE_THRESHOLDS
        }

        # Per-robustness-tag records
        self.robust_records: Dict[str, Dict[int, Dict]] = {
            tag: {c: {'tp': [], 'scores': [], 'n_gt': 0} for c in range(num_classes)}
            for tag in ROBUSTNESS_TAGS
        }

        # Confusion matrix (pred_class, gt_class)
        self.confusion = np.zeros((num_classes, num_classes), dtype=int)

    def update(
        self,
        detections: List[torch.Tensor],   # per image: (N,6) [x1y1x2y2, score, cls]
        targets: torch.Tensor,             # (M, 6) [batch_idx, cls, x,y,w,h] normalized
        img_paths: List[str],
        orig_shapes: List[Tuple[int, int]],
    ):
        """
        Process one batch of detections and ground-truth targets.

        Matches predictions to GT using IoU, fills records for global,
        size, and robustness breakdowns, and updates the confusion matrix.
        """
        for b_idx, (dets, img_path, orig_shape) in enumerate(
            zip(detections, img_paths, orig_shapes)
        ):
            batch_targets = targets[targets[:, 0] == b_idx]

            # Convert GT xywh normalized -> xyxy pixel (at resolution)
            gt_cls   = batch_targets[:, 1].long().cpu()
            gt_xywh  = batch_targets[:, 2:].cpu()
            gt_xyxy  = self._xywh_norm_to_xyxy_px(gt_xywh, self.resolution)
            gt_wh_px = gt_xywh[:, 2:] * self.resolution   # pixel wh for size bucket

            rob_tag = get_robustness_tag(img_path)

            # Count GT per class
            for gi, gc in enumerate(gt_cls.tolist()):
                bkt = get_size_bucket(gt_wh_px[gi, 0].item(), gt_wh_px[gi, 1].item())
                self.global_records[gc]['n_gt'] += 1
                self.size_records[bkt][gc]['n_gt'] += 1
                if rob_tag:
                    self.robust_records[rob_tag][gc]['n_gt'] += 1

            if dets.shape[0] == 0:
                continue

            det_boxes  = dets[:, :4].cpu()
            det_scores = dets[:, 4].cpu()
            det_cls    = dets[:, 5].long().cpu()

            if gt_xyxy.shape[0] == 0:
                # All detections are FP
                for score, cls_id in zip(det_scores.tolist(), det_cls.tolist()):
                    self.global_records[cls_id]['scores'].append(score)
                    self.global_records[cls_id]['tp'].append(0)
                continue

            # IoU matrix: (N_det, N_gt)
            iou_mat = box_iou_xyxy(det_boxes, gt_xyxy)

            matched_gt = set()
            for di in range(dets.shape[0]):
                score   = det_scores[di].item()
                pred_cls = det_cls[di].item()

                self.global_records[pred_cls]['scores'].append(score)

                # Find best matching GT (same class, IoU ≥ 0.5)
                best_iou  = 0.0
                best_gi   = -1
                for gi in range(gt_xyxy.shape[0]):
                    if gt_cls[gi].item() != pred_cls:
                        continue
                    iou_val = iou_mat[di, gi].item()
                    if iou_val > best_iou and gi not in matched_gt:
                        best_iou = iou_val
                        best_gi  = gi

                is_tp = int(best_iou >= self.iou_thresh and best_gi >= 0)
                self.global_records[pred_cls]['tp'].append(is_tp)

                if is_tp:
                    matched_gt.add(best_gi)
                    gt_bkt = get_size_bucket(
                        gt_wh_px[best_gi, 0].item(), gt_wh_px[best_gi, 1].item()
                    )
                    self.size_records[gt_bkt][pred_cls]['tp'].append(1)
                    self.size_records[gt_bkt][pred_cls]['scores'].append(score)
                    if rob_tag:
                        self.robust_records[rob_tag][pred_cls]['tp'].append(1)
                        self.robust_records[rob_tag][pred_cls]['scores'].append(score)

                    # Confusion matrix: true positive
                    gt_cls_matched = gt_cls[best_gi].item()
                    self.confusion[pred_cls, gt_cls_matched] += 1
                else:
                    # Check if this prediction overlapped with a different class GT (cross-class confusion)
                    best_cross_iou = 0.0
                    best_cross_gi = -1
                    for gi in range(gt_xyxy.shape[0]):
                        if gt_cls[gi].item() != pred_cls:
                            iou_val = iou_mat[di, gi].item()
                            if iou_val > best_cross_iou and gi not in matched_gt:
                                best_cross_iou = iou_val
                                best_cross_gi = gi
                    if best_cross_iou >= self.iou_thresh and best_cross_gi >= 0:
                        gt_cross_cls = gt_cls[best_cross_gi].item()
                        self.confusion[pred_cls, gt_cross_cls] += 1
                        matched_gt.add(best_cross_gi)

    @staticmethod
    def _xywh_norm_to_xyxy_px(xywh: torch.Tensor, resolution: int) -> torch.Tensor:
        """Convert normalized xywh to pixel-space xyxy."""
        xywh_px = xywh * resolution
        x1 = xywh_px[:, 0] - xywh_px[:, 2] / 2
        y1 = xywh_px[:, 1] - xywh_px[:, 3] / 2
        x2 = xywh_px[:, 0] + xywh_px[:, 2] / 2
        y2 = xywh_px[:, 1] + xywh_px[:, 3] / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _compute_prec_rec_f1(
        self, records: Dict[int, Dict], iou_thresh: float = 0.5
    ) -> Dict[int, Dict]:
        """Compute Precision, Recall, F1, AP@iou_thresh per class."""
        results = {}
        for cls_id, rec in records.items():
            tp    = np.array(rec['tp'])
            scores = np.array(rec['scores'])
            n_gt  = rec['n_gt']

            if len(tp) == 0:
                results[cls_id] = {'P': 0.0, 'R': 0.0, 'F1': 0.0, 'AP': 0.0, 'n_gt': n_gt}
                continue

            order   = np.argsort(-scores)
            tp_sort = tp[order]

            cum_tp  = np.cumsum(tp_sort)
            cum_fp  = np.cumsum(1 - tp_sort)
            prec    = cum_tp[-1] / (cum_tp[-1] + cum_fp[-1] + 1e-9)
            rec_val = cum_tp[-1] / (n_gt + 1e-9)
            f1      = 2 * prec * rec_val / (prec + rec_val + 1e-9)
            ap      = compute_ap_from_records(tp.tolist(), scores.tolist(), n_gt, iou_thresh)

            results[cls_id] = {
                'P': float(prec), 'R': float(rec_val), 'F1': float(f1),
                'AP': float(ap), 'n_gt': int(n_gt)
            }
        return results

    def summarize(self) -> Dict:
        """Compute and return all metric summaries."""
        # -- Global metrics --
        class_metrics = self._compute_prec_rec_f1(self.global_records)
        all_aps = [m['AP'] for m in class_metrics.values()]
        map50   = float(np.mean(all_aps)) if all_aps else 0.0
        map5095 = compute_map_range(
            self.global_records,
            self.cfg['eval']['map_iou_start'],
            self.cfg['eval']['map_iou_end'],
            self.cfg['eval']['map_iou_step'],
        )
        all_p  = [m['P']  for m in class_metrics.values()]
        all_r  = [m['R']  for m in class_metrics.values()]
        all_f1 = [m['F1'] for m in class_metrics.values()]

        overall = {
            'mAP@0.5':     map50,
            'mAP@0.5:0.95': map5095,
            'Precision':   float(np.mean(all_p)),
            'Recall':      float(np.mean(all_r)),
            'F1':          float(np.mean(all_f1)),
        }

        # -- Per-class breakdown --
        per_class = {
            self.CLASS_NAMES[c]: m for c, m in class_metrics.items()
            if c < len(self.CLASS_NAMES)
        }

        # -- Size breakdown --
        size_breakdown = {}
        for bkt, records in self.size_records.items():
            bkt_metrics = self._compute_prec_rec_f1(records)
            bkt_aps = [m['AP'] for m in bkt_metrics.values()]
            bkt_p   = [m['P']  for m in bkt_metrics.values()]
            bkt_r   = [m['R']  for m in bkt_metrics.values()]
            size_breakdown[bkt] = {
                'Precision': float(np.mean(bkt_p)) if bkt_p else 0.0,
                'Recall':    float(np.mean(bkt_r)) if bkt_r else 0.0,
                'AP@0.5':    float(np.mean(bkt_aps)) if bkt_aps else 0.0,
            }

        # -- Robustness breakdown --
        robust_breakdown = {}
        for tag, records in self.robust_records.items():
            rob_metrics = self._compute_prec_rec_f1(records)
            rob_aps = [m['AP'] for m in rob_metrics.values()]
            rob_p   = [m['P']  for m in rob_metrics.values()]
            rob_r   = [m['R']  for m in rob_metrics.values()]
            robust_breakdown[tag] = {
                'Precision': float(np.mean(rob_p)) if rob_p else 0.0,
                'Recall':    float(np.mean(rob_r)) if rob_r else 0.0,
                'mAP@0.5':   float(np.mean(rob_aps)) if rob_aps else 0.0,
            }

        return {
            'overall':      overall,
            'per_class':    per_class,
            'size_breakdown': size_breakdown,
            'robustness':   robust_breakdown,
            'confusion_matrix': self.confusion.tolist(),
        }


# -- Model complexity -----------------------------------------------------------

def measure_complexity(model: LCMSDB, resolution: int, device: torch.device) -> Dict:
    """
    Measure model complexity: params, FLOPs, file size, inference speed.

    Args:
        model:      LCMS-DB model
        resolution: input image size (square)
        device:     torch device

    Returns:
        dict with 'params', 'flops', 'fps', 'inference_ms', 'model_size_mb'
    """
    try:
        from thop import profile as thop_profile
        dummy = torch.randn(1, 3, resolution, resolution).to(device)
        flops, params = thop_profile(model, inputs=(dummy,), verbose=False)
        flops_g = flops / 1e9
        params_m = params / 1e6
    except ImportError:
        flops_g  = -1.0
        params_m = sum(p.numel() for p in model.parameters()) / 1e6

    # Inference speed (warm up then measure 100 iterations)
    model.eval()
    dummy = torch.randn(1, 3, resolution, resolution).to(device)
    with torch.no_grad():
        for _ in range(10):   # warm-up
            _ = model(dummy)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    n_iters = 100
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    fps = n_iters / elapsed
    inf_ms = elapsed * 1000 / n_iters

    return {
        'params_M':       round(params_m, 3),
        'flops_G':        round(flops_g,  3),
        'fps':            round(fps, 2),
        'inference_ms':   round(inf_ms, 3),
        # Model file size is filled in after checkpoint is found
        'model_size_mb':  None,
    }


def forward_tta(model: torch.nn.Module, images: torch.Tensor) -> List[torch.Tensor]:
    """Test-Time Augmentation: average original and horizontally flipped predictions."""
    preds_orig = model(images)
    images_flip = torch.flip(images, dims=[-1])
    preds_flip = model(images_flip)
    preds_merged = []
    for po, pf in zip(preds_orig, preds_flip):
        pm = (po + pf) * 0.5
        preds_merged.append(pm)
    return preds_merged


# -- Main -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate LCMS-DB')
    parser.add_argument('--config',     required=True,  help='Path to YAML config')
    parser.add_argument('--checkpoint', required=True,  help='Path to model checkpoint (.pt)')
    parser.add_argument('--output',     default=None,   help='Output directory for results')
    parser.add_argument('--device',     default='',     help='cuda:0 / cpu (auto if empty)')
    parser.add_argument('--split',      default='test', choices=['test', 'val'],
                        help='Dataset split to evaluate')
    parser.add_argument('--soft-nms',   action='store_true', help='Use Soft-NMS instead of standard NMS')
    parser.add_argument('--tta',        action='store_true', help='Enable Test-Time Augmentation (TTA)')
    parser.add_argument('--conf-thresh', type=float, default=None, help='Confidence threshold override')
    parser.add_argument('--iou-thresh',  type=float, default=None, help='IoU threshold override')
    args = parser.parse_args()

    cfg = load_config(args.config)

    conf_thresh = args.conf_thresh if args.conf_thresh is not None else cfg['eval']['conf_threshold']
    iou_thresh  = args.iou_thresh if args.iou_thresh is not None else cfg['eval']['iou_threshold']

    # -- Device --
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"[eval] Device: {device} | Soft-NMS: {args.soft_nms} | TTA: {args.tta} | Conf: {conf_thresh} | IoU: {iou_thresh}")

    # -- Output directory --
    out_dir = Path(args.output) if args.output else Path(args.checkpoint).parent / 'eval_results'
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Model --
    model = build_model(cfg).to(device)
    model, ckpt_meta = load_checkpoint(model, args.checkpoint, str(device))
    model.eval()
    print(f"[eval] Loaded checkpoint from epoch {ckpt_meta.get('epoch', '?')}")

    # -- Data --
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    loader = test_loader if args.split == 'test' else val_loader
    print(f"[eval] Evaluating on '{args.split}' split ({len(loader)} batches)")

    nc = cfg['model']['num_classes']
    resolution = cfg['resolution']
    evaluator = Evaluator(cfg, num_classes=nc, conf_thresh=conf_thresh, iou_thresh=iou_thresh)

    # -- Inference loop --
    use_amp = device.type == 'cuda'
    t_total = 0.0
    n_images = 0

    pbar = tqdm(loader, desc='[eval] Inference', unit='batch')
    for batch in pbar:
        images      = batch['images'].to(device, non_blocking=True)
        targets     = batch['labels'].to(device, non_blocking=True)
        img_paths   = batch['img_paths']
        orig_shapes = batch['orig_shapes']

        t0 = time.perf_counter()
        with torch.no_grad():
            from torch.cuda.amp import autocast
            with autocast(enabled=use_amp):
                if args.tta:
                    raw_preds = forward_tta(model, images)
                else:
                    raw_preds = model(images)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_total += time.perf_counter() - t0
        n_images += images.shape[0]

        # Decode + NMS / Soft-NMS
        detections = decode_and_postprocess(
            raw_preds,
            strides=model.strides,
            resolution=resolution,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            num_classes=nc,
            use_soft_nms=args.soft_nms,
            center_decode=getattr(model.head, "center_decode", "sigmoid"),
        )

        evaluator.update(detections, targets, img_paths, orig_shapes)

    # -- Compute metrics --
    fps     = n_images / t_total
    inf_ms  = t_total * 1000 / n_images
    summary = evaluator.summarize()
    summary['speed'] = {'fps': round(fps, 2), 'inference_ms': round(inf_ms, 3), 'n_images': n_images}

    # -- Model complexity --
    complexity = measure_complexity(model, resolution, device)
    ckpt_size  = Path(args.checkpoint).stat().st_size / (1024 ** 2)
    complexity['model_size_mb'] = round(ckpt_size, 2)
    summary['complexity'] = complexity

    # -- Print & save --
    print("\n" + "=" * 60)
    print("  LCMS-DB Evaluation Results")
    print("=" * 60)
    print(f"\n{'Overall Metrics':}")
    for k, v in summary['overall'].items():
        print(f"  {k:<20}: {v:.4f}")

    print(f"\n{'Per-Class Metrics':}")
    for cls_name, m in summary['per_class'].items():
        print(f"  {cls_name}: P={m['P']:.4f}  R={m['R']:.4f}  F1={m['F1']:.4f}  AP@0.5={m['AP']:.4f}  (GT={m['n_gt']})")

    print(f"\n{'Object-Size Breakdown':}")
    for bkt, m in summary['size_breakdown'].items():
        print(f"  {bkt:<8}: P={m['Precision']:.4f}  R={m['Recall']:.4f}  AP@0.5={m['AP@0.5']:.4f}")

    print(f"\n{'Confusion Matrix (rows=pred, cols=gt)':}")
    print("           Bird   Drone")
    for ri, row in enumerate(summary['confusion_matrix']):
        rname = evaluator.CLASS_NAMES[ri] if ri < len(evaluator.CLASS_NAMES) else str(ri)
        print(f"  {rname:<8}: {row[0]:5d}  {row[1]:5d}")

    print(f"\n{'Robustness Breakdown':}")
    for tag, m in summary['robustness'].items():
        print(f"  {tag:<20}: P={m['Precision']:.4f}  R={m['Recall']:.4f}  mAP@0.5={m['mAP@0.5']:.4f}")

    print(f"\n{'Speed':}")
    print(f"  FPS:           {summary['speed']['fps']}")
    print(f"  Inference (ms):{summary['speed']['inference_ms']}")

    print(f"\n{'Model Complexity':}")
    for k, v in complexity.items():
        print(f"  {k:<20}: {v}")

    # Save JSON
    results_path = out_dir / 'eval_results.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[eval] Full results saved to: {results_path}")


if __name__ == '__main__':
    main()
