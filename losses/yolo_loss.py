"""
losses/yolo_loss.py
YOLO detection loss for LCMS-DB.

Three components:
  * Box regression  : Complete-IoU (CIoU) loss on matched positive anchors.
  * Objectness      : Binary cross-entropy (BCE) between predicted objectness
                      scores and {1 for matched cell, 0 for background}.
  * Classification  : Focal-BCE loss for the predicted class distributions
                      on matched positive anchors only.

Target format (per row of the targets tensor):
    [batch_idx, class_id, x_center_norm, y_center_norm, w_norm, h_norm]
    Coordinates are normalised to [0, 1] relative to the image side.

Anchor-free cell matching strategy:
    For each ground-truth box, the grid cell whose top-left corner is
    closest to the box centre is assigned as the positive. No IoU
    thresholding is needed — only centre-point assignment.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# IoU utilities
# ---------------------------------------------------------------------------


def bbox_iou(
    box1: Tensor,
    box2: Tensor,
    iou_type: str = "ciou",
) -> Tensor:
    """Compute IoU (and its variants) between two sets of boxes in xywh format.

    Boxes are centre-based: (cx, cy, w, h).  They may be in absolute pixel
    coordinates or normalised — both formats are handled identically.

    Args:
        box1:     Predicted boxes, shape (N, 4) in cx-cy-w-h format.
        box2:     Target boxes,    shape (N, 4) in cx-cy-w-h format.
        iou_type: One of ``'iou'``, ``'giou'``, ``'diou'``, ``'ciou'``.

    Returns:
        Per-pair IoU scores, shape (N,).  Values are in (-1, 1] for GIoU,
        and approximately in [0, 1] for the others.

    Raises:
        ValueError: If an unknown ``iou_type`` is specified.
    """
    if iou_type not in {"iou", "giou", "diou", "ciou", "siou", "wiou"}:
        raise ValueError(f"Unknown iou_type '{iou_type}'. "
                         f"Choose from: 'iou', 'giou', 'diou', 'ciou', 'siou', 'wiou'.")

    # Convert (cx, cy, w, h) -> (x1, y1, x2, y2)
    b1_x1 = box1[:, 0] - box1[:, 2] / 2
    b1_y1 = box1[:, 1] - box1[:, 3] / 2
    b1_x2 = box1[:, 0] + box1[:, 2] / 2
    b1_y2 = box1[:, 1] + box1[:, 3] / 2

    b2_x1 = box2[:, 0] - box2[:, 2] / 2
    b2_y1 = box2[:, 1] - box2[:, 3] / 2
    b2_x2 = box2[:, 0] + box2[:, 2] / 2
    b2_y2 = box2[:, 1] + box2[:, 3] / 2

    # Intersection
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w  = (inter_x2 - inter_x1).clamp(min=0)
    inter_h  = (inter_y2 - inter_y1).clamp(min=0)
    inter    = inter_w * inter_h

    # Areas
    area1 = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
    area2 = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)

    union = area1 + area2 - inter + 1e-9
    iou   = inter / union  # (N,)

    if iou_type == "iou":
        return iou

    # Enclosing box for GIoU / DIoU / CIoU / SIoU / WIoU
    enc_x1 = torch.min(b1_x1, b2_x1)
    enc_y1 = torch.min(b1_y1, b2_y1)
    enc_x2 = torch.max(b1_x2, b2_x2)
    enc_y2 = torch.max(b1_y2, b2_y2)
    enc_w  = (enc_x2 - enc_x1).clamp(min=1e-9)
    enc_h  = (enc_y2 - enc_y1).clamp(min=1e-9)
    enc_area = enc_w * enc_h + 1e-9

    if iou_type == "giou":
        return iou - (enc_area - union) / enc_area

    # Squared distance between centres
    cx1 = (b1_x1 + b1_x2) / 2
    cy1 = (b1_y1 + b1_y2) / 2
    cx2 = (b2_x1 + b2_x2) / 2
    cy2 = (b2_y1 + b2_y2) / 2
    d2  = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Squared diagonal of enclosing box
    c2  = enc_w ** 2 + enc_h ** 2 + 1e-9

    if iou_type == "diou":
        return iou - d2 / c2

    if iou_type == "wiou":
        # Wise-IoU v1: distance-based dynamic attention
        dist_penalty = torch.exp((d2 / c2).clamp(max=4.0))
        return 1.0 - dist_penalty * (1.0 - iou)

    if iou_type == "siou":
        # SIoU: Angle cost + Distance cost + Shape cost
        s_cw = (cx2 - cx1).abs() + 1e-9
        s_ch = (cy2 - cy1).abs() + 1e-9
        sigma = torch.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) + 1e-9
        sin_alpha_1 = s_ch / sigma
        sin_alpha_2 = s_cw / sigma
        sin_alpha = torch.where(sin_alpha_1 > 0.707106, sin_alpha_2, sin_alpha_1)
        angle_cost = torch.cos(torch.asin(sin_alpha.clamp(-0.999, 0.999)) * 2 - torch.pi / 2)

        gamma = 2 - angle_cost
        rho_x = ((cx2 - cx1) / enc_w) ** 2
        rho_y = ((cy2 - cy1) / enc_h) ** 2
        dist_cost = 2 - torch.exp(-gamma * rho_x) - torch.exp(-gamma * rho_y)

        w_dis = ((box1[:, 2] - box2[:, 2]).abs() / torch.max(box1[:, 2], box2[:, 2]).clamp(min=1e-9))
        h_dis = ((box1[:, 3] - box2[:, 3]).abs() / torch.max(box1[:, 3], box2[:, 3]).clamp(min=1e-9))
        omega_w = (1 - torch.exp(-w_dis)) ** 4
        omega_h = (1 - torch.exp(-h_dis)) ** 4
        shape_cost = omega_w + omega_h

        return iou - (dist_cost + shape_cost) / 2.0

    # CIoU: adds aspect-ratio consistency term v
    v = (4 / (torch.pi ** 2)) * (
        torch.atan(box2[:, 2] / (box2[:, 3] + 1e-9))
        - torch.atan(box1[:, 2] / (box1[:, 3] + 1e-9))
    ) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-9)

    return iou - d2 / c2 - alpha * v   # CIoU


# ---------------------------------------------------------------------------
# Focal BCE helper
# ---------------------------------------------------------------------------


def focal_bce_loss(
    pred: Tensor,
    target: Tensor,
    gamma: float = 1.5,
    alpha: float = 0.25,
) -> Tensor:
    """Focal binary cross-entropy loss (per-element, averaged over elements).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        pred:   Raw logits, any shape.
        target: Binary targets (0/1), same shape as pred.
        gamma:  Focusing parameter (default 1.5).
        alpha:  Weighting factor for positive class (default 0.25).

    Returns:
        Scalar mean focal loss.
    """
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p_t = torch.exp(-bce)  # probability of true class
    focal_weight = alpha * (1.0 - p_t) ** gamma
    return (focal_weight * bce).mean()


# ---------------------------------------------------------------------------
# Main loss class
# ---------------------------------------------------------------------------


class YOLOLoss(nn.Module):
    """Anchor-free YOLO detection loss (CIoU + BCE obj + Focal cls).

    Args:
        cfg:    Full YAML config dict.  Reads ``cfg['loss']`` and
                ``cfg['model']['num_classes']``.
        device: Device string (e.g. ``'cpu'``, ``'cuda:0'``).

    ``cfg['loss']`` expected keys (with defaults):
        box_weight : float = 5.0
        cls_weight : float = 0.5
        obj_weight : float = 1.0
        iou_type   : str   = 'ciou'
    """

    def __init__(self, cfg: Dict, device: str = "cpu") -> None:
        super().__init__()
        loss_cfg = cfg.get("loss", {})
        self.box_weight: float = float(loss_cfg.get("box_weight", 5.0))
        self.cls_weight: float = float(loss_cfg.get("cls_weight", 0.5))
        self.obj_weight: float = float(loss_cfg.get("obj_weight", 1.0))
        self.aux_cls_weight: float = float(loss_cfg.get("aux_cls_weight", 0.5))
        self.iou_type:   str   = loss_cfg.get("iou_type", "wiou")
        self.num_classes: int  = cfg["model"]["num_classes"]
        self.cls_alpha = loss_cfg.get("cls_alpha", 0.25)
        self.strides: List[int] = loss_cfg.get("strides", [8, 16, 32])
        self.device = device

        # ---- Target assignment (added in rev. 2) ---------------------------------
        # 'center'      : original rule - one positive cell (the one containing the box
        #                 centre) on EVERY level, whatever the object size.
        # 'scale_aware' : each object is assigned only to the level(s) whose size range
        #                 contains max(w, h), and to every cell of that level whose centre
        #                 lies inside the box and within `center_radius` strides of the
        #                 box centre (the centre cell is always positive). A cell claimed by
        #                 several objects goes to the smallest one. Requires
        #                 model.center_decode = 'linear' so that off-centre cells can
        #                 regress the object centre.
        self.assigner: str = loss_cfg.get("assigner", "center")
        if self.assigner not in ("center", "scale_aware"):
            raise ValueError(f"loss.assigner must be 'center' or 'scale_aware', got {self.assigner!r}")
        self.center_radius: float = float(loss_cfg.get("center_radius", 2.5))
        self.level_ranges = loss_cfg.get("level_ranges", None)   # list of [lo, hi] px on max(w,h)
        # Objectness target at positives: 1.0 (original) or the IoU of the predicted box.
        self.obj_iou_target: bool = bool(loss_cfg.get("obj_iou_target", False))
        self.center_decode: str = cfg.get("model", {}).get("center_decode", "sigmoid")
        if self.assigner == "scale_aware" and self.center_decode != "linear":
            raise ValueError("loss.assigner='scale_aware' needs model.center_decode='linear' "
                             "(the sigmoid decode cannot reach the object centre from neighbouring cells).")

    @staticmethod
    def _default_level_ranges(n_levels: int):
        """Size ranges (pixels, on max(w, h)) per level. Neighbouring ranges overlap so
        objects near a boundary are learned on two levels."""
        if n_levels == 4:   # P2, P3, P4, P5
            return [[0, 64], [32, 128], [96, 256], [192, 1e9]]
        return [[0, 96], [64, 256], [192, 1e9]]   # P3, P4, P5

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_grid(
        grid_size: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """Build (grid_y, grid_x) cell index tensors for a given scale.

        Returns:
            grid_y: (grid_size*grid_size,) row indices (0, 1, ..., grid_size-1).
            grid_x: (grid_size*grid_size,) col indices (0, 1, ..., grid_size-1).
        """
        gy, gx = torch.meshgrid(
            torch.arange(grid_size, dtype=torch.float32, device=device),
            torch.arange(grid_size, dtype=torch.float32, device=device),
            indexing="ij",
        )
        return gy.reshape(-1), gx.reshape(-1)

    def _assign_targets(
        self,
        targets: Tensor,
        grid_size: int,
        stride: int,
        resolution: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Assign ground-truth boxes to anchor cells by centre-point rule.

        Each GT box is assigned to the cell whose top-left corner is
        closest to the box centre (equivalently: the cell the centre falls in).

        Args:
            targets:     (N_gt, 6) — [batch_idx, cls, cx_n, cy_n, w_n, h_n]
                         Coordinates are normalised to [0, 1].
            grid_size:   H == W of the current feature map.
            stride:      Downsampling stride for this scale.
            resolution:  Input image resolution in pixels (assumed square).

        Returns:
            batch_ids  : (N_pos,)  batch indices of positives.
            cell_ids   : (N_pos,)  flattened anchor indices.
            cls_ids    : (N_pos,)  class indices.
            gt_boxes_px: (N_pos, 4) GT boxes in absolute pixels (cx, cy, w, h).
        """
        if targets.shape[0] == 0:
            empty = targets.new_zeros(0, dtype=torch.long)
            return empty, empty, empty, targets.new_zeros(0, 4)

        # Convert normalised coords -> pixel coords
        cx_px = targets[:, 2] * resolution
        cy_px = targets[:, 3] * resolution
        w_px  = targets[:, 4] * resolution
        h_px  = targets[:, 5] * resolution

        # Cell indices: (cx_px / stride).long(), clamped to grid
        col_idx = (cx_px / stride).long().clamp(0, grid_size - 1)
        row_idx = (cy_px / stride).long().clamp(0, grid_size - 1)
        cell_ids = row_idx * grid_size + col_idx  # flattened

        batch_ids   = targets[:, 0].long()
        cls_ids     = targets[:, 1].long()
        gt_boxes_px = torch.stack([cx_px, cy_px, w_px, h_px], dim=1)

        return batch_ids, cell_ids, cls_ids, gt_boxes_px

    def _assign_scale_aware(
        self,
        targets: Tensor,
        grid_size: int,
        stride: int,
        resolution: int,
        size_range,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Scale-aware, multi-positive centre-sampling assignment (see __init__).

        Returns the same tuple as ``_assign_targets``; every (batch, cell) pair
        appears at most once.
        """
        dev = targets.device
        empty = (targets.new_zeros(0, dtype=torch.long),) * 3 + (targets.new_zeros(0, 4),)
        if targets.shape[0] == 0:
            return empty

        cx = targets[:, 2] * resolution
        cy = targets[:, 3] * resolution
        w = targets[:, 4] * resolution
        h = targets[:, 5] * resolution
        size = torch.maximum(w, h)
        lo, hi = float(size_range[0]), float(size_range[1])
        keep = (size > lo) & (size <= hi)
        if not bool(keep.any()):
            return empty
        t, cx, cy, w, h = targets[keep], cx[keep], cy[keep], w[keep], h[keep]

        ci = (cx / stride).long().clamp(0, grid_size - 1)
        cj = (cy / stride).long().clamp(0, grid_size - 1)
        r = int(math.ceil(self.center_radius))
        rng = torch.arange(-r, r + 1, device=dev)
        oy, ox = torch.meshgrid(rng, rng, indexing="ij")
        ox, oy = ox.reshape(-1), oy.reshape(-1)                       # (K,)
        gx = ci[:, None] + ox[None, :]                                # (N, K)
        gy = cj[:, None] + oy[None, :]
        valid = (gx >= 0) & (gx < grid_size) & (gy >= 0) & (gy < grid_size)
        ccx = (gx.float() + 0.5) * stride
        ccy = (gy.float() + 0.5) * stride
        in_box = ((ccx - cx[:, None]).abs() < w[:, None] / 2) & ((ccy - cy[:, None]).abs() < h[:, None] / 2)
        rad = self.center_radius * stride
        in_ctr = ((ccx - cx[:, None]).abs() < rad) & ((ccy - cy[:, None]).abs() < rad)
        is_ctr = ((ox == 0) & (oy == 0))[None, :]
        pos = valid & ((in_box & in_ctr) | is_ctr)

        gi, ki = pos.nonzero(as_tuple=True)
        cell = gy[gi, ki] * grid_size + gx[gi, ki]
        bid = t[gi, 0].long()
        area = (w * h)[gi]

        # Conflict resolution: the smallest object keeps a contested cell.
        order = torch.argsort(area)
        key = (bid * grid_size * grid_size + cell)[order]
        uniq, inv = torch.unique(key, return_inverse=True)
        first = torch.full((uniq.numel(),), key.numel(), dtype=torch.long, device=dev)
        first = first.scatter_reduce(0, inv, torch.arange(key.numel(), device=dev), reduce="amin")
        sel = order[first]

        boxes = torch.stack([cx, cy, w, h], dim=1)[gi[sel]]
        return bid[sel], cell[sel], t[gi[sel], 1].long(), boxes

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        predictions: List[Tensor],
        targets: Tensor,
        resolution: int,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute the total YOLO loss over all scales.

        Args:
            predictions: List of 3 tensors from the detection head, each
                         shape (B, Hi*Wi, 4 + 1 + num_classes) — raw logits.
            targets:     (N_gt, 6) ground-truth tensor with columns:
                         [batch_idx, class_id, cx_norm, cy_norm, w_norm, h_norm].
            resolution:  Input image size in pixels (square assumed).

        Returns:
            total_loss:  Scalar weighted sum of all loss components.
            loss_dict:   Dict ``{'box': float, 'cls': float, 'obj': float}``
                         with the unweighted individual losses for logging.
        """
        device = predictions[0].device
        targets = targets.to(device)

        total_box_loss = torch.zeros(1, device=device)
        total_obj_loss = torch.zeros(1, device=device)
        total_cls_loss = torch.zeros(1, device=device)

        num_pos_total = 0  # used for normalisation

        # Dynamic strides for 3-scale (v1) or 4-scale (v2 with P2)
        strides = [4, 8, 16, 32] if len(predictions) == 4 else self.strides
        ranges = self.level_ranges or self._default_level_ranges(len(predictions))

        for level, (pred, stride) in enumerate(zip(predictions, strides)):
            B, num_anchors, _ = pred.shape
            grid_size = int(num_anchors ** 0.5)  # H == W

            # ------------------------------------------------------------------
            # 1. Build target masks for this scale
            # ------------------------------------------------------------------
            obj_target = torch.zeros(B, num_anchors, device=device)
            cls_target = torch.zeros(B, num_anchors, self.num_classes, device=device)
            box_target  = torch.zeros(B, num_anchors, 4,               device=device)

            if self.assigner == "scale_aware":
                batch_ids, cell_ids, cls_ids, gt_boxes_px = self._assign_scale_aware(
                    targets, grid_size, stride, resolution, ranges[level]
                )
            else:
                batch_ids, cell_ids, cls_ids, gt_boxes_px = self._assign_targets(
                    targets, grid_size, stride, resolution
                )

            num_pos = len(batch_ids)
            if num_pos > 0:
                obj_target[batch_ids, cell_ids] = 1.0
                cls_target[batch_ids, cell_ids, cls_ids] = 1.0
                box_target[batch_ids, cell_ids] = gt_boxes_px
            num_pos_total += num_pos

            # ------------------------------------------------------------------
            # 3. Per-positive box + class losses
            # ------------------------------------------------------------------
            if num_pos > 0:
                # --- Box regression (IoU/CIoU/SIoU/WIoU) ---
                grid_y_flat, grid_x_flat = self._build_grid(grid_size, device)

                # Predicted box components at positive anchors
                pred_pos = pred[batch_ids, cell_ids]          # (N_pos, 4+1+nc)
                tx = pred_pos[:, 0]
                ty = pred_pos[:, 1]
                tw = pred_pos[:, 2]
                th = pred_pos[:, 3]

                if self.center_decode == "linear":
                    cx_pred = (tx + grid_x_flat[cell_ids] + 0.5) * stride
                    cy_pred = (ty + grid_y_flat[cell_ids] + 0.5) * stride
                else:
                    cx_pred = (torch.sigmoid(tx) + grid_x_flat[cell_ids]) * stride
                    cy_pred = (torch.sigmoid(ty) + grid_y_flat[cell_ids]) * stride
                w_pred  = torch.exp(tw).clamp(max=resolution) * stride
                h_pred  = torch.exp(th).clamp(max=resolution) * stride

                pred_boxes_px = torch.stack([cx_pred, cy_pred, w_pred, h_pred], dim=1)
                tgt_boxes_px  = box_target[batch_ids, cell_ids]  # (N_pos, 4)

                iou_val = bbox_iou(pred_boxes_px, tgt_boxes_px, iou_type=self.iou_type)
                box_loss = (1.0 - iou_val).mean()
                total_box_loss = total_box_loss + box_loss

                if self.obj_iou_target:
                    # IoU-aware objectness: positives are pushed towards the quality of
                    # their own box, which ranks well-localised boxes first at test time.
                    with torch.no_grad():
                        q = bbox_iou(pred_boxes_px.float(), tgt_boxes_px.float(), iou_type="iou").clamp(0, 1)
                    obj_target[batch_ids, cell_ids] = q.to(obj_target.dtype)

                # --- Classification loss (Focal BCE) ---
                cls_pred_pos = pred[batch_ids, cell_ids, 5:]           # (N_pos, nc)
                cls_tgt_pos  = cls_target[batch_ids, cell_ids]         # (N_pos, nc)
                cls_loss = focal_bce_loss(cls_pred_pos, cls_tgt_pos, alpha=self.cls_alpha)

                # Auxiliary Bird <-> Drone confusion penalty
                if self.aux_cls_weight > 0 and self.num_classes == 2:
                    aux_loss = F.cross_entropy(cls_pred_pos, cls_ids)
                    cls_loss = cls_loss + self.aux_cls_weight * aux_loss

                total_cls_loss = total_cls_loss + cls_loss

            # ------------------------------------------------------------------
            # Objectness loss (BCE, all anchors) - computed after the box loss so
            # that the optional IoU-aware targets are available.
            # ------------------------------------------------------------------
            obj_pred = pred[..., 4]   # (B, num_anchors) raw logit
            obj_loss = F.binary_cross_entropy_with_logits(
                obj_pred, obj_target, reduction="mean"
            )
            total_obj_loss = total_obj_loss + obj_loss

        # ------------------------------------------------------------------
        # Weighted total
        # ------------------------------------------------------------------
        total_loss = (
            self.box_weight * total_box_loss
            + self.obj_weight * total_obj_loss
            + self.cls_weight * total_cls_loss
        )

        loss_dict: Dict[str, float] = {
            "box": float(total_box_loss.item()),
            "obj": float(total_obj_loss.item()),
            "cls": float(total_cls_loss.item()),
        }

        return total_loss.squeeze(), loss_dict


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simulate model predictions
    B, nc = 2, 2
    num_outputs = 4 + 1 + nc
    preds = [
        torch.randn(B, 80 * 80, num_outputs),
        torch.randn(B, 40 * 40, num_outputs),
        torch.randn(B, 20 * 20, num_outputs),
    ]

    # Ground-truth targets: 5 boxes across 2 images
    # [batch_idx, cls, cx_n, cy_n, w_n, h_n]
    targets = torch.tensor([
        [0, 0, 0.3, 0.4, 0.1, 0.15],
        [0, 1, 0.6, 0.6, 0.2, 0.25],
        [1, 0, 0.5, 0.5, 0.3, 0.3],
        [1, 1, 0.2, 0.8, 0.05, 0.08],
        [1, 0, 0.7, 0.3, 0.12, 0.18],
    ])

    cfg = {
        "model": {"num_classes": nc},
        "loss":  {
            "box_weight": 5.0,
            "cls_weight": 0.5,
            "obj_weight": 1.0,
            "iou_type":   "ciou",
        },
    }

    criterion = YOLOLoss(cfg, device="cpu")

    loss, components = criterion(preds, targets, resolution=640)

    print(f"Total loss : {loss.item():.4f}")
    print(f"  box_loss : {components['box']:.4f}")
    print(f"  obj_loss : {components['obj']:.4f}")
    print(f"  cls_loss : {components['cls']:.4f}")

    assert loss.ndim == 0, "Loss must be a scalar tensor"
    assert loss.item() > 0, "Loss must be positive"

    # --- bbox_iou sanity checks ---
    b1 = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    b2 = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    assert abs(float(bbox_iou(b1, b2, "iou"))  - 1.0) < 1e-5, "IoU self-overlap must be 1"
    assert abs(float(bbox_iou(b1, b2, "giou")) - 1.0) < 1e-5, "GIoU self-overlap must be 1"
    assert abs(float(bbox_iou(b1, b2, "diou")) - 1.0) < 1e-5, "DIoU self-overlap must be 1"
    assert abs(float(bbox_iou(b1, b2, "ciou")) - 1.0) < 1e-5, "CIoU self-overlap must be 1"

    print("\nAll YOLOLoss assertions passed [OK]")
