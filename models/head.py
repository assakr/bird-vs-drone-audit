"""
models/head.py
Anchor-free YOLO detection head.

Each scale (P3/P4/P5) has its own DetectionBranch that predicts:
  - 4 values: (x, y, w, h) offsets relative to each grid cell
  - 1 value:  objectness / centerness score
  - num_classes values: class logits

The head operates in raw-logit mode during training. For inference /
evaluation, ``decode_predictions`` converts raw offsets to absolute
pixel coordinates and applies sigmoid to objectness and class scores.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

try:
    from .backbone import ConvBnAct
except (ImportError, ValueError):
    from models.backbone import ConvBnAct


class DetectionBranch(nn.Module):
    """Single-scale anchor-free detection branch.

    Architecture:
        ConvBnAct(C, C, k=3)
        ConvBnAct(C, C, k=3)
        Conv2d(C, num_outputs, k=1)   [raw predictions, no activation]

    The output is reshaped from (B, num_outputs, H, W) to
    (B, H*W, num_outputs) for compatibility with the loss function, which
    iterates over all spatial locations as candidate anchors.

    Args:
        in_ch:       Input channel count C.
        num_classes: Number of object classes (default 2).
    """

    def __init__(self, in_ch: int, num_classes: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = 4 + 1 + num_classes  # xywh + obj + cls

        self.stem = nn.Sequential(
            ConvBnAct(in_ch, in_ch, k=3),   # spatial feature mixing
            ConvBnAct(in_ch, in_ch, k=3),   # second refinement pass
        )
        # Final 1×1 conv produces raw predictions (no BN, no activation)
        self.pred = nn.Conv2d(in_ch, self.num_outputs, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Predict at a single scale.

        Args:
            x: Feature map of shape (B, C, H, W).

        Returns:
            Prediction tensor of shape (B, H*W, 4 + 1 + num_classes).
        """
        b, _, h, w = x.shape
        feat = self.stem(x)                                # (B, C, H, W)
        pred = self.pred(feat)                             # (B, num_outputs, H, W)
        # Flatten spatial dims -> anchor candidates for loss compatibility
        pred = pred.permute(0, 2, 3, 1).reshape(b, h * w, self.num_outputs)
        return pred                                        # (B, H*W, num_outputs)


class DetectionHead(nn.Module):
    """Three-scale anchor-free detection head.

    One DetectionBranch is instantiated per feature scale (P3, P4, P5).
    During training, raw logits are returned. During evaluation,
    ``decode_predictions`` is called to obtain absolute coordinates.

    Args:
        channels:    Dict ``{'P3': int, 'P4': int, 'P5': int}``.
        num_classes: Number of object categories (default 2).
        strides:     Downsampling strides corresponding to P3, P4, P5.
    """

    def __init__(
        self,
        channels: Dict[str, int],
        num_classes: int = 2,
        strides: List[int] | None = None,
        center_decode: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        # 'sigmoid' (original): centre = (sigmoid(t) + grid) * stride, confined to the own cell.
        # 'linear'  (new):      centre = (t + grid + 0.5) * stride, can reach neighbouring cells;
        #                       required when the loss assigns several positive cells per object.
        if center_decode not in ("sigmoid", "linear"):
            raise ValueError(f"center_decode must be 'sigmoid' or 'linear', got {center_decode!r}")
        self.center_decode = center_decode
        if "P2" in channels:
            self.scale_keys = ["P2", "P3", "P4", "P5"]
            default_strides = [4, 8, 16, 32]
        else:
            self.scale_keys = ["P3", "P4", "P5"]
            default_strides = [8, 16, 32]

        self.strides: List[int] = strides if strides is not None else default_strides
        self.num_outputs = 4 + 1 + num_classes

        # One branch per scale
        self.branches = nn.ModuleList([
            DetectionBranch(channels[key], num_classes)
            for key in self.scale_keys
        ])

    def forward(self, features: Dict[str, Tensor]) -> List[Tensor]:
        """Run detection branch on each feature scale.

        Args:
            features: Dict mapping scale names ('P2', 'P3', 'P4', 'P5') to tensors.

        Returns:
            List of prediction tensors [P_pred, ...],
            each of shape (B, Hi*Wi, 4 + 1 + num_classes).
        """
        return [
            branch(features[key])
            for branch, key in zip(self.branches, self.scale_keys)
        ]

    # ------------------------------------------------------------------
    # Decode helper (used at inference / evaluation time)
    # ------------------------------------------------------------------

    def decode_predictions(
        self,
        preds: List[Tensor],
        resolution: int,
    ) -> Tensor:
        """Convert raw head outputs to absolute pixel coordinates.

        Decoding rules (anchor-free, center-offset):
            x_abs = sigmoid(tx) * stride + grid_col * stride
            y_abs = sigmoid(ty) * stride + grid_row * stride
            w_abs = exp(tw) * stride
            h_abs = exp(th) * stride
            obj   = sigmoid(to)
            cls   = sigmoid(tc_i)   for i in [0, num_classes)

        All scales are decoded and concatenated into a single tensor.

        Args:
            preds:      List of raw prediction tensors from ``forward``.
            resolution: Input image resolution (assumed square, in pixels).

        Returns:
            Decoded tensor of shape (B, total_anchors, 4 + 1 + num_classes)
            where total_anchors = sum(Hi * Wi) over all scales.
        """
        decoded_scales: List[Tensor] = []

        for pred, stride in zip(preds, self.strides):
            b, num_anchors, _ = pred.shape
            grid_size = int(num_anchors ** 0.5)  # H == W at each scale

            # Build (H, W) grid of cell top-left coordinates
            # grid_x[i,j] = j * stride,  grid_y[i,j] = i * stride
            grid_y, grid_x = torch.meshgrid(
                torch.arange(grid_size, device=pred.device, dtype=pred.dtype),
                torch.arange(grid_size, device=pred.device, dtype=pred.dtype),
                indexing="ij",
            )
            # Flatten to (num_anchors,) vectors
            grid_x = grid_x.reshape(-1)   # (H*W,)
            grid_y = grid_y.reshape(-1)   # (H*W,)

            # Split prediction along last dim
            tx = pred[..., 0]   # (B, H*W)
            ty = pred[..., 1]
            tw = pred[..., 2]
            th = pred[..., 3]
            obj = pred[..., 4:5]          # (B, H*W, 1)
            cls = pred[..., 5:]           # (B, H*W, num_classes)

            # Decode absolute coords (pixel space)
            if self.center_decode == "linear":
                x_abs = (tx + grid_x + 0.5) * stride
                y_abs = (ty + grid_y + 0.5) * stride
            else:
                x_abs = (torch.sigmoid(tx) + grid_x) * stride   # (B, H*W)
                y_abs = (torch.sigmoid(ty) + grid_y) * stride
            w_abs = torch.exp(tw).clamp(max=resolution) * stride
            h_abs = torch.exp(th).clamp(max=resolution) * stride

            boxes = torch.stack([x_abs, y_abs, w_abs, h_abs], dim=-1)  # (B, H*W, 4)
            obj   = torch.sigmoid(obj)                                   # (B, H*W, 1)
            cls   = torch.sigmoid(cls)                                   # (B, H*W, nc)

            decoded = torch.cat([boxes, obj, cls], dim=-1)  # (B, H*W, 4+1+nc)
            decoded_scales.append(decoded)

        # Concatenate all scales along anchor dimension
        return torch.cat(decoded_scales, dim=1)   # (B, total_anchors, 4+1+nc)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    num_classes = 2
    channels = {"P3": 128, "P4": 256, "P5": 256}

    head = DetectionHead(channels, num_classes=num_classes)

    dummy_features: Dict[str, Tensor] = {
        "P3": torch.zeros(2, 128, 80, 80),
        "P4": torch.zeros(2, 256, 40, 40),
        "P5": torch.zeros(2, 256, 20, 20),
    }

    with torch.no_grad():
        preds = head(dummy_features)

    expected_anchors = [80 * 80, 40 * 40, 20 * 20]
    for i, (pred, na) in enumerate(zip(preds, expected_anchors)):
        print(f"P{3+i}_pred: {tuple(pred.shape)}")
        assert pred.shape == (2, na, 4 + 1 + num_classes), \
            f"Head output shape mismatch at scale {i}: {pred.shape}"

    # Test decode
    decoded = head.decode_predictions(preds, resolution=640)
    total_anchors = sum(expected_anchors)
    print(f"Decoded: {tuple(decoded.shape)}")
    assert decoded.shape == (2, total_anchors, 4 + 1 + num_classes), \
        f"Decoded shape mismatch: {decoded.shape}"

    print("All DetectionHead shape assertions passed [OK]")
