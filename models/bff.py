"""
models/bff.py
Efficient Bi-directional Feature Fusion (Section 5.4).

Implements a PANet-style neck:
  * Top-down path (FPN)  : P5 -> P4 -> P3   (high-level semantics enrich lower levels)
  * Bottom-up path (PAN) : P3 -> P4 -> P5   (high-resolution details enrich upper levels)

When ``enabled=False``, only the top-down (FPN) path is executed, providing
a natural single-direction baseline for ablation studies.

Channel layout after fusion (for base_channels=32):
  P3: 4c = 128,  P4: 8c = 256,  P5: 8c = 256
All output channel counts match the corresponding input channel counts
so downstream modules (detection head) need no channel adapters.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

try:
    from .backbone import ConvBnAct
except (ImportError, ValueError):
    from models.backbone import ConvBnAct


class FusionBlock(nn.Module):
    """Concatenate two same-spatial-size feature maps and project to ``out_ch``.

    Used to merge top-down / bottom-up paths with the lateral connections
    at each scale.

    Args:
        in_ch_a: Channel count of the first (upsampled/downsampled) feature map.
        in_ch_b: Channel count of the second (lateral) feature map.
        out_ch:  Channel count of the fused output.
    """

    def __init__(self, in_ch_a: int, in_ch_b: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBnAct(in_ch_a + in_ch_b, out_ch, k=3)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """Concatenate a and b along channel dim and project.

        Args:
            a: First feature map  (B, in_ch_a, H, W).
            b: Second feature map (B, in_ch_b, H, W).

        Returns:
            Fused feature map (B, out_ch, H, W).
        """
        return self.conv(torch.cat([a, b], dim=1))


class BidirectionalFeatureFusion(nn.Module):
    """FPN top-down + PAN bottom-up feature fusion neck.

    Top-down path
    -------------
    p5_td = P5                                   (no spatial change)
    p4_td = FusionBlock(upsample(p5_td) ‖ P4)
    p3_td = FusionBlock(upsample(p4_td) ‖ P3)

    Bottom-up path (only when enabled)
    -----------------------------------
    p4_bu = FusionBlock(downsample(p3_td) ‖ p4_td)
    p5_bu = FusionBlock(downsample(p4_bu) ‖ p5_td)

    Output scales:
      If enabled  -> {'P3': p3_td, 'P4': p4_bu, 'P5': p5_bu}
      If disabled -> {'P3': p3_td, 'P4': p4_td, 'P5': p5_td}

    Args:
        channels: Dict mapping ``{'P3': int, 'P4': int, 'P5': int}`` to
                  input channel counts (from backbone / CAAM output).
        enabled:  If False, only the top-down FPN path is applied.
    """

    def __init__(self, channels: Dict[str, int], enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled

        self.has_p2 = "P2" in channels
        p3_ch = channels["P3"]
        p4_ch = channels["P4"]
        p5_ch = channels["P5"]

        # ------------------------------------------------------------------
        # Top-down path blocks
        # ------------------------------------------------------------------
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # td_p4 fuses: upsample(P5) [p5_ch] + P4 [p4_ch] -> p4_ch
        self.td_p4 = FusionBlock(p5_ch, p4_ch, p4_ch)

        # td_p3 fuses: upsample(p4_td) [p4_ch] + P3 [p3_ch] -> p3_ch
        self.td_p3 = FusionBlock(p4_ch, p3_ch, p3_ch)

        if self.has_p2:
            p2_ch = channels["P2"]
            self.td_p2 = FusionBlock(p3_ch, p2_ch, p2_ch)
            self.down_p2 = ConvBnAct(p2_ch, p2_ch, k=3, s=2)
            self.bu_p3 = FusionBlock(p2_ch, p3_ch, p3_ch)

        # ------------------------------------------------------------------
        # Bottom-up path blocks (built regardless; used only if enabled)
        # ------------------------------------------------------------------
        self.down_p3 = ConvBnAct(p3_ch, p3_ch, k=3, s=2)  # p3 -> half res
        self.down_p4 = ConvBnAct(p4_ch, p4_ch, k=3, s=2)  # p4 -> half res

        self.bu_p4 = FusionBlock(p3_ch, p4_ch, p4_ch)
        self.bu_p5 = FusionBlock(p4_ch, p5_ch, p5_ch)

        # Store output channel counts for downstream modules
        self.out_channels: Dict[str, int] = {}
        if self.has_p2:
            self.out_channels["P2"] = channels["P2"]
        self.out_channels["P3"] = p3_ch
        self.out_channels["P4"] = p4_ch
        self.out_channels["P5"] = p5_ch

    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Run top-down (and optionally bottom-up) fusion.

        Args:
            features: Dict with keys 'P3', 'P4', 'P5' (and optionally 'P2').

        Returns:
            Dict with fused feature maps at each scale.
        """
        p3 = features["P3"]
        p4 = features["P4"]
        p5 = features["P5"]

        # ------------------------------------------------------------------
        # Top-down path
        # ------------------------------------------------------------------
        p5_td = p5
        p4_td = self.td_p4(self.upsample(p5_td), p4)
        p3_td = self.td_p3(self.upsample(p4_td), p3)

        if self.has_p2:
            p2 = features["P2"]
            p2_td = self.td_p2(self.upsample(p3_td), p2)

        if not self.enabled:
            if self.has_p2:
                return {"P2": p2_td, "P3": p3_td, "P4": p4_td, "P5": p5_td}
            return {"P3": p3_td, "P4": p4_td, "P5": p5_td}

        # ------------------------------------------------------------------
        # Bottom-up path (PAN)
        # ------------------------------------------------------------------
        if self.has_p2:
            p3_bu = self.bu_p3(self.down_p2(p2_td), p3_td)
            p4_bu = self.bu_p4(self.down_p3(p3_bu), p4_td)
            p5_bu = self.bu_p5(self.down_p4(p4_bu), p5_td)
            return {"P2": p2_td, "P3": p3_bu, "P4": p4_bu, "P5": p5_bu}
        else:
            p4_bu = self.bu_p4(self.down_p3(p3_td), p4_td)
            p5_bu = self.bu_p5(self.down_p4(p4_bu), p5_td)
            return {"P3": p3_td, "P4": p4_bu, "P5": p5_bu}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    channels = {"P3": 128, "P4": 256, "P5": 256}

    dummy_features: Dict[str, Tensor] = {
        "P3": torch.zeros(2, 128, 80, 80),
        "P4": torch.zeros(2, 256, 40, 40),
        "P5": torch.zeros(2, 256, 20, 20),
    }

    # --- Bi-directional (full BFF) ---
    bff = BidirectionalFeatureFusion(channels, enabled=True)
    with torch.no_grad():
        out = bff(dummy_features)

    expected = {"P3": (2, 128, 80, 80), "P4": (2, 256, 40, 40), "P5": (2, 256, 20, 20)}
    for scale, feat in out.items():
        print(f"BFF(enabled) {scale}: {tuple(feat.shape)}")
        assert tuple(feat.shape) == expected[scale], \
            f"BFF shape mismatch at {scale}: got {feat.shape}"

    # --- Top-down only (FPN baseline) ---
    bff_td = BidirectionalFeatureFusion(channels, enabled=False)
    with torch.no_grad():
        out_td = bff_td(dummy_features)

    for scale, feat in out_td.items():
        print(f"BFF(disabled) {scale}: {tuple(feat.shape)}")
        assert tuple(feat.shape) == expected[scale], \
            f"BFF(disabled) shape mismatch at {scale}: got {feat.shape}"

    print("All BFF shape assertions passed [OK]")
