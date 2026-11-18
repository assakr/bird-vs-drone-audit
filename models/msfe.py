"""
models/msfe.py
Multi-Scale Feature Enhancement Module (Section 5.2).

Applies per-scale residual refinement to each of the P3/P4/P5 feature maps
without changing their spatial resolution or channel count.

Each FeatureEnhancementBlock uses three parallel branches:
  1. 3×3 depthwise-separable conv  -> local texture detail
  2. 3×3 dilated conv (dilation=2) -> wider receptive field without stride
  3. 1×1 conv                      -> lightweight channel mixing

Their outputs are concatenated and projected back to the original channel
count. A residual connection is then added so the block cannot harm the
already-informative backbone features.

When ``enabled=False`` the module is a no-op identity, enabling clean
ablation studies (Section 6.2).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class FeatureEnhancementBlock(nn.Module):
    """Per-scale residual feature enhancement with three parallel branches.

    All three branches operate on the same (B, C, H, W) input and preserve
    spatial dimensions. Their outputs are concatenated to (B, 3C, H, W) and
    projected back to (B, C, H, W) via a 1×1 conv. A residual is added.

    Args:
        channels: Number of input (and output) channels C.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        c = channels

        # Branch 1: Depthwise separable conv (local detail)
        self.branch_dw = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False),  # depthwise
            nn.BatchNorm2d(c, eps=1e-3, momentum=0.03),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=1, bias=False),                       # pointwise
            nn.BatchNorm2d(c, eps=1e-3, momentum=0.03),
            nn.SiLU(inplace=True),
        )

        # Branch 2: Dilated 3×3 conv (wider context without resolution loss)
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(c, eps=1e-3, momentum=0.03),
            nn.SiLU(inplace=True),
        )

        # Branch 3: 1×1 conv (efficient channel mixing)
        self.branch_pw = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c, eps=1e-3, momentum=0.03),
            nn.SiLU(inplace=True),
        )

        # Projection: 3C -> C (fuse all branches back to original channel count)
        self.project = nn.Sequential(
            nn.Conv2d(3 * c, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c, eps=1e-3, momentum=0.03),
        )

        # Residual activation applied after adding skip
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply three-branch enhancement and add residual.

        Args:
            x: Feature map of shape (B, C, H, W).

        Returns:
            Refined feature map of shape (B, C, H, W) — same as input.
        """
        b1 = self.branch_dw(x)
        b2 = self.branch_dilated(x)
        b3 = self.branch_pw(x)

        # Concatenate along channel axis -> (B, 3C, H, W)
        fused = torch.cat([b1, b2, b3], dim=1)

        # Project back to C channels and add residual skip
        out = self.act(self.project(fused) + x)
        return out


class MultiScaleFeatureEnhancement(nn.Module):
    """Apply FeatureEnhancementBlock independently to each of P3, P4, P5.

    Can be toggled off for ablation studies via the ``enabled`` flag.

    Args:
        channels: Dict mapping scale names to channel counts,
                  e.g. ``{'P3': 128, 'P4': 256, 'P5': 256}``.
        enabled:  If False the module acts as a pure identity (no parameters
                  consumed in the forward pass).
    """

    def __init__(self, channels: Dict[str, int], enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled

        if enabled:
            # Build one enhancement block per scale
            self.blocks = nn.ModuleDict(
                {scale: FeatureEnhancementBlock(ch) for scale, ch in channels.items()}
            )

    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Enhance (or pass through) each feature map.

        Args:
            features: Dict ``{'P3': Tensor, 'P4': Tensor, 'P5': Tensor}``.

        Returns:
            Dict with the same keys; values are enhanced tensors (or the
            originals if ``enabled=False``).
        """
        if not self.enabled:
            return features  # identity for ablation

        enhanced: Dict[str, Tensor] = {}
        for scale, feat in features.items():
            out = self.blocks[scale](feat)
            # Safety check: shape must be preserved
            assert out.shape == feat.shape, (
                f"MSFE output shape mismatch at {scale}: "
                f"expected {feat.shape}, got {out.shape}"
            )
            enhanced[scale] = out

        return enhanced


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    channels = {"P3": 128, "P4": 256, "P5": 256}

    # Simulated backbone outputs (batch=2, input=640×640)
    dummy_features: Dict[str, Tensor] = {
        "P3": torch.zeros(2, 128, 80, 80),
        "P4": torch.zeros(2, 256, 40, 40),
        "P5": torch.zeros(2, 256, 20, 20),
    }

    # --- Enabled mode ---
    msfe = MultiScaleFeatureEnhancement(channels, enabled=True)
    with torch.no_grad():
        out = msfe(dummy_features)

    for scale, feat in out.items():
        print(f"MSFE(enabled) {scale}: {tuple(feat.shape)}")
        assert feat.shape == dummy_features[scale].shape

    # --- Disabled (identity) mode ---
    msfe_off = MultiScaleFeatureEnhancement(channels, enabled=False)
    out_off = msfe_off(dummy_features)
    for scale in ["P3", "P4", "P5"]:
        assert out_off[scale] is dummy_features[scale], \
            f"Identity mode should return original tensor for {scale}"

    print("All MSFE shape assertions passed [OK]")
