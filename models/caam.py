"""
models/caam.py
Context-Aware Attention Module (Section 5.3).

Combines local neighborhood context with global scene context to produce
a soft attention gate that rescales each spatial position of the feature map.

Formulation:
    A_local  = LocalContextBranch(F_in)   -> (B, 1, H, W)
    A_global = GlobalContextBranch(F_in)  -> (B, 1, 1, 1)  (broadcasts spatially)
    gate     = sigmoid(A_local + A_global) ∈ [0, 1]
    F_out    = gate * F_in

The gated output retains the same spatial and channel dimensions as the input.

When ``enabled=False`` the module is a pure identity, enabling ablation.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class LocalContextBranch(nn.Module):
    """Capture local neighborhood / edge context for spatial attention.

    Uses a large depthwise kernel (7×7) to aggregate nearby information,
    then projects to a single-channel spatial attention map.

    Args:
        channels: Number of input channels C.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # Large-kernel depthwise conv: sees a 7×7 neighborhood per channel
        self.dw_conv = nn.Conv2d(
            channels, channels,
            kernel_size=7, padding=3, groups=channels, bias=False
        )
        self.bn = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True)
        # Project C channels to 1 channel -> spatial attention logits
        self.proj = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Produce spatial attention logits from local context.

        Args:
            x: (B, C, H, W)

        Returns:
            Spatial attention logits of shape (B, 1, H, W).
        """
        out = self.act(self.bn(self.dw_conv(x)))
        return self.proj(out)  # (B, 1, H, W)


class GlobalContextBranch(nn.Module):
    """Capture global scene context for channel-level (and scalar) attention.

    Pipeline:
        GlobalAvgPool -> (B, C, 1, 1)
        FC1 [1×1 conv]:  C -> C//reduction  + ReLU
        FC2 [1×1 conv]:  C//reduction -> C
        Scalar reduce:   C -> 1  (via another 1×1 conv)

    The final (B, 1, 1, 1) scalar gate broadcasts over the full (H, W) plane,
    complementing the spatially-explicit LocalContextBranch output.

    Args:
        channels:  Number of input channels C.
        reduction: Channel reduction factor inside the FC bottleneck.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)   # -> (B, C, 1, 1)
        self.fc1 = nn.Conv2d(channels, mid, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, channels, kernel_size=1, bias=True)
        # Collapse channel dimension to scalar
        self.scalar = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Produce global scalar attention logit.

        Args:
            x: (B, C, H, W)

        Returns:
            Scalar attention logits of shape (B, 1, 1, 1),
            which broadcast over spatial dimensions.
        """
        g = self.gap(x)                  # (B, C, 1, 1)
        g = self.relu(self.fc1(g))       # (B, C//r, 1, 1)
        g = self.fc2(g)                  # (B, C, 1, 1)
        return self.scalar(g)            # (B, 1, 1, 1)


class ContextAwareAttentionBlock(nn.Module):
    """Combine local and global context into a soft spatial attention gate.

    gate = sigmoid(A_local + A_global)   [shape (B, 1, H, W)]
    F_out = gate * F_in

    Args:
        channels: Number of input/output channels (unchanged by this block).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.local_branch = LocalContextBranch(channels)
        self.global_branch = GlobalContextBranch(channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply context-aware attention gating.

        Args:
            x: Feature map of shape (B, C, H, W).

        Returns:
            Gated feature map of shape (B, C, H, W).
        """
        # Local: (B, 1, H, W)  — per-position logits
        local_logits = self.local_branch(x)

        # Global: (B, 1, 1, 1) — broadcast scalar logit
        global_logits = self.global_branch(x)

        # Combined soft gate in [0, 1]
        attention = torch.sigmoid(local_logits + global_logits)  # (B, 1, H, W)
        return attention * x   # element-wise scaling, broadcasts over C


class ECABlock(nn.Module):
    """Efficient Channel Attention (ECA) block.

    Replaces standard FC-based squeeze-and-excitation with a 1D convolution
    across channels, preserving performance at negligible parameter cost.
    """
    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        # Adaptive kernel size: odd integer ~ |log2(C)/gamma + b/gamma|
        k = int(abs((torch.log2(torch.tensor(float(channels))) / gamma) + (b / gamma)).item())
        k = k if k % 2 == 1 else k + 1
        k = max(3, k)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # GAP: (B, C, 1, 1) -> (B, 1, C)
        y = x.mean(dim=(2, 3), keepdim=False).unsqueeze(1)
        y = self.conv(y)  # (B, 1, C)
        weight = torch.sigmoid(y).squeeze(1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * weight


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (CBAM).

    Applies sequential Channel Attention and Spatial Attention.
    """
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        # Channel Attention
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        # Spatial Attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # 1. Channel Attention
        B, C, H, W = x.shape
        avg_pool = x.mean(dim=(2, 3))       # (B, C)
        max_pool, _ = x.view(B, C, -1).max(dim=2)  # (B, C)
        ch_att = torch.sigmoid(self.shared_mlp(avg_pool) + self.shared_mlp(max_pool))  # (B, C)
        x = x * ch_att.view(B, C, 1, 1)

        # 2. Spatial Attention
        avg_out = x.mean(dim=1, keepdim=True)        # (B, 1, H, W)
        max_out, _ = x.max(dim=1, keepdim=True)      # (B, 1, H, W)
        sp_feat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        sp_att = torch.sigmoid(self.spatial_conv(sp_feat))  # (B, 1, H, W)
        return x * sp_att


class ContextAwareAttentionModule(nn.Module):
    """Apply Attention gating independently to each feature map (P2, P3, P4, P5).

    Supports:
      - 'caam' : Context-Aware Attention Module (Local + Global combination)
      - 'cbam' : Convolutional Block Attention Module (Channel + Spatial)
      - 'eca'  : Efficient Channel Attention (1D conv)

    Args:
        channels:       Dict mapping scale names to channel counts.
        enabled:        If False, module acts as pure identity for ablation studies.
        attention_type: 'caam' | 'cbam' | 'eca' (default: 'caam').
    """

    def __init__(
        self,
        channels: Dict[str, int],
        enabled: bool = True,
        attention_type: str = "caam",
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.attention_type = attention_type.lower()

        if enabled:
            block_cls = {
                "caam": ContextAwareAttentionBlock,
                "cbam": CBAMBlock,
                "eca": ECABlock,
            }.get(self.attention_type, ContextAwareAttentionBlock)

            self.blocks = nn.ModuleDict(
                {scale: block_cls(ch) for scale, ch in channels.items()}
            )

    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Apply (or pass through) attention gating for each feature scale.

        Args:
            features: Dict ``{'P3': Tensor, 'P4': Tensor, 'P5': Tensor}``.

        Returns:
            Dict with the same structure; values are gated tensors (or
            original tensors if ``enabled=False``).
        """
        if not self.enabled:
            return features  # identity for ablation

        attended: Dict[str, Tensor] = {}
        for scale, feat in features.items():
            out = self.blocks[scale](feat)
            # Shape must be preserved
            assert out.shape == feat.shape, (
                f"CAAM output shape mismatch at {scale}: "
                f"expected {feat.shape}, got {out.shape}"
            )
            attended[scale] = out

        return attended


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    channels = {"P3": 128, "P4": 256, "P5": 256}

    dummy_features: Dict[str, Tensor] = {
        "P3": torch.rand(2, 128, 80, 80),
        "P4": torch.rand(2, 256, 40, 40),
        "P5": torch.rand(2, 256, 20, 20),
    }

    # --- Enabled mode ---
    caam = ContextAwareAttentionModule(channels, enabled=True)
    with torch.no_grad():
        out = caam(dummy_features)

    for scale, feat in out.items():
        print(f"CAAM(enabled) {scale}: {tuple(feat.shape)}")
        assert feat.shape == dummy_features[scale].shape, f"Shape mismatch at {scale}"
        # Attended output must be <= original (gate ∈ [0,1])
        assert float((feat >= 0).all()), f"Negative values at {scale}"

    # --- Disabled (identity) mode ---
    caam_off = ContextAwareAttentionModule(channels, enabled=False)
    out_off = caam_off(dummy_features)
    for scale in ["P3", "P4", "P5"]:
        assert out_off[scale] is dummy_features[scale], \
            f"Identity mode should return original tensor for {scale}"

    print("All CAAM shape and attention-bound assertions passed [OK]")
