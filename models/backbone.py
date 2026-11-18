"""
models/backbone.py
Lightweight YOLO backbone using depthwise separable convolutions and CSP blocks.
Outputs three feature maps: P3, P4, P5 at strides /8, /16, /32 respectively.

Architecture summary (base_channels c = 32):
  Stem   : Conv(3->c, k=6, s=2) -> Conv(c->2c, k=3, s=2)   [/4 of input]
  Stage-1: CSP(2c->2c) -> Conv(2c->4c, s=2)  -> CSP(4c->4c)  -> P3 [/8]
  Stage-2: Conv(4c->8c, s=2) -> CSP(8c->8c)                  -> P4 [/16]
  Stage-3: Conv(8c->8c, s=2) -> CSP(8c->8c) -> SPPF(8c->8c)  -> P5 [/32]
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Primitive building blocks
# ---------------------------------------------------------------------------


class ConvBnAct(nn.Module):
    """Standard Conv2d -> BatchNorm2d -> Activation layer.

    This is the workhorse primitive used throughout the backbone.

    Args:
        in_ch:  Number of input channels.
        out_ch: Number of output channels.
        k:      Kernel size (default 1).
        s:      Stride (default 1).
        p:      Padding. If None, auto-computed as k//2 (same-padding for odd k).
        g:      Groups for grouped/depthwise convolutions (default 1).
        act:    If True, append SiLU activation; if False, no activation.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        # Auto-pad: ensures spatial size is preserved for odd kernels
        if p is None:
            padding = 2 if (k == 6 and s == 2) else k // 2
        else:
            padding = p
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, padding, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        return self.act(self.bn(self.conv(x)))


class DepthwiseSepConv(nn.Module):
    """Depthwise separable convolution: depthwise 3×3 + pointwise 1×1.

    Reduces parameters and FLOPs vs. a standard Conv2d while keeping
    the same receptive field.

    Args:
        in_ch:  Input channels.
        out_ch: Output channels.
        k:      Depthwise kernel size (default 3).
        s:      Stride applied to the depthwise step (default 1).
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1) -> None:
        super().__init__()
        # Depthwise: groups=in_ch -> one filter per input channel
        self.dw = ConvBnAct(in_ch, in_ch, k=k, s=s, g=in_ch)
        # Pointwise: mix channels with 1×1 conv
        self.pw = ConvBnAct(in_ch, out_ch, k=1)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        return self.pw(self.dw(x))


class Bottleneck(nn.Module):
    """Lightweight bottleneck: 1×1 expand -> depthwise 3×3 -> 1×1 compress + residual.

    A residual (skip) connection is added when ``shortcut=True`` and the
    channel count does not change.

    Args:
        ch:       Input (and output) channel count.
        shortcut: Whether to add the residual skip connection.
        e:        Expansion ratio for the hidden dimension (default 0.5).
    """

    def __init__(self, ch: int, shortcut: bool = True, e: float = 0.5) -> None:
        super().__init__()
        hidden = int(ch * e)
        self.cv1 = ConvBnAct(ch, hidden, k=1)           # channel expand
        self.cv2 = DepthwiseSepConv(hidden, hidden, k=3) # spatial mixing
        self.cv3 = ConvBnAct(hidden, ch, k=1)            # channel compress
        self.add = shortcut  # residual flag

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        out = self.cv3(self.cv2(self.cv1(x)))
        # Residual skip: only when channel dims match (always true here)
        return x + out if self.add else out


class CSPBlock(nn.Module):
    """Cross Stage Partial block.

    Splits the input into two halves:
      - One half passes through N Bottleneck layers.
      - The other is an identity skip.
    Both are concatenated and projected back to ``out_ch``.

    Args:
        in_ch:    Input channels.
        out_ch:   Output channels.
        n:        Number of stacked Bottleneck layers.
        shortcut: Whether Bottlenecks use residual connections.
        e:        Expansion ratio inside each Bottleneck.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n: int = 1,
        shortcut: bool = True,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        hidden = out_ch // 2  # each split gets half the output channels

        # Project full input to hidden size before splitting
        self.cv1 = ConvBnAct(in_ch, hidden, k=1)
        self.cv2 = ConvBnAct(in_ch, hidden, k=1)

        # Stack of Bottlenecks on the "main" branch
        self.m = nn.Sequential(
            *[Bottleneck(hidden, shortcut=shortcut, e=e) for _ in range(n)]
        )

        # Final projection: concat(main, skip) -> out_ch
        self.cv3 = ConvBnAct(2 * hidden, out_ch, k=1)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        # Main branch: through bottlenecks
        y1 = self.m(self.cv1(x))
        # Skip branch: identity projection
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling – Fast (SPPF).

    Applies three sequential MaxPool2d(k=5) operations and concatenates
    all intermediate feature maps (original + 3 pooled) to capture
    multi-scale context efficiently.

    Args:
        in_ch:  Input channels.
        out_ch: Output channels.
        k:      MaxPool kernel size (default 5).
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 5) -> None:
        super().__init__()
        hidden = in_ch // 2
        self.cv1 = ConvBnAct(in_ch, hidden, k=1)        # channel squeeze
        # Three sequential max-pools; padding preserves spatial size
        self.mp = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        # Concat of [x, mp(x), mp²(x), mp³(x)] -> 4×hidden channels
        self.cv2 = ConvBnAct(4 * hidden, out_ch, k=1)   # channel expand

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        x = self.cv1(x)
        y1 = self.mp(x)
        y2 = self.mp(y1)
        y3 = self.mp(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


# ---------------------------------------------------------------------------
# Full backbone
# ---------------------------------------------------------------------------


class LightweightBackbone(nn.Module):
    """Full lightweight backbone producing multi-scale feature maps P3, P4, P5.

    Uses depthwise separable convolutions and CSP blocks to minimise
    parameter count while maintaining representational capacity.

    Output strides relative to the input:
      P3 -> stride /8  (fine-grained, for small objects)
      P4 -> stride /16 (medium objects)
      P5 -> stride /32 (coarse, for large objects) + SPPF enrichment

    Args:
        base_channels: Width multiplier ``c`` (default 32).
                       Channel counts scale as multiples of c.
    """

    def __init__(self, base_channels: int = 32, use_p2: bool = False) -> None:
        super().__init__()
        c = base_channels  # shorthand
        self.use_p2 = use_p2

        # ------------------------------------------------------------------
        # Stem: downsample /2 twice -> /4 total
        # ------------------------------------------------------------------
        self.stem = nn.Sequential(
            ConvBnAct(3, c, k=6, s=2),    # /2  : (B, c,   H/2,  W/2)
            ConvBnAct(c, 2 * c, k=3, s=2), # /2  : (B, 2c,  H/4,  W/4)
        )

        # ------------------------------------------------------------------
        # Stage 1: /8 -> produces P3 (and P2 at stride /4 before downsampling)
        # ------------------------------------------------------------------
        self.stage1_csp = CSPBlock(2 * c, 2 * c, n=1)          # (B, 2c, H/4, W/4) -> P2
        self.stage1_down = ConvBnAct(2 * c, 4 * c, k=3, s=2)   # (B, 4c, H/8, W/8)
        self.stage1_p3 = CSPBlock(4 * c, 4 * c, n=1)           # P3 output

        # ------------------------------------------------------------------
        # Stage 2: /16 -> produces P4
        # ------------------------------------------------------------------
        self.stage2_down = ConvBnAct(4 * c, 8 * c, k=3, s=2)   # (B, 8c, H/16, W/16)
        self.stage2_p4 = CSPBlock(8 * c, 8 * c, n=2)           # P4 output

        # ------------------------------------------------------------------
        # Stage 3: /32 -> produces P5 (with SPPF for global context)
        # ------------------------------------------------------------------
        self.stage3_down = ConvBnAct(8 * c, 8 * c, k=3, s=2)   # (B, 8c, H/32, W/32)
        self.stage3_csp = CSPBlock(8 * c, 8 * c, n=3)
        self.stage3_sppf = SPPF(8 * c, 8 * c, k=5)             # P5 output

        # ------------------------------------------------------------------
        # Channel registry — consumed by downstream modules
        # ------------------------------------------------------------------
        self.out_channels: Dict[str, int] = {}
        if self.use_p2:
            self.out_channels["P2"] = 2 * c  # 64 when c=32
        self.out_channels["P3"] = 4 * c      # 128 when c=32
        self.out_channels["P4"] = 8 * c      # 256 when c=32
        self.out_channels["P5"] = 8 * c      # 256 when c=32

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Run forward pass and return multi-scale feature maps.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Dictionary with keys 'P3', 'P4', 'P5' (and 'P2' if use_p2=True).
        """
        # Stem
        x = self.stem(x)                          # (B, 2c, H/4, W/4)

        # Stage 1 -> P2 and P3
        p2 = self.stage1_csp(x)                   # (B, 2c, H/4, W/4)
        x  = self.stage1_down(p2)                 # (B, 4c, H/8, W/8)
        p3 = self.stage1_p3(x)                    # (B, 4c, H/8, W/8)

        # Stage 2 -> P4
        x  = self.stage2_down(p3)                 # (B, 8c, H/16, W/16)
        p4 = self.stage2_p4(x)                    # (B, 8c, H/16, W/16)

        # Stage 3 -> P5
        x  = self.stage3_down(p4)                 # (B, 8c, H/32, W/32)
        x  = self.stage3_csp(x)
        p5 = self.stage3_sppf(x)                  # (B, 8c, H/32, W/32)

        if self.use_p2:
            return {"P2": p2, "P3": p3, "P4": p4, "P5": p5}
        return {"P3": p3, "P4": p4, "P5": p5}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    backbone = LightweightBackbone(base_channels=32)
    dummy = torch.zeros(1, 3, 640, 640)

    with torch.no_grad():
        out = backbone(dummy)

    for name, feat in out.items():
        print(f"{name}: {tuple(feat.shape)}")

    # Validate expected shapes for base_channels=32, input=640×640
    assert out["P3"].shape == (1, 128, 80, 80),  f"P3 mismatch: {out['P3'].shape}"
    assert out["P4"].shape == (1, 256, 40, 40),  f"P4 mismatch: {out['P4'].shape}"
    assert out["P5"].shape == (1, 256, 20, 20),  f"P5 mismatch: {out['P5'].shape}"

    print("All backbone shape assertions passed [OK]")
