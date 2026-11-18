"""
models/lcms_db.py
Full LCMS-DB model assembly.

Pipeline:
    Input (B, 3, H, W)
        |
        ▼
    LightweightBackbone  ->  {'P3', 'P4', 'P5'}
        |
        ▼
    MultiScaleFeatureEnhancement (MSFE)  [ablatable]
        |
        ▼
    ContextAwareAttentionModule (CAAM)   [ablatable]
        |
        ▼
    BidirectionalFeatureFusion (BFF)     [ablatable]
        |
        ▼
    DetectionHead  ->  list[Tensor]  (raw logits per scale)

Each custom module (MSFE, CAAM, BFF) can be independently disabled via
the config dict for clean ablation experiments (Section 6.2).

Config dict structure expected by LCMSDB.__init__:
    cfg['model']:
        num_classes  : int
        base_channels: int
        use_msfe     : bool
        use_caam     : bool
        use_bff      : bool
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from .backbone import LightweightBackbone
    from .msfe import MultiScaleFeatureEnhancement
    from .caam import ContextAwareAttentionModule
    from .bff import BidirectionalFeatureFusion
    from .head import DetectionHead
except (ImportError, ValueError):
    from models.backbone import LightweightBackbone
    from models.msfe import MultiScaleFeatureEnhancement
    from models.caam import ContextAwareAttentionModule
    from models.bff import BidirectionalFeatureFusion
    from models.head import DetectionHead


class LightweightTransformerBlock(nn.Module):
    """Lightweight single-layer Transformer encoder block for P5 global scene context."""
    def __init__(self, channels: int, num_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn  = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        norm_flat = self.norm1(flat)
        attn_out, _ = self.attn(norm_flat, norm_flat, norm_flat)
        flat = flat + attn_out
        flat = flat + self.mlp(self.norm2(flat))
        return flat.transpose(1, 2).view(B, C, H, W)


class LCMSDB(nn.Module):
    """LCMS-DB: Lightweight Cascade Multi-Scale Detection with Bi-directional fusion.

    Assembles the full detection pipeline from backbone to prediction head.
    Every custom module is individually ablatable via boolean flags in the
    config dict. Supports v1 (P3-P5) and v2 (P2-P5, CBAM/ECA, Transformer).
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()

        model_cfg: Dict[str, Any] = cfg["model"]

        self.num_classes: int     = model_cfg["num_classes"]
        base_channels: int        = model_cfg.get("base_channels", 32)
        use_msfe: bool            = model_cfg.get("use_msfe", True)
        use_caam: bool            = model_cfg.get("use_caam", True)
        use_bff: bool             = model_cfg.get("use_bff", True)
        self.use_p2: bool         = model_cfg.get("use_p2", False)
        attention_type: str       = model_cfg.get("attention_type", "caam")
        self.use_transformer: bool = model_cfg.get("use_transformer", False)

        self.strides: List[int] = [4, 8, 16, 32] if self.use_p2 else [8, 16, 32]

        # ------------------------------------------------------------------
        # 1. Backbone
        # ------------------------------------------------------------------
        self.backbone = LightweightBackbone(base_channels=base_channels, use_p2=self.use_p2)
        channels: Dict[str, int] = self.backbone.out_channels

        # ------------------------------------------------------------------
        # 2. Multi-Scale Feature Enhancement (MSFE)
        # ------------------------------------------------------------------
        self.msfe = MultiScaleFeatureEnhancement(channels, enabled=use_msfe)

        # ------------------------------------------------------------------
        # 3. Context-Aware Attention Module (CAAM / CBAM / ECA)
        # ------------------------------------------------------------------
        self.caam = ContextAwareAttentionModule(
            channels, enabled=use_caam, attention_type=attention_type
        )

        # Optional Transformer block at P5
        self.p5_transformer = (
            LightweightTransformerBlock(channels["P5"])
            if self.use_transformer
            else None
        )

        # ------------------------------------------------------------------
        # 4. Bi-directional Feature Fusion (BFF)
        # ------------------------------------------------------------------
        self.bff = BidirectionalFeatureFusion(channels, enabled=use_bff)

        # ------------------------------------------------------------------
        # 5. Detection Head
        # ------------------------------------------------------------------
        self.head = DetectionHead(
            channels=self.bff.out_channels,
            num_classes=self.num_classes,
            strides=self.strides,
            center_decode=model_cfg.get("center_decode", "sigmoid"),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> List[Tensor]:
        """Run the full LCMS-DB pipeline.

        Args:
            x: Input image batch of shape (B, 3, H, W).

        Returns:
            List of raw prediction tensors (one per scale) each of shape
            (B, Hi*Wi, 4 + 1 + num_classes).
        """
        # Step 1: multi-scale feature extraction
        features: Dict[str, Tensor] = self.backbone(x)

        # Step 2: per-scale residual refinement
        features = self.msfe(features)

        # Step 3: context-aware attention gating
        features = self.caam(features)

        # Step 3.5: optional transformer context at P5
        if self.p5_transformer is not None:
            features["P5"] = self.p5_transformer(features["P5"])

        # Step 4: bi-directional feature fusion
        features = self.bff(features)

        # Step 5: anchor-free detection predictions
        predictions: List[Tensor] = self.head(features)

        return predictions

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_info(self) -> Dict[str, int]:
        """Return parameter count statistics.

        Returns:
            Dict with keys:
              ``total_params``     – total parameter count.
              ``trainable_params`` – count of parameters with requires_grad=True.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_params": total, "trainable_params": trainable}


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def build_model(cfg: Dict[str, Any]) -> LCMSDB:
    """Instantiate LCMSDB from a config dict.

    Args:
        cfg: Full YAML config dictionary.

    Returns:
        Initialised (untrained) LCMSDB model.
    """
    return LCMSDB(cfg)


def load_checkpoint(
    model: LCMSDB,
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[LCMSDB, Dict[str, Any]]:
    """Load a saved checkpoint into an existing LCMSDB model.

    The checkpoint file is expected to be a dict with at least:
        ``state_dict`` : model weights (from ``model.state_dict()``)
        ``epoch``      : int
        ``best_map``   : float
        ``config``     : the original config dict used during training

    Args:
        model:           An already-instantiated LCMSDB model.
        checkpoint_path: Absolute or relative path to the ``.pt`` file.
        device:          Target device string (e.g. ``'cpu'``, ``'cuda:0'``).

    Returns:
        Tuple of (model_with_loaded_weights, checkpoint_meta_dict) where
        checkpoint_meta_dict contains everything in the checkpoint except
        the state_dict itself.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt: Dict[str, Any] = torch.load(checkpoint_path, map_location=device)

    # Load weights — handle either 'model_state', 'state_dict', or raw state_dict
    state_dict = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_checkpoint] Missing keys: {missing}")
    if unexpected:
        print(f"[load_checkpoint] Unexpected keys: {unexpected}")

    model.to(device)

    # Build meta dict: everything the caller might need except weights
    meta: Dict[str, Any] = {
        "epoch":    ckpt.get("epoch", -1),
        "best_map": ckpt.get("best_map") or ckpt.get("metrics", {}).get("val_map50", 0.0),
        "config":   ckpt.get("config", {}),
        "metrics":  ckpt.get("metrics", {}),
    }
    return model, meta


# ---------------------------------------------------------------------------
# Integration self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    # ------------------------------------------------------------------
    # Minimal config with all modules enabled
    # ------------------------------------------------------------------
    cfg = {
        "model": {
            "num_classes":   2,
            "base_channels": 32,
            "use_msfe":      True,
            "use_caam":      True,
            "use_bff":       True,
        }
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}\n")

    # Build model
    model = build_model(cfg).to(device)
    model.eval()

    # Dummy batch: 2 images, 640×640 resolution
    dummy_input = torch.randn(2, 3, 640, 640, device=device)

    # Forward pass with timing
    t0 = time.perf_counter()
    with torch.no_grad():
        preds = model(dummy_input)
    elapsed = time.perf_counter() - t0

    print("=== Forward pass outputs ===")
    expected_anchors = [80 * 80, 40 * 40, 20 * 20]
    num_outputs = 4 + 1 + cfg["model"]["num_classes"]

    for i, (pred, na) in enumerate(zip(preds, expected_anchors)):
        scale = f"P{3 + i}"
        print(f"  {scale}: {tuple(pred.shape)}")
        assert pred.shape == (2, na, num_outputs), \
            f"Shape mismatch at {scale}: {pred.shape}"

    print(f"\nForward pass time: {elapsed * 1000:.1f} ms")

    # Parameter count
    info = model.get_info()
    total_m = info["total_params"] / 1e6
    train_m = info["trainable_params"] / 1e6
    print(f"\n=== Model info ===")
    print(f"  Total params:     {total_m:.2f} M")
    print(f"  Trainable params: {train_m:.2f} M")

    # ------------------------------------------------------------------
    # Quick ablation smoke-test: disable each module in turn
    # ------------------------------------------------------------------
    for flag_key in ["use_msfe", "use_caam", "use_bff"]:
        ablation_cfg = {
            "model": {
                "num_classes":   2,
                "base_channels": 32,
                "use_msfe":      flag_key != "use_msfe",
                "use_caam":      flag_key != "use_caam",
                "use_bff":       flag_key != "use_bff",
            }
        }
        m = build_model(ablation_cfg).to(device).eval()
        with torch.no_grad():
            p = m(dummy_input)
        assert all(pred.shape == (2, na, num_outputs)
                   for pred, na in zip(p, expected_anchors))
        disabled = flag_key.replace("use_", "").upper()
        print(f"  Ablation ({disabled} off): shapes OK [OK]")

    print("\nAll integration assertions passed [OK]")
