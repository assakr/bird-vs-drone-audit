"""
losses/__init__.py
Public API for the LCMS-DB losses package.
"""

from .yolo_loss import YOLOLoss  # noqa: F401

__all__ = ["YOLOLoss"]
