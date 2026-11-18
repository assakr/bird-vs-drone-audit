"""
data/preprocess.py
==================
Preprocessing utilities shared across train / val / test sets.
No augmentation is applied here — only resize + normalize.
This function is the single source of truth for image pre-processing,
called identically from all three DataLoaders.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# -- Letterbox resize -----------------------------------------------------------

def letterbox(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    stride: int = 32,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image to target_size while preserving aspect ratio.
    The empty region is filled with a constant colour (gray by default).

    Args:
        image:       HWC BGR uint8 numpy array.
        target_size: (width, height) of the output image.
        stride:      Output dimensions are rounded to a multiple of this value.
        color:       Padding fill colour in BGR.

    Returns:
        padded_image: Resized and padded image (H_out, W_out, 3) uint8.
        scale:        Uniform scale factor applied to both axes.
        pad:          (pad_w, pad_h) — total padding in pixels on each axis.
                      Half is applied to each side (top/bottom or left/right).
    """
    h0, w0 = image.shape[:2]
    target_w, target_h = target_size

    # Scale so that the longer side fits within target
    scale = min(target_w / w0, target_h / h0)

    new_w = int(round(w0 * scale))
    new_h = int(round(h0 * scale))

    # Resize with bilinear interpolation
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute padding (make dims divisible by stride)
    pad_w = target_w - new_w
    pad_h = target_h - new_h

    # Distribute padding symmetrically
    top    = pad_h // 2
    bottom = pad_h - top
    left   = pad_w // 2
    right  = pad_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color,
    )

    return padded, scale, (pad_w, pad_h)


# -- Pixel normalisation --------------------------------------------------------

def normalize(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std:  Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Convert a BGR uint8 image to a normalised float32 CHW array.

    Steps:
        1. BGR -> RGB channel swap.
        2. Divide by 255 -> [0, 1].
        3. Subtract ImageNet mean, divide by ImageNet std.

    Args:
        image: HWC BGR uint8 array, values in [0, 255].
        mean:  Per-channel mean in RGB order.
        std:   Per-channel std in RGB order.

    Returns:
        CHW float32 array, normalised.
    """
    img = image[:, :, ::-1].astype(np.float32) / 255.0   # BGR->RGB, scale

    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr  = np.array(std,  dtype=np.float32).reshape(1, 1, 3)

    img = (img - mean_arr) / (std_arr + 1e-7)
    img = img.transpose(2, 0, 1)   # HWC -> CHW
    return np.ascontiguousarray(img, dtype=np.float32)


# -- Combined entry-point -------------------------------------------------------

def preprocess(
    image: np.ndarray,
    resolution: int = 640,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std:  Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Full preprocessing pipeline: letterbox -> normalise.

    This is the function imported and called by every DataLoader
    (train / val / test). Augmentation is NOT applied here.

    Args:
        image:      HWC BGR uint8 numpy array (original image).
        resolution: Target square size, e.g. 640 or 1280.
        mean:       ImageNet channel mean (RGB order).
        std:        ImageNet channel std (RGB order).

    Returns:
        processed:  CHW float32 array, normalised.
        scale:      Scale factor (original -> letterboxed).
        pad:        (pad_w, pad_h) total padding applied.
    """
    padded, scale, pad = letterbox(image, target_size=(resolution, resolution))
    processed = normalize(padded, mean=mean, std=std)
    return processed, scale, pad


# -- Inverse transform (for post-processing / visualisation) -------------------

def undo_letterbox(
    boxes_xyxy: np.ndarray,
    scale: float,
    pad: Tuple[int, int],
    orig_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Map predicted boxes from letterboxed space back to original image coords.

    Args:
        boxes_xyxy:  (N, 4) array of boxes in [x1, y1, x2, y2] pixel format,
                     relative to the letterboxed / padded image.
        scale:       Scale factor returned by ``letterbox``.
        pad:         (pad_w, pad_h) returned by ``letterbox``.
        orig_shape:  (H, W) of the original (un-padded) image.

    Returns:
        (N, 4) boxes in original image pixel coordinates, clipped to bounds.
    """
    boxes = boxes_xyxy.copy().astype(np.float32)

    pad_w, pad_h = pad
    # Remove symmetric padding offset (half on each side)
    boxes[:, [0, 2]] -= pad_w / 2   # x offset
    boxes[:, [1, 3]] -= pad_h / 2   # y offset

    # Undo scaling
    boxes /= scale

    # Clip to original image bounds
    orig_h, orig_w = orig_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)

    return boxes
