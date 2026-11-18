"""
data/augment.py
===============
Training-only data augmentation pipeline.

Each augmentation is independently toggleable via the ``augmentation`` sub-dict
of the YAML config, enabling clean ablations. Val/test DataLoaders must NOT
pass images through this pipeline.

Label format throughout: numpy array of shape (N, 5)
    [class_id, x_center, y_center, width, height]   (all coordinates normalized 0-1)
"""

from __future__ import annotations

import math
import random
from typing import Optional, Tuple

import cv2
import numpy as np


# -- Utility helpers ------------------------------------------------------------

def _clip_boxes(labels: np.ndarray, min_wh: float = 1e-3) -> np.ndarray:
    """
    Clip normalised boxes to [0, 1] and drop degenerate ones.

    Args:
        labels:  (N, 5) array [cls, xc, yc, w, h] normalised.
        min_wh:  Minimum box width/height to keep (normalised units).

    Returns:
        Filtered (M, 5) array, M <= N.
    """
    if labels.shape[0] == 0:
        return labels
    labels = labels.copy()
    labels[:, 1:] = labels[:, 1:].clip(0.0, 1.0)
    keep = (labels[:, 3] >= min_wh) & (labels[:, 4] >= min_wh)
    return labels[keep]


def _xywh_to_xyxy_norm(labels: np.ndarray) -> np.ndarray:
    """Convert normalised [cls, xc, yc, w, h] -> [cls, x1, y1, x2, y2]."""
    out = labels.copy()
    out[:, 1] = labels[:, 1] - labels[:, 3] / 2   # x1
    out[:, 2] = labels[:, 2] - labels[:, 4] / 2   # y1
    out[:, 3] = labels[:, 1] + labels[:, 3] / 2   # x2
    out[:, 4] = labels[:, 2] + labels[:, 4] / 2   # y2
    return out


def _xyxy_to_xywh_norm(labels: np.ndarray) -> np.ndarray:
    """Convert normalised [cls, x1, y1, x2, y2] -> [cls, xc, yc, w, h]."""
    out = labels.copy()
    out[:, 1] = (labels[:, 1] + labels[:, 3]) / 2   # xc
    out[:, 2] = (labels[:, 2] + labels[:, 4]) / 2   # yc
    out[:, 3] = labels[:, 3] - labels[:, 1]          # w
    out[:, 4] = labels[:, 4] - labels[:, 2]          # h
    return out


# -- Individual augmentation classes -------------------------------------------

class RandomHorizontalFlip:
    """
    Flip image horizontally with probability `prob`.

    Updates x_center = 1 - x_center for all boxes.
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(
        self, image: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.prob:
            image = image[:, ::-1].copy()
            if labels.shape[0] > 0:
                labels = labels.copy()
                labels[:, 1] = 1.0 - labels[:, 1]   # flip x_center
        return image, labels


class RandomScale:
    """
    Scale image by a random factor in ``scale_range``, padding/cropping to
    maintain the original spatial size.
    """

    def __init__(self, scale_range: Tuple[float, float] = (0.5, 1.5), prob: float = 0.5):
        self.scale_range = scale_range
        self.prob = prob

    def __call__(
        self, image: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() >= self.prob:
            return image, labels

        h, w = image.shape[:2]
        scale = random.uniform(*self.scale_range)
        new_h, new_w = int(h * scale), int(w * scale)

        image_scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Paste onto a blank canvas of original size
        canvas = np.full((h, w, 3), 114, dtype=np.uint8)
        x_off = (w - new_w) // 2
        y_off = (h - new_h) // 2

        # Source/dest crop windows
        src_x1 = max(0, -x_off);  src_y1 = max(0, -y_off)
        dst_x1 = max(0,  x_off);  dst_y1 = max(0,  y_off)
        copy_w = min(new_w - src_x1, w - dst_x1)
        copy_h = min(new_h - src_y1, h - dst_y1)

        if copy_w > 0 and copy_h > 0:
            canvas[dst_y1:dst_y1+copy_h, dst_x1:dst_x1+copy_w] = \
                image_scaled[src_y1:src_y1+copy_h, src_x1:src_x1+copy_w]

        # Adjust labels: scale then offset
        if labels.shape[0] > 0:
            labs = _xywh_to_xyxy_norm(labels)   # [cls, x1,y1,x2,y2] norm
            # To pixel
            labs[:, 1] = labs[:, 1] * new_w + x_off
            labs[:, 2] = labs[:, 2] * new_h + y_off
            labs[:, 3] = labs[:, 3] * new_w + x_off
            labs[:, 4] = labs[:, 4] * new_h + y_off
            # Back to norm
            labs[:, 1] /= w;  labs[:, 2] /= h
            labs[:, 3] /= w;  labs[:, 4] /= h
            labs[:, 1:] = labs[:, 1:].clip(0.0, 1.0)
            labels = _xyxy_to_xywh_norm(labs)
            labels = _clip_boxes(labels)

        return canvas, labels


class RandomTranslation:
    """
    Translate image by up to ``translate_frac`` of its dimensions.
    Boxes outside the canvas after translation are dropped.
    """

    def __init__(self, translate_frac: float = 0.1, prob: float = 0.5):
        self.translate_frac = translate_frac
        self.prob = prob

    def __call__(
        self, image: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() >= self.prob:
            return image, labels

        h, w = image.shape[:2]
        tx = int(random.uniform(-self.translate_frac, self.translate_frac) * w)
        ty = int(random.uniform(-self.translate_frac, self.translate_frac) * h)

        M = np.float32([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))

        if labels.shape[0] > 0:
            labs = _xywh_to_xyxy_norm(labels)
            labs[:, 1] = labs[:, 1] + tx / w
            labs[:, 2] = labs[:, 2] + ty / h
            labs[:, 3] = labs[:, 3] + tx / w
            labs[:, 4] = labs[:, 4] + ty / h
            labs[:, 1:] = labs[:, 1:].clip(0.0, 1.0)
            labels = _xyxy_to_xywh_norm(labs)
            labels = _clip_boxes(labels)

        return image, labels


class RandomCrop:
    """
    Random crop: keep at least ``crop_frac`` of the image area.
    Boxes that lose more than 75 % of their area after cropping are dropped.
    """

    def __init__(self, crop_frac: float = 0.9, prob: float = 0.5):
        self.crop_frac = crop_frac
        self.prob = prob

    def __call__(
        self, image: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() >= self.prob:
            return image, labels

        h, w = image.shape[:2]
        crop_h = int(h * random.uniform(self.crop_frac, 1.0))
        crop_w = int(w * random.uniform(self.crop_frac, 1.0))
        y_off  = random.randint(0, h - crop_h)
        x_off  = random.randint(0, w - crop_w)

        image = image[y_off:y_off+crop_h, x_off:x_off+crop_w]
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        if labels.shape[0] > 0:
            labs = _xywh_to_xyxy_norm(labels)
            # Pixel coords in original image
            labs_px = labs.copy()
            labs_px[:, 1] *= w;  labs_px[:, 3] *= w
            labs_px[:, 2] *= h;  labs_px[:, 4] *= h
            # Compute intersection with crop window
            inter_x1 = np.maximum(labs_px[:, 1], x_off)
            inter_y1 = np.maximum(labs_px[:, 2], y_off)
            inter_x2 = np.minimum(labs_px[:, 3], x_off + crop_w)
            inter_y2 = np.minimum(labs_px[:, 4], y_off + crop_h)
            inter_area  = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
            orig_area   = (labs_px[:, 3] - labs_px[:, 1]) * (labs_px[:, 4] - labs_px[:, 2])
            keep = inter_area / (orig_area + 1e-9) >= 0.25
            if not keep.any():
                return image, np.zeros((0, 5), dtype=np.float32)

            labs_px = labs_px[keep]
            labs_px[:, 1] = (np.maximum(labs_px[:, 1], x_off) - x_off) / crop_w
            labs_px[:, 2] = (np.maximum(labs_px[:, 2], y_off) - y_off) / crop_h
            labs_px[:, 3] = (np.minimum(labs_px[:, 3], x_off + crop_w) - x_off) / crop_w
            labs_px[:, 4] = (np.minimum(labs_px[:, 4], y_off + crop_h) - y_off) / crop_h
            labels = _xyxy_to_xywh_norm(labs_px)
            labels = _clip_boxes(labels)

        return image, labels


class BrightnessContrastJitter:
    """
    Randomly adjust brightness and contrast of the image.
    Labels are unchanged.
    """

    def __init__(
        self,
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        prob: float = 0.5,
    ):
        self.brightness_limit = brightness_limit
        self.contrast_limit   = contrast_limit
        self.prob = prob

    def __call__(
        self, image: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() >= self.prob:
            return image, labels

        alpha = 1.0 + random.uniform(-self.contrast_limit,   self.contrast_limit)
        beta  = random.uniform(-self.brightness_limit, self.brightness_limit) * 255

        image = image.astype(np.float32)
        image = image * alpha + beta
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image, labels


class MosaicAugmentation:
    """
    Combine 4 images (current + 3 random samples) into a 2×2 mosaic.

    This is one of the most effective augmentations for small-object detection:
    it synthesises varied contexts, scales, and densities in a single training image.

    Args:
        dataset:   Reference to the BirDroneDataset (used to sample extra images).
        img_size:  Size of each quadrant (total mosaic = 2*img_size × 2*img_size,
                   then centre-cropped back to img_size × img_size).
        prob:      Probability of applying the augmentation.
    """

    def __init__(self, dataset, img_size: int = 640, prob: float = 0.5):
        self.dataset  = dataset
        self.img_size = img_size
        self.prob     = prob

    def __call__(
        self, image: np.ndarray, labels: np.ndarray, idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() >= self.prob:
            return image, labels

        s = self.img_size
        # Centre of the 2s × 2s canvas
        cx = int(random.uniform(s * 0.5, s * 1.5))
        cy = int(random.uniform(s * 0.5, s * 1.5))

        # Sample 3 additional images (excluding self)
        n = len(self.dataset)
        extra_idxs = random.choices([i for i in range(n) if i != idx], k=3)

        canvas  = np.full((2 * s, 2 * s, 3), 114, dtype=np.uint8)
        all_labels: list[np.ndarray] = []

        images_and_labels = [(image, labels)] + [
            self.dataset._load_raw(i) for i in extra_idxs
        ]

        positions = [
            (0, 0, cx, cy),           # top-left   (quadrant 0)
            (cx, 0, 2*s, cy),         # top-right  (quadrant 1)
            (0, cy, cx, 2*s),         # bot-left   (quadrant 2)
            (cx, cy, 2*s, 2*s),       # bot-right  (quadrant 3)
        ]

        for (img_i, lbl_i), (x1c, y1c, x2c, y2c) in zip(images_and_labels, positions):
            h_i, w_i = img_i.shape[:2]
            qw = x2c - x1c
            qh = y2c - y1c
            img_r = cv2.resize(img_i, (qw, qh), interpolation=cv2.INTER_LINEAR)
            canvas[y1c:y2c, x1c:x2c] = img_r

            if lbl_i.shape[0] > 0:
                lbl_adj = lbl_i.copy()
                lbl_adj[:, 1] = lbl_i[:, 1] * qw / (2 * s) + x1c / (2 * s)
                lbl_adj[:, 2] = lbl_i[:, 2] * qh / (2 * s) + y1c / (2 * s)
                lbl_adj[:, 3] = lbl_i[:, 3] * qw / (2 * s)
                lbl_adj[:, 4] = lbl_i[:, 4] * qh / (2 * s)
                all_labels.append(lbl_adj)

        # Centre-crop back to s × s
        x1 = cx - s // 2;  y1 = cy - s // 2
        x2 = x1 + s;       y2 = y1 + s
        canvas = canvas[y1:y2, x1:x2]

        # Adjust label coordinates to the cropped canvas
        final_labels: list[np.ndarray] = []
        for lbl in all_labels:
            if lbl.shape[0] == 0:
                continue
            lbl_c = lbl.copy()
            lbl_c[:, 1] = lbl[:, 1] - x1 / (2 * s)
            lbl_c[:, 2] = lbl[:, 2] - y1 / (2 * s)
            # Re-normalise to the cropped canvas (s/2s = 0.5 factor)
            lbl_c[:, 1:] *= 2.0
            lbl_c = _clip_boxes(lbl_c)
            if lbl_c.shape[0] > 0:
                final_labels.append(lbl_c)

        out_labels = (
            np.concatenate(final_labels, axis=0)
            if final_labels
            else np.zeros((0, 5), dtype=np.float32)
        )
        return canvas, out_labels


class MixUp:
    """
    Blend the current image with a randomly sampled image.

    The mix weight λ ~ Beta(alpha, alpha). Both label sets are preserved
    (concatenated), which is the standard object-detection MixUp variant.

    Args:
        dataset:  Reference to BirDroneDataset for sampling.
        prob:     Application probability.
        alpha:    Beta distribution parameter (lower = more extreme mixes).
    """

    def __init__(self, dataset, prob: float = 0.15, alpha: float = 0.2):
        self.dataset = dataset
        self.prob    = prob
        self.alpha   = alpha

    def __call__(
        self, image: np.ndarray, labels: np.ndarray, idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() >= self.prob:
            return image, labels

        n = len(self.dataset)
        j = random.choice([i for i in range(n) if i != idx])
        img2, lbl2 = self.dataset._load_raw(j)

        # Resize partner to match current image size
        h, w = image.shape[:2]
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_LINEAR)

        lam  = np.random.beta(self.alpha, self.alpha)
        mixed = (image.astype(np.float32) * lam +
                 img2.astype(np.float32)  * (1 - lam)).astype(np.uint8)

        combined = (
            np.concatenate([labels, lbl2], axis=0)
            if (labels.shape[0] > 0 and lbl2.shape[0] > 0)
            else (labels if labels.shape[0] > 0 else lbl2)
        )
        return mixed, combined


class CopyPasteAugmentation:
    """
    Extract small drone/bird instances from random dataset images and paste them
    onto the current frame at random locations and scales.
    Proven to boost small-object detection accuracy significantly.
    """
    def __init__(self, dataset, prob: float = 0.3, max_paste: int = 3):
        self.dataset = dataset
        self.prob = prob
        self.max_paste = max_paste

    def __call__(self, image: np.ndarray, labels: np.ndarray, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() >= self.prob or self.dataset is None or len(self.dataset) == 0:
            return image, labels

        h, w = image.shape[:2]
        # Sample an extra image
        other_idx = random.randint(0, len(self.dataset) - 1)
        other_img, other_lbls = self.dataset._load_raw(other_idx)
        if other_lbls.shape[0] == 0:
            return image, labels

        oh, ow = other_img.shape[:2]
        pasted_boxes = []
        num_pastes = random.randint(1, min(self.max_paste, other_lbls.shape[0]))
        chosen_indices = random.sample(range(other_lbls.shape[0]), num_pastes)

        for ci in chosen_indices:
            cls_id, ocx, ocy, obw, obh = other_lbls[ci]
            ox1 = int(max(0, (ocx - obw / 2.0) * ow))
            oy1 = int(max(0, (ocy - obh / 2.0) * oh))
            ox2 = int(min(ow, (ocx + obw / 2.0) * ow))
            oy2 = int(min(oh, (ocy + obh / 2.0) * oh))

            crop = other_img[oy1:oy2, ox1:ox2]
            ch, cw = crop.shape[:2]
            if ch < 4 or cw < 4:
                continue

            scale = random.uniform(0.6, 1.4)
            nw, nh = max(4, int(cw * scale)), max(4, int(ch * scale))
            if nw >= w or nh >= h:
                continue
            crop_resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)

            dx1 = random.randint(0, w - nw)
            dy1 = random.randint(0, h - nh)
            dx2 = dx1 + nw
            dy2 = dy1 + nh

            image[dy1:dy2, dx1:dx2] = crop_resized

            ncx = (dx1 + nw / 2.0) / w
            ncy = (dy1 + nh / 2.0) / h
            nbw = nw / w
            nbh = nh / h
            pasted_boxes.append([cls_id, ncx, ncy, nbw, nbh])

        if pasted_boxes:
            new_lbls = np.array(pasted_boxes, dtype=np.float32)
            labels = np.concatenate([labels, new_lbls], axis=0) if labels.shape[0] > 0 else new_lbls

        return image, labels


class TileSliceAugmentation:
    """
    SAHI-style tiling augmentation: crops a sub-region centered on small objects
    and resizes back to input dimensions to provide multi-scale zoom resolution.
    """
    def __init__(self, prob: float = 0.25, tile_size: int = 400):
        self.prob = prob
        self.tile_size = tile_size

    def __call__(self, image: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() >= self.prob or labels.shape[0] == 0:
            return image, labels

        h, w = image.shape[:2]
        ts = min(self.tile_size, min(h, w))

        chosen_box = random.choice(labels)
        _, cx, cy, bw, bh = chosen_box
        cx_px = int(cx * w)
        cy_px = int(cy * h)

        x1 = max(0, min(w - ts, cx_px - ts // 2))
        y1 = max(0, min(h - ts, cy_px - ts // 2))
        x2 = x1 + ts
        y2 = y1 + ts

        cropped = image[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        kept_boxes = []
        for box in labels:
            cls_id, bcx, bcy, bbw, bbh = box
            bx1 = (bcx - bbw / 2.0) * w
            by1 = (bcy - bbh / 2.0) * h
            bx2 = (bcx + bbw / 2.0) * w
            by2 = (bcy + bbh / 2.0) * h

            ix1 = max(x1, bx1)
            iy1 = max(y1, by1)
            ix2 = min(x2, bx2)
            iy2 = min(y2, by2)

            if ix2 > ix1 and iy2 > iy1:
                orig_area = max(1.0, (bx2 - bx1) * (by2 - by1))
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                if inter_area / orig_area >= 0.3:
                    nx = ((ix1 + ix2) / 2.0 - x1) / ts
                    ny = ((iy1 + iy2) / 2.0 - y1) / ts
                    nw = (ix2 - ix1) / ts
                    nh = (iy2 - iy1) / ts
                    kept_boxes.append([cls_id, nx, ny, nw, nh])

        if kept_boxes:
            return resized, np.array(kept_boxes, dtype=np.float32)
        return image, labels


# -- Pipeline -------------------------------------------------------------------

class AugmentationPipeline:
    """
    Composable augmentation pipeline for training.

    Each augmentation is instantiated only when enabled in the config.
    The ordering follows best-practice YOLO conventions:
        flip -> scale -> translate -> crop -> brightness/contrast -> mosaic -> mixup

    Args:
        cfg:     The ``augmentation`` sub-dict from the YAML config.
        dataset: Reference to BirDroneDataset (required for Mosaic and MixUp).
    """

    def __init__(self, cfg: dict, dataset=None):
        self.cfg     = cfg
        self.dataset = dataset
        self._build_transforms()

    def _build_transforms(self):
        c = self.cfg

        self.flip = (
            RandomHorizontalFlip(prob=0.5)
            if c.get('horizontal_flip', True)
            else None
        )

        scale_cfg = c.get('random_scale', {})
        self.scale = (
            RandomScale(
                scale_range=scale_cfg.get('scale_range', [0.5, 1.5]),
                prob=0.5,
            )
            if scale_cfg.get('enabled', True)
            else None
        )

        trans_cfg = c.get('translation', {})
        self.translate = (
            RandomTranslation(
                translate_frac=trans_cfg.get('translate_frac', 0.1),
                prob=0.5,
            )
            if trans_cfg.get('enabled', True)
            else None
        )

        crop_cfg = c.get('random_crop', {})
        self.crop = (
            RandomCrop(
                crop_frac=crop_cfg.get('crop_frac', 0.9),
                prob=0.5,
            )
            if crop_cfg.get('enabled', True)
            else None
        )

        bc_cfg = c.get('brightness_contrast', {})
        self.bc = (
            BrightnessContrastJitter(
                brightness_limit=bc_cfg.get('brightness_limit', 0.2),
                contrast_limit=bc_cfg.get('contrast_limit', 0.2),
                prob=0.5,
            )
            if bc_cfg.get('enabled', True)
            else None
        )

        mosaic_cfg = c.get('mosaic', {})
        self.mosaic = (
            MosaicAugmentation(
                dataset=self.dataset,
                prob=mosaic_cfg.get('prob', 0.5),
            )
            if mosaic_cfg.get('enabled', True) and self.dataset is not None
            else None
        )

        mixup_cfg = c.get('mixup', {})
        self.mixup = (
            MixUp(
                dataset=self.dataset,
                prob=mixup_cfg.get('prob', 0.15),
                alpha=mixup_cfg.get('alpha', 0.2),
            )
            if mixup_cfg.get('enabled', True) and self.dataset is not None
            else None
        )

        cp_cfg = c.get('copy_paste', {})
        self.copy_paste = (
            CopyPasteAugmentation(
                dataset=self.dataset,
                prob=cp_cfg.get('prob', 0.3),
                max_paste=cp_cfg.get('max_paste', 3),
            )
            if cp_cfg.get('enabled', False) and self.dataset is not None
            else None
        )

        tile_cfg = c.get('tile_slice', {})
        self.tile_slice = (
            TileSliceAugmentation(
                prob=tile_cfg.get('prob', 0.25),
                tile_size=tile_cfg.get('tile_size', 400),
            )
            if tile_cfg.get('enabled', False)
            else None
        )

    def __call__(
        self, image: np.ndarray, labels: np.ndarray, idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply enabled augmentations in sequence.

        Args:
            image:   HWC BGR uint8 image.
            labels:  (N, 5) float32 [cls, xc, yc, w, h] normalised.
            idx:     Dataset index of this sample (used by Mosaic / MixUp).

        Returns:
            Augmented (image, labels) pair.
        """
        if self.flip       is not None: image, labels = self.flip(image, labels)
        if self.scale      is not None: image, labels = self.scale(image, labels)
        if self.translate  is not None: image, labels = self.translate(image, labels)
        if self.crop       is not None: image, labels = self.crop(image, labels)
        if self.bc         is not None: image, labels = self.bc(image, labels)
        if self.copy_paste is not None: image, labels = self.copy_paste(image, labels, idx)
        if self.tile_slice is not None: image, labels = self.tile_slice(image, labels)
        if self.mosaic     is not None: image, labels = self.mosaic(image, labels, idx)
        if self.mixup      is not None: image, labels = self.mixup(image, labels, idx)
        return image, labels
