"""
data/dataset.py
===============
BirDrone PyTorch Dataset and DataLoader builder.

Dataset structure expected on disk:
    <root>/images/train/   (*.jpg / *.jpeg / *.png)
    <root>/images/val/
    <root>/images/test/
    <root>/labels/train/   (*. txt  — one per image, YOLO format)
    <root>/labels/val/
    <root>/labels/test/

Each label .txt has one row per object:
    class_id  x_center  y_center  width  height    (all normalised 0-1)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from data.augment import AugmentationPipeline
from data.preprocess import preprocess


# -- Dataset --------------------------------------------------------------------

class BirDroneDataset(Dataset):
    """
    PyTorch Dataset for the BirDrone bird/drone detection benchmark.

    Preprocessing (letterbox + normalisation) is always applied.
    Augmentation is applied only when ``augment=True`` (i.e., for the
    training split — val and test must stay unaugmented).

    Args:
        images_dir:    Directory containing image files.
        labels_dir:    Directory containing YOLO .txt annotation files.
        resolution:    Target square inference resolution (e.g. 640 or 1280).
        augment:       Whether to apply data augmentation.
        augment_cfg:   Augmentation sub-dict from YAML config (used when augment=True).
        preprocess_cfg: Optional dict with 'mean' and 'std' keys for normalisation.
    """

    IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        resolution: int = 640,
        augment: bool = False,
        augment_cfg: Optional[Dict] = None,
        preprocess_cfg: Optional[Dict] = None,
        prefix_classes: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.resolution = resolution
        self.augment    = augment
        self.prefix_classes = prefix_classes

        # Normalisation params
        pp = preprocess_cfg or {}
        self.mean = tuple(pp.get('mean', (0.485, 0.456, 0.406)))
        self.std  = tuple(pp.get('std',  (0.229, 0.224, 0.225)))

        # Build list of (img_path, label_path) pairs
        self.samples: List[Tuple[Path, Path]] = []
        self._scan_directory()

        # Augmentation pipeline (lazy: needs self to be ready for mosaic/mixup)
        self._aug_pipeline: Optional[AugmentationPipeline] = None
        self._augment_cfg = augment_cfg or {}
        if self.augment:
            self._aug_pipeline = AugmentationPipeline(self._augment_cfg, dataset=self)

    # -- Directory scan ----------------------------------------------------------

    def _scan_directory(self):
        """Pair every image file with its label file. Skip unpaired images."""
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        missing_labels = 0
        for img_path in sorted(self.images_dir.iterdir()):
            if img_path.suffix.lower() not in self.IMG_EXTENSIONS:
                continue
            label_path = self.labels_dir / (img_path.stem + '.txt')
            if not label_path.exists():
                missing_labels += 1
                continue
            self.samples.append((img_path, label_path))

        if not self.samples:
            raise RuntimeError(
                f"No valid image+label pairs found in {self.images_dir}. "
                "Check that images and labels directories are correctly set."
            )
        if missing_labels:
            print(
                f"[dataset] Warning: {missing_labels} images in "
                f"{self.images_dir} have no matching label file and were skipped."
            )

    # -- Raw load (used internally by Mosaic / MixUp augmentations) ------------

    def _load_raw(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a raw (un-preprocessed) image and its labels.

        Called by MosaicAugmentation and MixUp to sample partner images.

        Args:
            idx: Dataset index.

        Returns:
            (image_bgr_uint8, labels_float32)
        """
        img_path, lbl_path = self.samples[idx]
        image  = cv2.imread(str(img_path))
        if image is None:
            image = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        labels = self._load_labels(lbl_path, filename_stem=img_path.stem)
        return image, labels

    # -- Label loading -----------------------------------------------------------

    def _load_labels(self, label_path: Path, filename_stem: str = "") -> np.ndarray:
        """
        Load YOLO-format labels from a .txt file.
        Supports both 5-field bboxes [cls, cx, cy, w, h] and segmentation polygons
        [cls, x1, y1, x2, y2, ...] converted automatically to bounding boxes.
        If prefix_classes is True, class ID is inferred from filename prefix ('B'->0, 'D'->1).

        Returns:
            Float32 array of shape (N, 5): [class_id, xc, yc, w, h] normalised.
            Returns (0, 5) zero array for empty or missing files.
        """
        if not label_path.exists() or label_path.stat().st_size == 0:
            return np.zeros((0, 5), dtype=np.float32)

        inferred_class = None
        if self.prefix_classes and filename_stem:
            stem_upper = filename_stem.upper()
            if stem_upper.startswith('B'):
                inferred_class = 0
            elif stem_upper.startswith('D'):
                inferred_class = 1

        boxes = []
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(float(parts[0]))
                    except ValueError:
                        continue

                    if inferred_class is not None:
                        cls_id = inferred_class

                    try:
                        coords = [float(x) for x in parts[1:]]
                    except ValueError:
                        continue

                    if len(coords) == 4:
                        cx, cy, w, h = coords
                        boxes.append([cls_id, cx, cy, w, h])
                    elif len(coords) >= 6:
                        # Convert polygon coordinates to bounding box
                        xs = coords[0::2]
                        ys = coords[1::2]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        w = max(1e-4, min(1.0, x_max - x_min))
                        h = max(1e-4, min(1.0, y_max - y_min))
                        cx = max(0.0, min(1.0, x_min + w / 2.0))
                        cy = max(0.0, min(1.0, y_min + h / 2.0))
                        boxes.append([cls_id, cx, cy, w, h])

            if not boxes:
                return np.zeros((0, 5), dtype=np.float32)
            return np.array(boxes, dtype=np.float32)
        except Exception:
            return np.zeros((0, 5), dtype=np.float32)

    # -- Dataset protocol -------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load, preprocess (and optionally augment) one sample.

        Returns:
            dict with keys:
                'image':       (3, H, W) float32 tensor, normalised.
                'labels':      (N, 5) float32 tensor [cls, xc, yc, w, h].
                'img_path':    str — path to the source image.
                'orig_shape':  (H, W) original image shape before any transform.
        """
        img_path, lbl_path = self.samples[idx]

        # -- Load image --
        image = cv2.imread(str(img_path))
        if image is None:
            # Fallback: black image so training doesn't crash on a bad file
            image = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)

        orig_h, orig_w = image.shape[:2]
        labels = self._load_labels(lbl_path, filename_stem=img_path.stem)

        # -- Augmentation (train only) --
        if self.augment and self._aug_pipeline is not None:
            image, labels = self._aug_pipeline(image, labels, idx)

        # -- Preprocessing (always) --
        processed, _, _ = preprocess(
            image, resolution=self.resolution,
            mean=self.mean, std=self.std,
        )

        # -- Convert to tensors --
        image_tensor  = torch.from_numpy(processed)                # (3, H, W)
        labels_tensor = torch.from_numpy(
            labels if labels.shape[0] > 0
            else np.zeros((0, 5), dtype=np.float32)
        )

        return {
            'image':      image_tensor,
            'labels':     labels_tensor,
            'img_path':   str(img_path),
            'orig_shape': (orig_h, orig_w),
        }

    # -- Custom collate ----------------------------------------------------------

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function for variable-length label tensors.

        Labels from all images in the batch are concatenated into a single
        (total_boxes, 6) tensor where column 0 is the batch index. This is
        the standard YOLO label format for batch-wise loss computation.

        Args:
            batch: List of dicts returned by ``__getitem__``.

        Returns:
            dict:
                'images':      (B, 3, H, W) float32 tensor.
                'labels':      (total_boxes, 6) float32 [batch_idx, cls, xc, yc, w, h].
                'img_paths':   List[str] of source image paths.
                'orig_shapes': List[(H, W)] of original image shapes.
        """
        images      = torch.stack([s['image']  for s in batch], dim=0)
        img_paths   = [s['img_path']   for s in batch]
        orig_shapes = [s['orig_shape'] for s in batch]

        # Prepend batch index to each label row
        all_labels = []
        for i, s in enumerate(batch):
            lbl = s['labels']   # (N, 5)
            if lbl.shape[0] > 0:
                batch_col = torch.full((lbl.shape[0], 1), float(i))
                all_labels.append(torch.cat([batch_col, lbl], dim=1))   # (N, 6)

        labels = (
            torch.cat(all_labels, dim=0)
            if all_labels
            else torch.zeros((0, 6), dtype=torch.float32)
        )

        return {
            'images':      images,
            'labels':      labels,
            'img_paths':   img_paths,
            'orig_shapes': orig_shapes,
        }


# -- DataLoader builder ---------------------------------------------------------

def build_dataloaders(
    cfg: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from a YAML config dict.

    The data.yaml file pointed to by ``cfg['data']['yaml']`` defines the
    image directory roots and class names. Label directories are inferred
    by replacing ``images`` with ``labels`` in the path.

    Args:
        cfg: Full config dict (loaded by ``train.load_config``).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_yaml_path = Path(cfg['data']['yaml'])
    if not data_yaml_path.is_absolute():
        data_yaml_path = (Path.cwd() / data_yaml_path).resolve()
    with open(data_yaml_path) as f:
        data_info = yaml.safe_load(f)

    resolution   = cfg['resolution']
    num_workers  = cfg['data'].get('num_workers', 4)
    batch_size   = cfg['train']['batch_size']
    augment_cfg  = cfg.get('augmentation', {})

    is_flat = data_info.get('flat', False)

    def _labels_dir(images_dir: str) -> str:
        """Infer the labels directory path from the images directory path."""
        if is_flat:
            return images_dir
        p = Path(images_dir)
        # Replace the first 'images' component with 'labels'
        parts = list(p.parts)
        for i, part in enumerate(parts):
            if part == 'images':
                parts[i] = 'labels'
                break
        candidate = Path(*parts)
        if candidate.exists():
            return str(candidate)
        # Fallback to images_dir if labels dir doesn't exist (flat directory)
        return images_dir

    train_images = data_info.get('train_dir') or data_info.get('train')
    val_images   = data_info.get('val_dir') or data_info.get('valid') or data_info.get('val')
    test_images  = data_info.get('test_dir') or data_info.get('test') or val_images

    train_labels = _labels_dir(train_images)
    val_labels   = _labels_dir(val_images)
    test_labels  = _labels_dir(test_images)

    prefix_classes = data_info.get('prefix_classes', False)

    train_ds = BirDroneDataset(
        images_dir=train_images,
        labels_dir=train_labels,
        resolution=resolution,
        augment=True,
        augment_cfg=augment_cfg,
        prefix_classes=prefix_classes,
    )
    val_ds = BirDroneDataset(
        images_dir=val_images,
        labels_dir=val_labels,
        resolution=resolution,
        augment=False,
        prefix_classes=prefix_classes,
    )
    test_ds = BirDroneDataset(
        images_dir=test_images,
        labels_dir=test_labels,
        resolution=resolution,
        augment=False,
        prefix_classes=prefix_classes,
    )

    # Class-balanced sampling
    train_sampler = None
    shuffle_train = True
    if cfg.get('train', {}).get('class_balanced', False):
        from torch.utils.data import WeightedRandomSampler
        bird_count = 0
        drone_count = 0
        sample_classes = []
        for img_p, _ in train_ds.samples:
            stem_up = img_p.stem.upper()
            if stem_up.startswith('B'):
                c = 0
            elif stem_up.startswith('D'):
                c = 1
            else:
                c = 0
            sample_classes.append(c)
            if c == 0:
                bird_count += 1
            else:
                drone_count += 1

        w_bird = 1.0 / max(1, bird_count)
        w_drone = 1.0 / max(1, drone_count)
        weights = [w_bird if c == 0 else w_drone for c in sample_classes]
        weights_t = torch.as_tensor(weights, dtype=torch.double)
        max_samples = cfg.get('train', {}).get('max_train_samples', len(train_ds))
        train_sampler = WeightedRandomSampler(
            weights_t, num_samples=min(max_samples, len(train_ds)), replacement=True
        )
        shuffle_train = False
        print(f"[dataset] Class-balanced sampler active: {bird_count} Birds, {drone_count} Drones")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=BirDroneDataset.collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=BirDroneDataset.collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=BirDroneDataset.collate_fn,
        pin_memory=True,
    )

    print(
        f"[dataset] Train: {len(train_ds)} | "
        f"Val: {len(val_ds)} | "
        f"Test: {len(test_ds)} samples"
    )
    return train_loader, val_loader, test_loader
