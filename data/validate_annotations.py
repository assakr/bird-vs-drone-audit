"""
validate_annotations.py
========================
Validates YOLO-format annotation files against their corresponding images
for the LCMS-DB / BirDrone dataset.

Each label file contains one row per bounding box:
    class_id  x_center  y_center  width  height   (all space-separated, normalised 0-1)

Usage (CLI):
    python data/validate_annotations.py \\
        --images-dir data/images/train \\
        --labels-dir data/labels/train \\
        [--valid-classes 0 1] \\
        [--remove-invalid] \\
        [--log-file data/validation_train.log]

Authors : LCMS-DB team
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Supported image extensions (OpenCV readable)
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Per-sample validation
# ---------------------------------------------------------------------------

def validate_single(
    img_path: str | Path,
    label_path: str | Path,
    valid_classes: Sequence[int] = (0, 1),
) -> list[str]:
    """Validate one image / label pair and return a list of error strings.

    An empty list means the pair is fully valid.

    Parameters
    ----------
    img_path:
        Absolute or relative path to the image file.
    label_path:
        Absolute or relative path to the corresponding YOLO .txt label file.
    valid_classes:
        Iterable of accepted integer class IDs (default: 0 = Bird, 1 = Drone).

    Returns
    -------
    list[str]
        Human-readable error descriptions; empty if no errors were found.
    """
    img_path = Path(img_path)
    label_path = Path(label_path)
    errors: list[str] = []

    # ------------------------------------------------------------------
    # Check 1: label file exists and is non-empty
    # ------------------------------------------------------------------
    if not label_path.exists():
        errors.append(f"MISSING_LABEL: {label_path} does not exist")
        return errors  # cannot continue without a label file

    if label_path.stat().st_size == 0:
        errors.append(f"EMPTY_LABEL: {label_path} is empty (no annotations)")
        return errors

    # ------------------------------------------------------------------
    # Read image dimensions (needed for boundary checks)
    # ------------------------------------------------------------------
    img = cv2.imread(str(img_path))
    if img is None:
        errors.append(f"UNREADABLE_IMAGE: cv2 could not open {img_path}")
        # We can still validate box ranges without image dims (use unit square)
        img_h, img_w = 1, 1
    else:
        img_h, img_w = img.shape[:2]

    # ------------------------------------------------------------------
    # Parse label rows
    # ------------------------------------------------------------------
    valid_class_set = set(valid_classes)
    raw_lines = label_path.read_text(encoding="utf-8").strip().splitlines()

    for line_idx, raw_line in enumerate(raw_lines, start=1):
        line = raw_line.strip()
        if not line:
            continue  # skip blank rows silently

        fields = line.split()

        # Check 2: each row must have exactly 5 fields
        if len(fields) != 5:
            errors.append(
                f"BAD_FIELD_COUNT [row {line_idx}]: expected 5 fields, "
                f"got {len(fields)} -> '{raw_line}'"
            )
            continue  # remaining checks require valid field count

        try:
            class_id = int(fields[0])
            x_c, y_c, bw, bh = (float(f) for f in fields[1:])
        except ValueError as exc:
            errors.append(
                f"PARSE_ERROR [row {line_idx}]: cannot convert fields to numbers "
                f"({exc}) -> '{raw_line}'"
            )
            continue

        # Check 3: class_id must be in valid_classes
        if class_id not in valid_class_set:
            errors.append(
                f"INVALID_CLASS [row {line_idx}]: class_id={class_id} "
                f"not in {sorted(valid_class_set)}"
            )

        # Check 4: x_center, y_center, width, height must all be in (0, 1]
        for name, val in [("x_center", x_c), ("y_center", y_c), ("width", bw), ("height", bh)]:
            if not (0.0 < val <= 1.0):
                errors.append(
                    f"OUT_OF_RANGE [row {line_idx}]: {name}={val:.6f} "
                    f"is outside (0, 1]"
                )

        # Check 5: box must not exceed image boundaries
        #   x_center ± width/2  ∈ [0, 1]
        #   y_center ± height/2 ∈ [0, 1]
        x_min = x_c - bw / 2.0
        x_max = x_c + bw / 2.0
        y_min = y_c - bh / 2.0
        y_max = y_c + bh / 2.0

        if x_min < 0.0 or x_max > 1.0:
            errors.append(
                f"BOUNDARY_EXCEEDED [row {line_idx}]: x range [{x_min:.4f}, {x_max:.4f}] "
                f"exceeds [0, 1]"
            )
        if y_min < 0.0 or y_max > 1.0:
            errors.append(
                f"BOUNDARY_EXCEEDED [row {line_idx}]: y range [{y_min:.4f}, {y_max:.4f}] "
                f"exceeds [0, 1]"
            )

        # Check 6: box area must be > 0 (degenerate box guard)
        area = bw * bh
        if area <= 0.0:
            errors.append(
                f"DEGENERATE_BOX [row {line_idx}]: area={area:.8f} — "
                f"width={bw:.6f}, height={bh:.6f}"
            )

    return errors


# ---------------------------------------------------------------------------
# Directory-level validation
# ---------------------------------------------------------------------------

def validate_split(
    images_dir: str | Path,
    labels_dir: str | Path,
    valid_classes: Sequence[int] = (0, 1),
    remove_invalid: bool = False,
    log_file: str | Path | None = None,
) -> dict:
    """Walk *images_dir*, validate each image against its label in *labels_dir*.

    Parameters
    ----------
    images_dir:
        Directory containing image files.
    labels_dir:
        Directory containing YOLO .txt label files (one per image, same stem).
    valid_classes:
        Accepted class IDs (default: 0=Bird, 1=Drone).
    remove_invalid:
        If True, delete both the image and its label file when errors are found.
    log_file:
        Optional path to write a detailed per-file log.

    Returns
    -------
    dict
        Summary statistics:
        ``{total, valid, with_errors, error_type_counts, invalid_files}``
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    # ------------------------------------------------------------------
    # Configure logging
    # ------------------------------------------------------------------
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path), mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Collect image paths
    # ------------------------------------------------------------------
    image_paths = sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        logger.warning("No images found in %s", images_dir)

    # ------------------------------------------------------------------
    # Per-image validation
    # ------------------------------------------------------------------
    total = 0
    valid_count = 0
    error_count = 0
    error_type_counts: dict[str, int] = {}
    invalid_files: list[dict] = []

    for img_path in image_paths:
        total += 1
        # Derive label path: same stem, .txt extension, under labels_dir
        rel = img_path.relative_to(images_dir)
        label_path = labels_dir / rel.with_suffix(".txt")

        errors = validate_single(img_path, label_path, valid_classes=valid_classes)

        if errors:
            error_count += 1
            invalid_files.append({"image": str(img_path), "label": str(label_path), "errors": errors})
            logger.warning("INVALID  %s", img_path.name)
            for err in errors:
                logger.warning("         %s", err)
                # Tally error type prefix (everything before the first '[' or ':')
                etype = err.split("[")[0].split(":")[0].strip()
                error_type_counts[etype] = error_type_counts.get(etype, 0) + 1

            # Optionally remove invalid pairs
            if remove_invalid:
                try:
                    img_path.unlink()
                    logger.info("  REMOVED image:  %s", img_path)
                except OSError as exc:
                    logger.error("  Could not remove image %s: %s", img_path, exc)
                if label_path.exists():
                    try:
                        label_path.unlink()
                        logger.info("  REMOVED label:  %s", label_path)
                    except OSError as exc:
                        logger.error("  Could not remove label %s: %s", label_path, exc)
        else:
            valid_count += 1
            logger.debug("OK       %s", img_path.name)

    stats = {
        "total": total,
        "valid": valid_count,
        "with_errors": error_count,
        "error_type_counts": error_type_counts,
        "invalid_files": invalid_files,
    }

    # ------------------------------------------------------------------
    # Summary banner
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("  Images dir : %s", images_dir)
    logger.info("  Labels dir : %s", labels_dir)
    logger.info("  Total checked  : %d", total)
    logger.info("  Valid          : %d", valid_count)
    logger.info("  With errors    : %d", error_count)
    if error_type_counts:
        logger.info("  Error breakdown:")
        for etype, cnt in sorted(error_type_counts.items(), key=lambda x: -x[1]):
            logger.info("    %-30s %d", etype, cnt)
    logger.info("=" * 60)

    return stats


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run split-level validation."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate YOLO-format annotations for the LCMS-DB / BirDrone dataset. "
            "Checks field count, class IDs, normalised coordinate ranges, "
            "boundary violations, and degenerate boxes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        type=Path,
        help="Directory containing image files (searched recursively).",
    )
    parser.add_argument(
        "--labels-dir",
        required=True,
        type=Path,
        help="Directory containing YOLO .txt label files.",
    )
    parser.add_argument(
        "--valid-classes",
        nargs="+",
        type=int,
        default=[0, 1],
        metavar="CLASS_ID",
        help="Whitelist of valid integer class IDs.",
    )
    parser.add_argument(
        "--remove-invalid",
        action="store_true",
        help="Delete image and label files that contain errors.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to save the detailed log to a text file.",
    )

    args = parser.parse_args()

    stats = validate_split(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        valid_classes=args.valid_classes,
        remove_invalid=args.remove_invalid,
        log_file=args.log_file,
    )

    # Final console print (machine-parseable for CI pipelines)
    print("\n--- FINAL STATS ---")
    print(f"Total checked : {stats['total']}")
    print(f"Valid         : {stats['valid']}")
    print(f"With errors   : {stats['with_errors']}")
    if stats["error_type_counts"]:
        print("Error types:")
        for etype, cnt in sorted(stats["error_type_counts"].items(), key=lambda x: -x[1]):
            print(f"  {etype:<30} {cnt}")


if __name__ == "__main__":
    main()
