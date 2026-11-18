"""
make_clean_split.py — build a clean Bird-vs-Drone split.

Why: an audit of the Kaggle `valid/` split found that ~80% of bird labels (24/30
randomly sampled BV files) do not overlap the bird in the image, so `valid/`
cannot be used for model selection or for reporting bird metrics. The `test/`
split was clean in the audit (8/8 BT, 4/4 DT correct).

What this script does (nothing is deleted or modified in the original Dataset):
  Dataset_clean/
    train/ images, labels   <- 90 % of original train (random, seed 0)
    val/   images, labels   <- 10 % of original train + valid DRONE images (DV*)
    test/  images, labels   <- original test, unchanged
  * labels are rewritten as plain YOLO boxes: polygons -> enclosing box,
    class id taken from the filename prefix (B* -> 0 bird, D* -> 1 drone)
  * images are hard-linked when possible (no extra disk space on NTFS), copied otherwise
  * data_clean.yaml is written for both this code base and Ultralytics

Usage:
  python tools/make_clean_split.py --src "D:/PROGRAMMING/dr shimaa/Bird vs Drone kaggel/Dataset"
"""
import argparse, os, random, shutil
from pathlib import Path


def to_boxes(label_path: Path, cls: int):
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            c = [float(v) for v in p[1:]]
        except ValueError:
            continue
        if len(c) == 4:
            cx, cy, w, h = c
        else:
            xs, ys = c[0::2], c[1::2]
            x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
            y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
            w, h, cx, cy = x2 - x1, y2 - y1, (x1 + x2) / 2, (y1 + y2) / 2
        if w > 1e-4 and h > 1e-4:
            out.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out


def place(img: Path, lbl: Path, dst: Path):
    cls = 0 if img.stem.upper().startswith("B") else 1
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    tgt = dst / "images" / img.name
    if not tgt.exists():
        try:
            os.link(img, tgt)
        except OSError:
            shutil.copy2(img, tgt)
    (dst / "labels" / (img.stem + ".txt")).write_text("\n".join(to_boxes(lbl, cls)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="original Dataset folder (contains train/valid/test)")
    ap.add_argument("--dst", default=None, help="output folder (default: <src>_clean)")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src = Path(a.src); dst = Path(a.dst) if a.dst else src.parent / (src.name + "_clean")
    exts = {".jpg", ".jpeg", ".png"}

    train = sorted(p for p in (src / "train" / "images").iterdir() if p.suffix.lower() in exts)
    random.Random(a.seed).shuffle(train)
    n_val = int(len(train) * a.val_frac)
    splits = {"val": train[:n_val], "train": train[n_val:]}
    splits["val"] += sorted(p for p in (src / "valid" / "images").iterdir()
                            if p.suffix.lower() in exts and p.stem.upper().startswith("D"))
    splits["test"] = sorted(p for p in (src / "test" / "images").iterdir() if p.suffix.lower() in exts)

    for name, files in splits.items():
        for img in files:
            sub = img.parent.parent.name  # train / valid / test
            place(img, src / sub / "labels" / (img.stem + ".txt"), dst / name)
        print(f"{name:5s}: {len(files)} images")

    y = dst / "data_clean.yaml"
    y.write_text(
        f"# Clean split built by tools/make_clean_split.py\n"
        f"path: {dst.as_posix()}\n"
        f"train: {(dst/'train'/'images').as_posix()}\n"
        f"val: {(dst/'val'/'images').as_posix()}\n"
        f"test: {(dst/'test'/'images').as_posix()}\n"
        f"prefix_classes: true\n"
        f"nc: 2\nnames:\n  0: Bird\n  1: Drone\n")
    print("wrote", y)


if __name__ == "__main__":
    main()
