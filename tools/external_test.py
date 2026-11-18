"""
external_test.py - cross-dataset test on images from a different source (rev. 5).

Why: tools/scene_overlap.py found that almost every Kaggle test frame has a same-scene image
in the training split, so the Kaggle test split cannot show how a model behaves on unseen
scenes. The Roboflow "birdrone" dataset (already in the project folder) was collected by a
different author, so it is used here as an external test set. No model is retrained.

  prep   copy the external images into a folder that tools/evaluate_v2.py can score.
         evaluate_v2.py takes the class from the file-name prefix (B = bird, D = drone), so
         every image is copied as B_<name> or D_<name> according to the class ids in its
         label file (0 = bird, 1 = drone, the order in the dataset's data.yaml). Images that
         contain both classes cannot be expressed this way and are skipped (counted in the
         report). Images without labels are kept as D_<name>; they only add false alarms.
  check  measure whether the external images share scenes with the Kaggle TRAINING split,
         with the same SIFT voting + RANSAC test as tools/scene_overlap.py, on a random
         sample. Reuses runs/scene_overlap/train_sift.npz.

Usage (from code/lcms_db):
  python tools/external_test.py prep
  python tools/external_test.py check --device cuda
Then score any model with tools/evaluate_v2.py --images <out>/images --labels <out>/labels
(tools/run_external_test.bat does all of it).
"""
import argparse, csv, json, os, random, shutil, sys
from pathlib import Path

EXTS = (".jpg", ".jpeg", ".png")
HERE = Path(__file__).resolve().parent
PROJECT = "D:/PROGRAMMING/dr shimaa"


def label_classes(path):
    ids = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.split()
            if len(p) >= 5:
                try:
                    ids.add(int(float(p[0])))
                except ValueError:
                    pass
    return ids


def prep(a):
    src, out = Path(a.src), Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(parents=True, exist_ok=True)
    rep = {"source": str(src), "splits": a.splits, "bird_images": 0, "drone_images": 0, "unlabelled_images": 0,
           "skipped_mixed_class": 0, "skipped_unknown_class": 0, "bird_boxes": 0, "drone_boxes": 0}
    for split in a.splits:
        idir, ldir = src / split / "images", src / split / "labels"
        if not idir.is_dir():
            sys.exit(f"missing folder: {idir}")
        for f in sorted(os.listdir(idir)):
            if not f.lower().endswith(EXTS):
                continue
            lab = ldir / (Path(f).stem + ".txt")
            ids = label_classes(lab)
            if ids - {0, 1}:
                rep["skipped_unknown_class"] += 1; continue
            if ids == {0, 1}:
                rep["skipped_mixed_class"] += 1; continue
            n = sum(1 for l in lab.read_text(encoding="utf-8", errors="ignore").splitlines() if len(l.split()) >= 5) if ids else 0
            if ids == {0}:
                pre = "B"; rep["bird_images"] += 1; rep["bird_boxes"] += n
            elif ids == {1}:
                pre = "D"; rep["drone_images"] += 1; rep["drone_boxes"] += n
            else:
                pre = "D"; rep["unlabelled_images"] += 1
            name = f"{pre}_{split}_{f}"
            shutil.copy2(idir / f, out / "images" / name)
            (out / "labels" / (Path(name).stem + ".txt")).write_text(
                lab.read_text(encoding="utf-8", errors="ignore") if lab.exists() else "", encoding="utf-8")
    rep["images_written"] = rep["bird_images"] + rep["drone_images"] + rep["unlabelled_images"]
    json.dump(rep, open(out / "prep_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    if rep["bird_images"] == 0 or rep["drone_images"] == 0:
        print("WARNING: one class has no images; check that label id 0 = bird and 1 = drone in", src / "data.yaml")


def check(a):
    import numpy as np, torch
    sys.path.insert(0, str(HERE))
    from scene_overlap import gray, rootsift, verify, list_images, wilson
    out = Path(a.out); ext = out / "images"
    tr_dir = Path(a.kaggle) / "train" / "images"
    tr_files = list_images(tr_dir)
    cache = Path(a.cache)
    if cache.exists():
        z = np.load(cache); D, owner = z["D"], z["owner"]
    else:
        print("no cached training descriptors; computing them (slow)")
        D, owner = [], []
        for i, f in enumerate(tr_files):
            _, d = rootsift(gray(tr_dir / f), 300); D.append(d.astype(np.float16)); owner += [i] * len(d)
        D, owner = np.concatenate(D), np.array(owner, np.int32)
    if int(owner.max()) >= len(tr_files):
        sys.exit("cached descriptors do not match the training folder; delete the cache and re-run scene_overlap.py scan")
    dev = torch.device(a.device)
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    Dt = torch.from_numpy(D).to(dev, dtype); own = torch.from_numpy(owner).long().to(dev)
    files = list_images(ext)
    files = random.Random(0).sample(files, min(a.sample, len(files)))
    rows = []
    for k, f in enumerate(files):
        g = gray(ext / f); _, q = rootsift(g, 300)
        votes = torch.zeros(len(tr_files), device=dev)
        if len(q):
            Q = torch.from_numpy(q).to(dev, dtype)
            for s in range(0, len(Dt), a.db_chunk):          # every close descriptor votes for its image
                sim = (Q @ Dt[s:s + a.db_chunk].T).float()
                hit = (sim > a.sim).nonzero()[:, 1] + s
                votes.index_add_(0, own[hit], torch.ones(len(hit), device=dev))
        cand = votes.topk(min(a.candidates, len(tr_files))).indices.tolist()
        best = max((verify(g, gray(tr_dir / tr_files[c]))[0], tr_files[c]) for c in cand)
        rows.append({"external_image": f, "inliers": best[0], "train_image": best[1]})
        if k % 25 == 0:
            print(f"  {k}/{len(files)}: {f} -> {best[1]} ({best[0]} geometric matches)", flush=True)
    with open(out / "overlap_with_kaggle_train.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n = len(rows); k = sum(r["inliers"] >= a.threshold for r in rows)
    rep = {"sampled_external_images": n, "inlier_threshold": a.threshold, "possible_same_scene": k,
           "rate": k / n, "ci95": wilson(k, n), "max_inliers": max(r["inliers"] for r in rows)}
    json.dump(rep, open(out / "overlap_with_kaggle_train.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print("Pairs at or above the threshold are listed in overlap_with_kaggle_train.csv; look at them before "
          "calling the external set independent.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "check"])
    ap.add_argument("--src", default=f"{PROJECT}/birdrone dataset.v10i.yolov8")
    ap.add_argument("--splits", nargs="+", default=["valid"],
                    help="external splits to use; 'train' holds two rotated copies of each source image")
    ap.add_argument("--out", default=f"{PROJECT}/External_birdrone")
    ap.add_argument("--kaggle", default=f"{PROJECT}/Bird vs Drone kaggel/Dataset")
    ap.add_argument("--cache", default="runs/scene_overlap/train_sift.npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--candidates", type=int, default=10)
    ap.add_argument("--sim", type=float, default=0.9, help="RootSIFT cosine similarity for a vote")
    ap.add_argument("--threshold", type=int, default=15, help="geometric matches that count as a possible same scene")
    ap.add_argument("--db-chunk", type=int, default=400000)
    a = ap.parse_args()
    prep(a) if a.mode == "prep" else check(a)
