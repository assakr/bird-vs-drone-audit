"""
scene_overlap.py - measure how many test frames also appear in the training split (rev. 4).

Training images are pre-made 4-image mosaics whose tiles are rescaled and strongly
colour-jittered, so colour/CNN embeddings miss many duplicates. This tool works on
grayscale local features instead:
  1. shortlist: RootSIFT descriptors (300 per image) of every test image vote for the
     training images that own their nearest neighbours (GPU matrix products);
  2. verify:    the top --candidates training images per test image are matched with SIFT
     (ratio test) and a RANSAC homography; the number of geometric inliers says how much of
     the test frame reappears in the training mosaic.

Step 1 (GPU recommended):  python tools/scene_overlap.py scan --dataset "D:/.../Dataset"
    -> runs/scene_overlap/matches.csv  and  runs/scene_overlap/review/review.html
       (pairs sampled across inlier levels: test frame | best training mosaic)
Step 2 (you):  open review.html, mark each pair same_scene / different, Save -> verdicts.csv
               (save it into runs/scene_overlap/review/)
Step 3:  python tools/scene_overlap.py summarize
    -> estimated number of test images that have a same-scene training image (95% CI), the
       inlier threshold where pairs become same-scene, and clean_test_images.txt to evaluate on:
       python tools/evaluate_v2.py ... --image-list runs/scene_overlap/clean_test_images.txt
"""
import argparse, csv, json, math, os, random
from pathlib import Path

import cv2
import numpy as np
import torch

EXTS = (".jpg", ".jpeg", ".png")
BINS = [(100, 10 ** 9), (50, 100), (25, 50), (15, 25), (0, 15)]   # RANSAC inlier ranges


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(EXTS))


def gray(path, size=640):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(im, (size, size)) if im.shape[:2] != (size, size) else im


def rootsift(img, n):
    kp, d = cv2.SIFT_create(nfeatures=n).detectAndCompute(img, None)
    if d is None or len(kp) == 0:
        return [], np.zeros((0, 128), np.float32)
    d = d / (d.sum(1, keepdims=True) + 1e-7)
    return kp, np.sqrt(d).astype(np.float32)          # RootSIFT: L2 distance -> Hellinger


def verify(ga, gb, n=1500):
    s = cv2.SIFT_create(nfeatures=n)
    ka, da = s.detectAndCompute(ga, None); kb, db = s.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0, None
    m = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 8:
        return 0, None
    H, mask = cv2.findHomography(np.float32([ka[g.queryIdx].pt for g in good]),
                                 np.float32([kb[g.trainIdx].pt for g in good]), cv2.RANSAC, 5.0)
    return (int(mask.sum()) if mask is not None else 0), H


def scan(a):
    out = Path(a.out); rev = out / "review"; rev.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device); root = Path(a.dataset)
    tr_dir, te_dir = root / a.train_split / "images", root / "test" / "images"
    tr_files, te_files = list_images(tr_dir), list_images(te_dir)
    cache = out / "train_sift.npz"
    if cache.exists():
        z = np.load(cache); D, owner = z["D"], z["owner"]; print(f"loaded cached training descriptors {D.shape}")
    else:
        D, owner = [], []
        for i, f in enumerate(tr_files):
            _, d = rootsift(gray(tr_dir / f), a.shortlist_kp); D.append(d.astype(np.float16)); owner += [i] * len(d)
            if i % 1000 == 0:
                print(f"  train descriptors {i}/{len(tr_files)}", flush=True)
        D, owner = np.concatenate(D), np.array(owner, np.int32); np.savez(cache, D=D, owner=owner)
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    Dt = torch.from_numpy(D).to(dev, dtype); own = torch.from_numpy(owner).long().to(dev)
    chunk = a.db_chunk
    rows = []
    for ti, f in enumerate(te_files):
        g = gray(te_dir / f); _, q = rootsift(g, a.shortlist_kp)
        votes = torch.zeros(len(tr_files), device=dev)
        if len(q):
            Q = torch.from_numpy(q).to(dev, dtype)
            best = torch.full((len(q), 2), -2.0, device=dev); bidx = torch.zeros(len(q), dtype=torch.long, device=dev)
            for k in range(0, len(Dt), chunk):        # running top-2 cosine similarity over the database
                s = (Q @ Dt[k:k + chunk].T).float()
                v, i = s.topk(2, dim=1)
                allv = torch.cat([best, v], 1); alli = torch.cat([bidx[:, None].expand(-1, 2), (i + k)], 1)
                o = allv.argsort(1, descending=True)[:, :2]
                bidx = torch.gather(alli, 1, o[:, :1]).squeeze(1); best = torch.gather(allv, 1, o)
            # ratio test on Hellinger distance: d = sqrt(2 - 2 cos)
            d1 = (2 - 2 * best[:, 0]).clamp(min=0).sqrt(); d2 = (2 - 2 * best[:, 1]).clamp(min=0).sqrt()
            keep = d1 < 0.8 * d2
            votes.index_add_(0, own[bidx[keep]], torch.ones(int(keep.sum()), device=dev))
        cand = votes.topk(min(a.candidates, len(tr_files))).indices.tolist()
        res = []
        for c in cand:
            inl, _ = verify(g, gray(tr_dir / tr_files[c]))
            res.append((inl, c, int(votes[c])))
        res.sort(reverse=True)
        rows.append({"test_image": f, "inliers": res[0][0], "train_image": tr_files[res[0][1]], "votes": res[0][2],
                     "second": f"{tr_files[res[1][1]]}:{res[1][0]}" if len(res) > 1 else ""})
        if ti % 50 == 0:
            print(f"  test {ti}/{len(te_files)}: {f} -> {rows[-1]['train_image']} ({rows[-1]['inliers']} inliers)", flush=True)
    with open(out / "matches.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    rng = random.Random(0); idx = []
    for lo, hi in BINS:
        pool = [r for r in rows if lo <= int(r["inliers"]) < hi]
        for r in (pool if len(pool) <= a.per_bin else rng.sample(pool, a.per_bin)):
            t = cv2.resize(cv2.imread(str(te_dir / r["test_image"])), (360, 360))
            c = cv2.resize(cv2.imread(str(tr_dir / r["train_image"])), (360, 360))
            name = f"{lo:03d}__{Path(r['test_image']).stem}.jpg".replace(" ", "_")
            cv2.imwrite(str(rev / name), np.concatenate([t, np.full((360, 8, 3), 255, np.uint8), c], 1))
            idx.append({"file": name, "test_image": r["test_image"], "train_image": r["train_image"],
                        "inliers": int(r["inliers"]), "bin": f"{lo}-{hi if hi < 10 ** 9 else 'inf'}", "pool": len(pool)})
    json.dump(idx, open(rev / "review_index.json", "w"), indent=1)
    write_html(rev, idx)
    hist = {f"{lo}-{hi if hi < 10 ** 9 else 'inf'}": sum(1 for r in rows if lo <= int(r["inliers"]) < hi) for lo, hi in BINS}
    json.dump({"n_test": len(rows), "n_train_images": len(tr_files), "hist_inliers": hist},
              open(out / "scan_summary.json", "w"), indent=1)
    print("best-match inlier histogram:", hist)
    print(f"Review {len(idx)} pairs: open {rev / 'review.html'}")


def write_html(rev, idx):
    html = """<!doctype html><html><head><meta charset="utf-8"><title>Scene overlap review</title>
<style>body{font-family:system-ui,sans-serif;margin:16px;background:#fafaf8;color:#111}
.card{display:inline-block;width:740px;margin:6px;vertical-align:top;background:#fff;border:1px solid #ddd;padding:6px}
.card img{width:728px}.meta{font-size:12px;color:#555;margin:4px 0}select{width:100%}
.bar{position:sticky;top:0;background:#fafaf8;padding:8px 0;border-bottom:1px solid #ddd;z-index:1}</style></head><body>
<div class="bar"><b>Scene overlap review</b> - left: test frame; right: best-matching training mosaic (tiles are colour-shifted).
Same scene = the test frame (or the same shot/background) appears in one of the mosaic tiles.
<button onclick="save()">Save verdicts</button> <span id="n"></span></div>
<div id="g"></div><script>const items=""" + json.dumps(idx) + """;
const g=document.getElementById('g');items.forEach((it,i)=>{const d=document.createElement('div');d.className='card';
d.innerHTML=`<img loading="lazy" src="${it.file}"><div class="meta">${it.test_image} vs ${it.train_image} - ${it.inliers} geometric matches</div>
<select id="v${i}"><option value="">- verdict -</option><option>same_scene</option><option>different</option><option>unsure</option></select>`;g.appendChild(d)});
function save(){let rows=['file,test_image,train_image,inliers,bin,pool,verdict'];let n=0;
items.forEach((it,i)=>{const v=document.getElementById('v'+i).value;if(v)n++;
rows.push([it.file,'"'+it.test_image+'"','"'+it.train_image+'"',it.inliers,it.bin,it.pool,v].join(','))});
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')],{type:'text/csv'}));
a.download='verdicts.csv';a.click();document.getElementById('n').textContent=n+' of '+items.length+' reviewed';}
</script></body></html>"""
    (rev / "review.html").write_text(html, encoding="utf-8")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(a):
    out = Path(a.out); rev = out / "review"
    rows = list(csv.DictReader(open(out / "matches.csv")))
    ver = [r for r in csv.DictReader(open(rev / "verdicts.csv")) if r["verdict"] in ("same_scene", "different")]
    report = {"bins": {}}; est = lo = hi = 0.0
    for blo, bhi in BINS:
        key = f"{blo}-{bhi if bhi < 10 ** 9 else 'inf'}"
        n_bin = sum(1 for r in rows if blo <= int(r["inliers"]) < bhi)
        v = [r for r in ver if r["bin"] == key]; k = sum(r["verdict"] == "same_scene" for r in v)
        ci = wilson(k, len(v))
        report["bins"][key] = {"test_images": n_bin, "reviewed": len(v), "same_scene": k,
                               "fraction": k / len(v) if v else None, "ci95": ci if v else None}
        if v:
            est += n_bin * k / len(v); lo += n_bin * ci[0]; hi += n_bin * ci[1]
    n = len(rows)
    report.update({"n_test": n, "est_same_scene": est, "est_rate": est / n, "ci95_rate": (lo / n, hi / n)})
    thr = a.threshold
    if thr is None:   # lowest bin edge whose reviewed pairs are mostly (>= 50%) the same scene
        ok = [int(k.split("-")[0]) for k, b in report["bins"].items() if b["fraction"] is not None and b["fraction"] >= 0.5]
        thr = min(ok) if ok else 10 ** 9
    clean = [r["test_image"] for r in rows if int(r["inliers"]) < thr]
    (out / "clean_test_images.txt").write_text("\n".join(clean) + "\n")
    report.update({"inlier_threshold": thr, "clean_test_images": len(clean)})
    json.dump(report, open(out / "overlap_estimate.json", "w"), indent=1)
    print(json.dumps(report["bins"], indent=1))
    print(f"~{est:.0f} of {n} test images ({100 * est / n:.1f}%, 95% CI {100 * lo / n:.1f}-{100 * hi / n:.1f}%) "
          f"have a same-scene training image.")
    print(f"Inlier threshold {thr}: {len(clean)} test images kept in clean_test_images.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scan", "summarize"])
    ap.add_argument("--dataset", default="D:/PROGRAMMING/dr shimaa/Bird vs Drone kaggel/Dataset")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--out", default="runs/scene_overlap")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shortlist-kp", type=int, default=300, help="SIFT keypoints per image for the shortlist")
    ap.add_argument("--candidates", type=int, default=10, help="training images verified per test image")
    ap.add_argument("--db-chunk", type=int, default=400000)
    ap.add_argument("--per-bin", type=int, default=25)
    ap.add_argument("--threshold", type=int, default=None, help="override the inlier threshold for the clean list")
    a = ap.parse_args()
    scan(a) if a.mode == "scan" else summarize(a)
