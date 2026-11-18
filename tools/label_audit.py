"""
label_audit.py - model-assisted audit of EVERY label in the Bird-vs-Drone dataset (rev. 4).

Step 1 (automatic, GPU):  python tools/label_audit.py scan --dataset "D:/.../Dataset" --weights runs/baselines/yolov8s-scratch/weights/best.pt
    A trained detector gives a second opinion on every label:
      ok          a same-class detection (conf >= 0.25) overlaps the label with IoU >= 0.5
      suspect     no detection overlaps the label (IoU < 0.3), but the detector confidently
                  (conf >= 0.5) sees an object of that class elsewhere in the image that no
                  label covers  -> label probably misplaced
      unconfirmed the detector does not see the object at all (hard object or bad label)
    Confident detections (conf >= 0.5) that no label covers are listed as possible missing labels.
    Writes runs/label_audit/labels.csv, summary.json and a review set:
      every suspect image (up to --max-suspects per split) plus a random sample of `ok` and
      `unconfirmed` images per split, drawn with labels (red) and detections (green), and
      review.html to record a verdict for each image in the browser.

Step 2 (you):  open runs/label_audit/review/review.html, choose a verdict for each image,
    press "Save verdicts" -> verdicts.csv (save it into runs/label_audit/review/).

Step 3:  python tools/label_audit.py summarize
    Estimates the misaligned-label rate per split and class from the verdicts:
    rate = sum over statuses of (labels with that status x fraction confirmed misaligned in
    the reviewed sample of that status), with 95% Wilson intervals on each fraction.

Caveat: the detector was trained on ~90% of the training split, so it tends to agree with
training labels; training-split error rates from this audit are lower bounds. The validation
bird images (BV) and the test split were never used to train it.
"""
import argparse, csv, json, math, os, random
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from evaluate_v2 import load_gt, box_iou, CLASS_NAMES   # same label parsing as the evaluator

EXTS = (".jpg", ".jpeg", ".png")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def scan(a):
    from ultralytics import YOLO
    model = YOLO(a.weights)
    name_to_idx = {n.lower(): i for i, n in enumerate(CLASS_NAMES)}
    cls_map = {int(k): name_to_idx[str(v).lower()] for k, v in model.names.items()}
    out = Path(a.out); rev = out / "review"; rev.mkdir(parents=True, exist_ok=True)
    rows, extra, per_image = [], [], {}
    for split in a.splits:
        idir = Path(a.dataset) / split / "images"; ldir = Path(a.dataset) / split / "labels"
        if not idir.exists():
            print(f"skip {split}: {idir} not found"); continue
        files = sorted(f for f in os.listdir(idir) if f.lower().endswith(EXTS))
        print(f"[{split}] {len(files)} images")
        for i in range(0, len(files), a.batch):
            chunk = files[i:i + a.batch]
            res = model.predict([str(idir / f) for f in chunk], imgsz=a.imgsz, conf=0.05, iou=0.6,
                                device=a.device, verbose=False)
            for f, r in zip(chunk, res):
                h0, w0 = r.orig_shape
                gt = load_gt(str(ldir / (Path(f).stem + ".txt")), Path(f).stem, w0, h0, 1.0, 0, 0)
                b = r.boxes
                det = np.zeros((0, 6), np.float32) if b is None or len(b) == 0 else np.concatenate(
                    [b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                     np.array([cls_map[int(k)] for k in b.cls.cpu().numpy()])[:, None]], 1)
                iou = box_iou(gt[:, 1:5], det[:, :4]) if len(gt) and len(det) else np.zeros((len(gt), len(det)))
                same = (gt[:, 0][:, None] == det[:, 5][None, :]) if len(gt) and len(det) else np.zeros_like(iou, bool)
                # confident detections not covered by any label (same class, IoU >= 0.3)
                covered = ((iou >= 0.3) & same).any(0) if len(gt) else np.zeros(len(det), bool)
                orphan = (det[:, 4] >= 0.5) & ~covered
                stats = []
                for g in range(len(gt)):
                    m = same[g] & (det[:, 4] >= 0.25)
                    best = float(iou[g][m].max()) if m.any() else 0.0
                    bconf = float(det[m][np.argmax(iou[g][m]), 4]) if m.any() else 0.0
                    elsewhere = bool((orphan & (det[:, 5] == gt[g, 0])).any())
                    st = "ok" if best >= 0.5 else ("suspect" if best < 0.3 and elsewhere else "unconfirmed")
                    stats.append(st)
                    rows.append({"split": split, "image": f, "label": g, "class": CLASS_NAMES[int(gt[g, 0])],
                                 "x1": round(float(gt[g, 1]), 1), "y1": round(float(gt[g, 2]), 1),
                                 "x2": round(float(gt[g, 3]), 1), "y2": round(float(gt[g, 4]), 1),
                                 "status": st, "best_iou": round(best, 3), "best_conf": round(bconf, 3)})
                for d in np.where(orphan)[0]:
                    extra.append({"split": split, "image": f, "class": CLASS_NAMES[int(det[d, 5])],
                                  "conf": round(float(det[d, 4]), 3),
                                  "box": [round(float(v), 1) for v in det[d, :4]]})
                img_status = "suspect" if "suspect" in stats else ("unconfirmed" if "unconfirmed" in stats else ("ok" if stats else "no_label"))
                per_image[(split, f)] = {"status": img_status, "gt": gt.tolist(), "det": det[det[:, 4] >= 0.25].tolist()}
            if (i // a.batch) % 50 == 0:
                print(f"  {split}: {min(i + a.batch, len(files))}/{len(files)}", flush=True)

    with open(out / "labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(extra, open(out / "possible_missing_labels.json", "w"), indent=1)
    with open(out / "images.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["split", "image", "status"])
        for (sp, f), v in sorted(per_image.items()):
            w.writerow([sp, f, v["status"]])
    summ = {}
    for r in rows:
        k = f"{r['split']}/{r['class']}"; summ.setdefault(k, {"ok": 0, "suspect": 0, "unconfirmed": 0})
        summ[k][r["status"]] += 1
    for k in summ:
        summ[k]["possible_missing"] = sum(1 for e in extra if f"{e['split']}/{e['class']}" == k)
    json.dump(summ, open(out / "summary.json", "w"), indent=1)

    # review set
    rng = random.Random(0); review = []
    for split in a.splits:
        imgs = [(k, v) for k, v in per_image.items() if k[0] == split]
        for status, cap in (("suspect", a.max_suspects), ("ok", a.sample_ok), ("unconfirmed", a.sample_unconfirmed)):
            pool = [k for k, v in imgs if v["status"] == status]
            pick = pool if len(pool) <= cap else rng.sample(pool, cap)
            review += [(k, status, len(pool)) for k in sorted(pick)]
    idx = []
    for (split, f), status, pool_n in review:
        im = cv2.imread(str(Path(a.dataset) / split / "images" / f))
        v = per_image[(split, f)]
        for d in v["det"]:
            cv2.rectangle(im, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (60, 200, 60), 2)
            cv2.putText(im, f"{CLASS_NAMES[int(d[5])][0]} {d[4]:.2f}", (int(d[0]), max(12, int(d[1]) - 4)), 0, 0.45, (60, 200, 60), 1)
        for g in v["gt"]:
            cv2.rectangle(im, (int(g[1]), int(g[2])), (int(g[3]), int(g[4])), (40, 40, 230), 2)
        name = f"{split}__{status}__{Path(f).stem}.jpg".replace(" ", "_")
        cv2.imwrite(str(rev / name), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
        idx.append({"file": name, "split": split, "image": f, "status": status, "pool": pool_n})
    json.dump(idx, open(rev / "review_index.json", "w"), indent=1)
    write_html(rev, idx)
    print(json.dumps(summ, indent=1))
    print(f"Review {len(idx)} images: open {rev / 'review.html'}")


def write_html(rev, idx):
    items = json.dumps(idx)
    html = """<!doctype html><html><head><meta charset="utf-8"><title>Label audit review</title>
<style>body{font-family:system-ui,sans-serif;margin:16px;background:#fafaf8;color:#111}
.card{display:inline-block;width:330px;margin:6px;vertical-align:top;background:#fff;border:1px solid #ddd;padding:6px}
.card img{width:320px;height:320px;object-fit:contain;background:#eee}.meta{font-size:12px;color:#555;margin:4px 0}
select{width:100%}.bar{position:sticky;top:0;background:#fafaf8;padding:8px 0;border-bottom:1px solid #ddd;z-index:1}
.k{display:inline-block;width:12px;height:12px;margin:0 4px -2px 10px}</style></head><body>
<div class="bar"><b>Label audit review</b> <span class="k" style="background:#e62828"></span>label
<span class="k" style="background:#3cc83c"></span>detector &nbsp; Verdict = is the RED label correct?
<button onclick="save()">Save verdicts</button> <span id="n"></span></div><div id="g"></div>
<script>const items=""" + items + """;
const g=document.getElementById('g');
items.forEach((it,i)=>{const d=document.createElement('div');d.className='card';
d.innerHTML=`<img loading="lazy" src="${it.file}"><div class="meta">${it.split} / ${it.image} - flagged: <b>${it.status}</b></div>
<select id="v${i}"><option value="">- verdict -</option><option>correct</option><option>misaligned</option>
<option>wrong_class</option><option>missing_label</option><option>unsure</option></select>`;g.appendChild(d)});
function save(){let rows=['file,split,image,status,pool,verdict'];let n=0;
items.forEach((it,i)=>{const v=document.getElementById('v'+i).value;if(v)n++;
rows.push([it.file,it.split,'"'+it.image+'"',it.status,it.pool,v].join(','))});
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')],{type:'text/csv'}));
a.download='verdicts.csv';a.click();document.getElementById('n').textContent=n+' of '+items.length+' reviewed';}
</script></body></html>"""
    (rev / "review.html").write_text(html, encoding="utf-8")


def summarize(a):
    out = Path(a.out); rev = out / "review"
    ver = {r["file"]: r for r in csv.DictReader(open(rev / "verdicts.csv")) if r.get("verdict")}
    idx = json.load(open(rev / "review_index.json"))
    # image-level fraction of reviewed images whose labels are misaligned/wrong, per split & status
    frac = {}
    for it in idx:
        v = ver.get(it["file"])
        if not v or v["verdict"] == "unsure":
            continue
        k = (it["split"], it["status"]); frac.setdefault(k, [0, 0])
        frac[k][1] += 1; frac[k][0] += v["verdict"] in ("misaligned", "wrong_class")
    report = {}
    images = list(csv.DictReader(open(out / "images.csv")))
    for split in sorted({r["split"] for r in images}):
        st_of = [r["status"] for r in images if r["split"] == split and r["status"] != "no_label"]
        n_img = len(st_of); est = lo = hi = 0.0; detail = {}
        for status in ("suspect", "unconfirmed", "ok"):
            n_s = st_of.count(status)
            k, n = frac.get((split, status), [0, 0])
            ci = wilson(k, n)
            detail[status] = {"images": n_s, "reviewed": n, "confirmed_bad": k,
                              "fraction": k / n if n else None, "ci95": ci if n else None}
            if n:
                est += n_s * k / n; lo += n_s * ci[0]; hi += n_s * ci[1]
            elif n_s:
                detail[status]["note"] = "not reviewed - not included in the estimate"
        report[split] = {"labelled_images": n_img, "est_images_with_bad_labels": est,
                         "est_rate": est / max(n_img, 1), "ci95_rate": (lo / max(n_img, 1), hi / max(n_img, 1)),
                         "by_status": detail}
    json.dump(report, open(out / "audit_estimate.json", "w"), indent=1)
    for s, r in report.items():
        print(f"{s}: ~{r['est_images_with_bad_labels']:.0f} of {r['labelled_images']} images have a misaligned/wrong label "
              f"({100 * r['est_rate']:.1f}%, 95% CI {100 * r['ci95_rate'][0]:.1f}-{100 * r['ci95_rate'][1]:.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scan", "summarize"])
    ap.add_argument("--dataset", default="D:/PROGRAMMING/dr shimaa/Bird vs Drone kaggel/Dataset")
    ap.add_argument("--splits", nargs="*", default=["train", "valid", "test"])
    ap.add_argument("--weights", default="runs/baselines/yolov8s-scratch/weights/best.pt")
    ap.add_argument("--out", default="runs/label_audit")
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-suspects", type=int, default=300)
    ap.add_argument("--sample-ok", type=int, default=100)
    ap.add_argument("--sample-unconfirmed", type=int, default=50)
    a = ap.parse_args()
    scan(a) if a.mode == "scan" else summarize(a)
