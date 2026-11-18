"""pitfalls.py - re-score one set of saved detections under each setting of the original
evaluation script, changing one thing at a time (Table II of the paper).

  python tools/evaluate_v2.py --config ... --checkpoint ... --images ... --labels ... ^
         --out runs/final_full_budget/eval_all --keep-all-dets --device cuda
  python tools/pitfalls.py final runs/final_full_budget/eval_all/per_image.json

Output: pitfalls_<name>.json next to the eval folder."""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_v2 as E   # box_iou, ap_from, size_bucket

name, src = sys.argv[1], sys.argv[2]
PI = json.load(open(src))


def match(det, gt, thr):
    """Greedy, score-ordered, class-aware matching at one IoU threshold (same as evaluate_v2)."""
    tp = np.zeros(len(det), bool); gidx = -np.ones(len(det), int)
    if len(det) == 0 or len(gt) == 0:
        return tp, gidx
    iou = E.box_iou(det[:, :4], gt[:, 1:5])
    iou = np.where(det[:, 5][:, None] == gt[:, 0][None, :], iou, 0.0)
    used = np.zeros(len(gt), bool)
    for d in np.argsort(-det[:, 4]):
        c = np.where((iou[d] >= thr) & ~used)[0]
        if len(c):
            g = c[np.argmax(iou[d, c])]; used[g] = True; tp[d] = True; gidx[d] = g
    return tp, gidx


def score(iou_thr=0.5, conf_floor=0.001):
    """Per-class AP and pooled records under one setting."""
    recs, gts = [], []
    for it in PI:
        det = np.array(it["det"], np.float32).reshape(-1, 6); gt = np.array(it["gt"], np.float32).reshape(-1, 5)
        det = det[det[:, 4] >= conf_floor]
        tp, gidx = match(det, gt, iou_thr)
        for d in range(len(det)):
            g = gidx[d]
            sz = E.size_bucket(gt[g, 3] - gt[g, 1], gt[g, 4] - gt[g, 2]) if g >= 0 else E.size_bucket(det[d, 2] - det[d, 0], det[d, 3] - det[d, 1])
            recs.append((int(det[d, 5]), float(det[d, 4]), bool(tp[d]), sz))
        for g in gt:
            gts.append((int(g[0]), E.size_bucket(g[3] - g[1], g[4] - g[2])))
    return recs, gts


def ap(recs, gts, cls, size=None, tp_only=False):
    r = [x for x in recs if x[0] == cls and (size is None or x[3] == size) and (not tp_only or x[2])]
    n = sum(1 for g in gts if g[0] == cls and (size is None or g[1] == size))
    a, _, _ = E.ap_from(np.array([x[2] for x in r], bool), np.array([x[1] for x in r]), n)
    return a


def mAP(recs, gts):
    return float(np.mean([ap(recs, gts, c) for c in (0, 1)]))


out = {"model": name}
ok = score(0.5, 0.001)
out["correct"] = {"mAP50": mAP(*ok), "Bird_AP50": ap(*ok, 0), "Drone_AP50": ap(*ok, 1)}
m5095 = [mAP(*score(t, 0.001)) for t in np.round(np.arange(0.5, 0.96, 0.05), 2)]
out["correct"]["mAP50_95"] = float(np.mean(m5095))
v = score(0.45, 0.001); out["iou045"] = {"mAP50": mAP(*v), "Bird_AP50": ap(*v, 0), "Drone_AP50": ap(*v, 1)}
v = score(0.5, 0.25); out["floor025"] = {"mAP50": mAP(*v), "Bird_AP50": ap(*v, 0), "Drone_AP50": ap(*v, 1)}
v = score(0.45, 0.25); out["old_combo"] = {"mAP50": mAP(*v), "Bird_AP50": ap(*v, 0), "Drone_AP50": ap(*v, 1)}
out["old_combo"]["mAP50_95_bug"] = out["old_combo"]["mAP50"]   # old code: threshold ignored

# size buckets: corrected AP vs. the old TP-only records at the old operating setting
sz = {}
for b in E.SIZE_BUCKETS:
    sz[b] = {}
    for c, cn in ((0, "Bird"), (1, "Drone")):
        n = sum(1 for g in ok[1] if g[0] == c and g[1] == b)
        if n == 0:
            continue
        sz[b][cn] = {"n": n, "AP50_correct": ap(*ok, c, b), "AP_tp_only_old": ap(*v, c, b, tp_only=True)}
out["size"] = sz

# confusion matrix without background (old) vs with background, at conf 0.25, IoU 0.5
cm = np.zeros((3, 3), int)
for it in PI:
    det = np.array(it["det"], np.float32).reshape(-1, 6); gt = np.array(it["gt"], np.float32).reshape(-1, 5)
    det = det[det[:, 4] >= 0.25]
    used = np.zeros(len(gt), bool)
    iou = E.box_iou(det[:, :4], gt[:, 1:5]) if len(det) and len(gt) else np.zeros((len(det), len(gt)))
    for d in np.argsort(-det[:, 4]):
        c = np.where((iou[d] >= 0.5) & ~used)[0] if len(gt) else []
        if len(c):
            g = c[np.argmax(iou[d, c])]; used[g] = True; cm[int(det[d, 5]), int(gt[g, 0])] += 1
        else:
            cm[int(det[d, 5]), 2] += 1
    for g in np.where(~used)[0]:
        cm[2, int(gt[g, 0])] += 1
out["cm_with_background"] = cm.tolist()
out["hidden_by_old_cm"] = {"false_alarms": int(cm[0, 2] + cm[1, 2]), "misses": int(cm[2, 0] + cm[2, 1])}
json.dump(out, open(Path(src).parent.parent / f"pitfalls_{name}.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ["correct", "iou045", "floor025", "old_combo", "hidden_by_old_cm"]}, indent=1))
