"""external_size_subset.py - recall (and AP@0.5) on the external set on the external set, all objects vs objects >= 32x32 px (COCO-style area range:
ground truth outside the range is ignored, as are detections matched to it and unmatched detections
outside the range)."""
import json, sys, numpy as np
# Usage (from the repository root, after tools/run_external_test.bat): python tools/external_size_subset.py
# AP here is computed from the saved detections, which stop at confidence 0.05; report recall, not AP.

class E:
    @staticmethod
    def box_iou(a, b):
        if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)))
        ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
        ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)
    @staticmethod
    def ap_from(tp, conf, n_gt):   # copied from tools/evaluate_v2.py
        if n_gt == 0: return float("nan"), None, None
        if len(tp) == 0: return 0.0, None, None
        o = np.argsort(-conf); tpc = np.cumsum(tp[o]); fpc = np.cumsum(~tp[o])
        rec = tpc / n_gt; prec = tpc / (tpc + fpc)
        mpre = np.concatenate([[1.0], prec, [0.0]]); mrec = np.concatenate([[0.0], rec, [1.0]])
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        x = np.linspace(0, 1, 101)
        return float(np.trapezoid(np.interp(x, mrec, mpre), x)), rec, prec

def evaluate(path, min_area):
    PI = json.load(open(path))
    res = {}
    for c in (0, 1):
        tps, confs, ngt, nrec = [], [], 0, 0
        for im in PI:
            g = np.array(im["gt"], float).reshape(-1, 5); d = np.array(im["det"], float).reshape(-1, 6)
            g = g[g[:, 0] == c]; d = d[d[:, 5] == c]
            ga = (g[:, 3] - g[:, 1]) * (g[:, 4] - g[:, 2]); valid = ga >= min_area
            ngt += int(valid.sum())
            if len(d) == 0: continue
            iou = E.box_iou(d[:, :4], g[:, 1:5]) if len(g) else np.zeros((len(d), 0))
            used = np.zeros(len(g), bool); da = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])
            hit25 = np.zeros(len(g), bool)
            for k in np.argsort(-d[:, 4]):
                cand = np.where((iou[k] >= 0.5) & ~used & valid)[0] if len(g) else []
                if len(cand):
                    j = cand[np.argmax(iou[k, cand])]; used[j] = True
                    tps.append(True); confs.append(d[k, 4])
                    if d[k, 4] >= 0.25: hit25[j] = True
                    continue
                ign = np.where((iou[k] >= 0.5) & ~valid)[0] if len(g) else []
                if len(ign) or da[k] < min_area: continue
                tps.append(False); confs.append(d[k, 4])
            nrec += int((hit25 & valid).sum())
        ap, _, _ = E.ap_from(np.array(tps, bool), np.array(confs), ngt)
        res[c] = dict(ap=100 * ap, n=ngt, recall25=100 * nrec / max(ngt, 1))
    return res

root = sys.argv[1] if len(sys.argv) > 1 else "."; R = root + "/runs/"
models = [("LCMS-DB (final)", "final_full_budget/eval_external"), ("LCMS-DB A7", "A7_full_wiou_scaleaware/eval_external"),
          ("YOLOv8s (scratch)", "baselines/yolov8s-scratch_external"), ("YOLOv8s (COCO)", "baselines/yolov8s_external"),
          ("YOLOv8n (COCO)", "baselines/yolov8n_external"), ("YOLOv8s-P2 (COCO)", "baselines/yolov8s-p2_external"),
          ("YOLO11s (COCO)", "baselines/yolo11s_external")]
out = {}
for name, p in models:
    a = evaluate(R + p + "/per_image.json", 0); b = evaluate(R + p + "/per_image.json", 32 * 32)
    t9 = json.load(open(R + p + "/eval_v2.json"))["overall"]["mAP50"] * 100
    out[name] = dict(all=a, ge32=b, table9_map50=t9)
    print(f"{name:20s} T9 {t9:5.1f} | all: mAP {(a[0]['ap']+a[1]['ap'])/2:5.1f} bird {a[0]['ap']:5.1f} drone {a[1]['ap']:5.1f} "
          f"| >=32px: mAP {(b[0]['ap']+b[1]['ap'])/2:5.1f} bird {b[0]['ap']:5.1f} (n={b[0]['n']}, R {b[0]['recall25']:4.1f}) "
          f"drone {b[1]['ap']:5.1f} (n={b[1]['n']}, R {b[1]['recall25']:4.1f})")
json.dump(out, open(R + "external_ge32.json", "w"), indent=1)
print("written", R + "external_ge32.json")
