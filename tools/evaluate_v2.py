"""
evaluate_v2.py  —  Corrected evaluation for LCMS-DB.

Fixes relative to the original evaluate.py:
  1. TP matching uses IoU >= 0.50 (the old code reused the NMS threshold 0.45).
  2. mAP@0.5:0.95 really re-matches detections at each IoU threshold
     (the old code ignored the threshold, so mAP@.5:.95 == mAP@.5 always).
  3. AP is computed from all detections down to conf=0.001 (COCO / Ultralytics
     practice); P/R/F1 are reported at the deployment threshold conf=0.25.
  4. Size-bucket AP includes false positives (old code stored only TPs, which
     forced precision = 1.0 and AP = recall).
  5. Confusion matrix has a background row/column, so false alarms and misses
     are visible (old matrix silently dropped them).
  6. GT boxes are mapped through the same letterbox transform as the image.
  7. Latency is measured at batch size 1, including decode + NMS.
  8. Per-class bounding boxes use class-aware NMS.

Usage (from code/lcms_db):
  python tools/evaluate_v2.py --config runs/v2_full/config.yaml --checkpoint runs/v2_full/best.pt ^
      --images "D:/PROGRAMMING/dr shimaa/Bird vs Drone kaggel/Dataset/test/images" ^
      --labels "D:/PROGRAMMING/dr shimaa/Bird vs Drone kaggel/Dataset/test/labels" ^
      --out runs/v2_full/eval_v2_test --device cuda
"""
import argparse, json, os, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import yaml

HERE = Path(__file__).resolve().parent
CLASS_NAMES = ["Bird", "Drone"]
IOU_THRS = np.round(np.arange(0.50, 0.96, 0.05), 2)
SIZE_BUCKETS = {"tiny": (0, 32 ** 2), "small": (32 ** 2, 96 ** 2),
                "medium": (96 ** 2, 256 ** 2), "large": (256 ** 2, 1e12)}


# ----------------------------------------------------------------- data ---
def letterbox(img, size=640, color=(114, 114, 114)):
    h0, w0 = img.shape[:2]
    s = min(size / w0, size / h0)
    nw, nh = int(round(w0 * s)), int(round(h0 * s))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pw, ph = size - nw, size - nh
    top, left = ph // 2, pw // 2
    img = cv2.copyMakeBorder(img, top, ph - top, left, pw - left,
                             cv2.BORDER_CONSTANT, value=color)
    return img, s, left, top


def to_tensor(img):
    x = img[:, :, ::-1].astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]


def load_gt(label_path, stem, w0, h0, s, left, top):
    """Return (N,5) [cls, x1,y1,x2,y2] in letterboxed pixel space.
    Class from filename prefix (B->0, D->1), polygons converted to boxes."""
    cls_from_name = 0 if stem.upper().startswith("B") else 1
    out = []
    if not os.path.exists(label_path):
        return np.zeros((0, 5), np.float32)
    for line in open(label_path):
        p = line.split()
        if len(p) < 5:
            continue
        c = [float(v) for v in p[1:]]
        if len(c) == 4:
            cx, cy, w, h = c
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        else:
            xs, ys = c[0::2], c[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        x1, x2 = np.clip([x1, x2], 0, 1) * w0
        y1, y2 = np.clip([y1, y2], 0, 1) * h0
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        out.append([cls_from_name, x1 * s + left, y1 * s + top, x2 * s + left, y2 * s + top])
    return np.array(out, np.float32).reshape(-1, 5)


# ---------------------------------------------------------- decoding ---
CENTER_DECODE = "sigmoid"


def decode(preds, strides, res, conf_thr=0.001, iou_thr=0.6, max_det=300, center_decode="sigmoid"):
    boxes, scores, clss = [], [], []
    for p, st in zip(preds, strides):
        p = p[0].float()
        g = int(p.shape[0] ** 0.5)
        gy, gx = torch.meshgrid(torch.arange(g, device=p.device), torch.arange(g, device=p.device), indexing="ij")
        gx, gy = gx.reshape(-1).float(), gy.reshape(-1).float()
        if center_decode == "linear":      # rev. 2 head (scale-aware assigner)
            x = (p[:, 0] + gx + 0.5) * st
            y = (p[:, 1] + gy + 0.5) * st
        else:                              # original head
            x = (p[:, 0].sigmoid() + gx) * st
            y = (p[:, 1].sigmoid() + gy) * st
        w = p[:, 2].exp().clamp(max=res) * st
        h = p[:, 3].exp().clamp(max=res) * st
        obj = p[:, 4].sigmoid()
        cl = p[:, 5:].sigmoid() * obj[:, None]
        sc, ci = cl.max(1)
        boxes.append(torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1))
        scores.append(sc); clss.append(ci)
    b, s, c = torch.cat(boxes), torch.cat(scores), torch.cat(clss)
    k = s > conf_thr
    b, s, c = b[k].clamp(0, res), s[k], c[k]
    keep = torchvision.ops.batched_nms(b, s, c, iou_thr)[:max_det]
    return torch.cat([b[keep], s[keep, None], c[keep, None].float()], 1).cpu().numpy()


# ---------------------------------------------------------- matching ---
def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def match_image(det, gt):
    """Greedy, score-ordered, class-aware matching at every IoU threshold.
    Returns tp (n_det, n_thr) bool and gt_idx (n_det,) at IoU .5 (-1 if none)."""
    n = len(det)
    tp = np.zeros((n, len(IOU_THRS)), bool)
    gt_idx50 = -np.ones(n, int)
    if n == 0 or len(gt) == 0:
        return tp, gt_idx50
    order = np.argsort(-det[:, 4])
    iou = box_iou(det[:, :4], gt[:, 1:5])
    same = det[:, 5][:, None] == gt[:, 0][None, :]
    iou = np.where(same, iou, 0.0)
    for t, thr in enumerate(IOU_THRS):
        used = np.zeros(len(gt), bool)
        for d in order:
            cand = np.where((iou[d] >= thr) & ~used)[0]
            if len(cand):
                g = cand[np.argmax(iou[d, cand])]
                used[g] = True; tp[d, t] = True
                if t == 0:
                    gt_idx50[d] = g
    return tp, gt_idx50


def ap_from(tp, conf, n_gt):
    """101-point interpolated AP (COCO style). tp: (n,) bool."""
    if n_gt == 0:
        return float("nan"), None, None
    if len(tp) == 0:
        return 0.0, np.array([0.0]), np.array([1.0])
    o = np.argsort(-conf)
    tpc = np.cumsum(tp[o]); fpc = np.cumsum(~tp[o])
    rec = tpc / n_gt; prec = tpc / (tpc + fpc)
    mpre = np.concatenate([[1.0], prec, [0.0]]); mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    ap = (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(np.interp(x, mrec, mpre), x)
    return float(ap), rec, prec


def size_bucket(w, h):
    a = w * h
    for k, (lo, hi) in SIZE_BUCKETS.items():
        if lo <= a < hi:
            return k
    return "large"


# -------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="LCMS-DB config (not needed with --yolo)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--code-dir", default=str(HERE.parent if (HERE.parent / "models").exists() else HERE))
    ap.add_argument("--conf", type=float, default=0.25, help="operating threshold for P/R/F1/confusion")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--timing-images", type=int, default=100)
    ap.add_argument("--yolo", action="store_true",
                    help="--checkpoint is an Ultralytics model (e.g. runs/baselines/yolov8s/weights/best.pt); "
                         "it is scored with exactly the same matching / AP / size / confusion code as LCMS-DB")
    ap.add_argument("--imgsz", type=int, default=640, help="input size for --yolo")
    ap.add_argument("--keep-all-dets", action="store_true",
                    help="store every detection (down to conf 0.001) in per_image.json; needed by tools/pitfalls.py")
    ap.add_argument("--image-list", default=None,
                    help="text file with one image file name per line; only these images are evaluated "
                         "(e.g. runs/scene_overlap/clean_test_images.txt)")
    args = ap.parse_args()
    if args.yolo:
        return main_common(args, *build_yolo_backend(args))
    if not args.config:
        ap.error("--config is required unless --yolo is given")

    sys.path.insert(0, args.code_dir)
    from models.lcms_db import build_model, load_checkpoint

    try:   # same loader as train.py: resolves `_base_:` inheritance used by the ablation configs
        from train import load_config
        cfg = load_config(args.config)
    except ImportError:
        cfg = yaml.safe_load(open(args.config))
        if "_base_" in cfg:
            base = yaml.safe_load(open(os.path.join(os.path.dirname(args.config), cfg.pop("_base_"))))
            def merge(a, b):
                out = dict(a)
                for k, v in b.items():
                    out[k] = merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
                return out
            cfg = merge(base, cfg)
    res = cfg["resolution"]
    dev = torch.device(args.device)
    model = build_model(cfg)
    model, meta = load_checkpoint(model, args.checkpoint, str(dev))
    model.eval()
    strides = model.strides
    global CENTER_DECODE
    CENTER_DECODE = getattr(model.head, "center_decode", "sigmoid")
    print(f"center decode: {CENTER_DECODE}")

    def infer(img, conf_thr=0.001):
        with torch.no_grad():
            preds = model(to_tensor(img).to(dev))
        return decode(preds, strides, res, conf_thr=conf_thr, center_decode=CENTER_DECODE)

    def complexity():
        try:
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(display=False) as fc:
                model(torch.zeros(1, 3, res, res).to(dev))
            c = {"GFLOPs": fc.get_total_flops() / 1e9}
        except Exception as e:
            c = {"GFLOPs": None, "err": str(e)}
        params = sum(p.numel() for p in model.parameters())
        c.update({"params_M": params / 1e6, "fp32_weights_MB": params * 4 / 2 ** 20,
                  "checkpoint_MB": os.path.getsize(args.checkpoint) / 2 ** 20,
                  "checkpoint_epoch": int(meta.get("epoch", -1)) + 1})
        return c

    return main_common(args, res, dev, infer, complexity, "LCMS-DB")


def build_yolo_backend(args):
    """Ultralytics model -> same infer() contract: (N,6) [x1,y1,x2,y2,conf,cls] in letterboxed pixels."""
    from ultralytics import YOLO
    ym = YOLO(args.checkpoint)
    dev = torch.device(args.device)
    res = args.imgsz
    name_to_idx = {n.lower(): i for i, n in enumerate(CLASS_NAMES)}
    cls_map = {int(k): name_to_idx[str(v).lower()] for k, v in ym.names.items()}
    ydev = "cpu" if dev.type == "cpu" else (dev.index if dev.index is not None else 0)
    print(f"Ultralytics model {args.checkpoint}  classes {ym.names} -> {cls_map}")

    def infer(img, conf_thr=0.001):
        # img is already letterboxed to res x res, so Ultralytics adds no padding and
        # returns boxes in the same pixel space as our ground truth.
        r = ym.predict(img, imgsz=res, conf=conf_thr, iou=0.6, max_det=300, device=ydev,
                       agnostic_nms=False, verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return np.zeros((0, 6), np.float32)
        c = np.array([cls_map[int(k)] for k in b.cls.cpu().numpy()], np.float32)
        return np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], c[:, None]], 1).astype(np.float32)

    def complexity():
        net = ym.model
        c = {}
        try:
            from torch.utils.flop_counter import FlopCounterMode
            p0 = next(net.parameters())
            with FlopCounterMode(display=False) as fc, torch.no_grad():
                net(torch.zeros(1, 3, res, res, device=p0.device, dtype=p0.dtype))
            c["GFLOPs"] = fc.get_total_flops() / 1e9
        except Exception as e:
            c.update({"GFLOPs": None, "err": str(e)})
        params = sum(p.numel() for p in net.parameters())
        ep = -1
        try:
            ep = int((ym.ckpt or {}).get("epoch", -2)) + 1
            ep = ep if ep > 0 else None      # Ultralytics clears the epoch in its final best.pt
        except Exception:
            pass
        c.update({"params_M": params / 1e6, "fp32_weights_MB": params * 4 / 2 ** 20,
                  "checkpoint_MB": os.path.getsize(args.checkpoint) / 2 ** 20, "checkpoint_epoch": ep})
        return c

    return res, dev, infer, complexity, "Ultralytics"


def main_common(args, res, dev, infer, complexity, backend):
    files = sorted(f for f in os.listdir(args.images) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if args.image_list:
        keep = {l.strip() for l in open(args.image_list, encoding="utf-8") if l.strip()}
        files = [f for f in files if f in keep]
        print(f"image list: {len(files)} of {len(keep)} listed images found")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    recs = []          # per detection: img, cls, conf, tp[10], gt_size, det_size
    gts = []           # per gt: img, cls, size bucket, matched_at_conf(set later)
    per_image = []
    for i, f in enumerate(files):
        stem = Path(f).stem
        img0 = cv2.imread(os.path.join(args.images, f))
        h0, w0 = img0.shape[:2]
        img, s, left, top = letterbox(img0, res)
        gt = load_gt(os.path.join(args.labels, stem + ".txt"), stem, w0, h0, s, left, top)
        det = infer(img)
        tp, g50 = match_image(det, gt)
        for d in range(len(det)):
            w, h = det[d, 2] - det[d, 0], det[d, 3] - det[d, 1]
            gb = size_bucket(gt[g50[d], 3] - gt[g50[d], 1], gt[g50[d], 4] - gt[g50[d], 2]) if g50[d] >= 0 else size_bucket(w, h)
            recs.append((i, int(det[d, 5]), float(det[d, 4]), tp[d], gb, int(g50[d])))
        for g in range(len(gt)):
            gts.append((i, int(gt[g, 0]), size_bucket(gt[g, 3] - gt[g, 1], gt[g, 4] - gt[g, 2])))
        per_image.append({"file": f, "gt": gt.tolist(), "det": (det if args.keep_all_dets else det[det[:, 4] >= 0.05]).tolist() if len(det) else []})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)

    conf = np.array([r[2] for r in recs]); dcls = np.array([r[1] for r in recs])
    TP = np.array([r[3] for r in recs]).reshape(-1, len(IOU_THRS))
    dsz = np.array([r[4] for r in recs])
    gcls = np.array([g[1] for g in gts]); gsz = np.array([g[2] for g in gts])

    result = {"split_images": len(files), "n_gt": {CLASS_NAMES[c]: int((gcls == c).sum()) for c in range(2)},
              "backend": backend, "checkpoint": str(args.checkpoint),
              "protocol": {"match_iou": 0.5, "ap_conf_floor": 0.001, "nms_iou": 0.6,
                           "operating_conf": args.conf, "ap_interp": "Ultralytics-style"},
              "per_class": {}, "pr_curves": {}}
    ap50s, ap5095s = [], []
    for c in range(2):
        m = dcls == c; ngt = int((gcls == c).sum())
        a50, rec, prec = ap_from(TP[m, 0], conf[m], ngt)
        aall = [ap_from(TP[m, t], conf[m], ngt)[0] for t in range(len(IOU_THRS))]
        a5095 = float(np.mean(aall))
        mo = m & (conf >= args.conf)
        tpn = int(TP[mo, 0].sum()); fpn = int(mo.sum() - tpn)
        P = tpn / max(tpn + fpn, 1); R = tpn / max(ngt, 1); F1 = 2 * P * R / max(P + R, 1e-9)
        result["per_class"][CLASS_NAMES[c]] = {"AP50": a50, "AP50_95": a5095, "AP75": aall[5],
                                               "P": P, "R": R, "F1": F1, "TP": tpn, "FP": fpn,
                                               "FN": ngt - tpn, "n_gt": ngt}
        if rec is not None:
            idx = np.linspace(0, len(rec) - 1, min(len(rec), 400)).astype(int)
            o = np.argsort(-conf[m])
            result["pr_curves"][CLASS_NAMES[c]] = {"recall": rec[idx].tolist(), "precision": prec[idx].tolist(),
                                                   "conf": conf[m][o][idx].tolist()}
        ap50s.append(a50); ap5095s.append(a5095)
    pc = result["per_class"]
    result["overall"] = {"mAP50": float(np.mean(ap50s)), "mAP50_95": float(np.mean(ap5095s)),
                         "P": float(np.mean([pc[k]["P"] for k in pc])), "R": float(np.mean([pc[k]["R"] for k in pc])),
                         "F1": float(np.mean([pc[k]["F1"] for k in pc]))}

    # size buckets (AP50 per class, pooled over classes for P/R at operating conf)
    sb = {}
    for b in SIZE_BUCKETS:
        row = {"n_gt": int((gsz == b).sum())}
        for c in range(2):
            m = (dcls == c) & (dsz == b); ngt = int(((gcls == c) & (gsz == b)).sum())
            a, _, _ = ap_from(TP[m, 0], conf[m], ngt)
            row[f"{CLASS_NAMES[c]}_AP50"] = a; row[f"{CLASS_NAMES[c]}_n_gt"] = ngt
        mo = (dsz == b) & (conf >= args.conf); tpn = int(TP[mo, 0].sum())
        row["P"] = tpn / max(int(mo.sum()), 1) if mo.sum() else float("nan")
        row["R"] = tpn / max(row["n_gt"], 1) if row["n_gt"] else float("nan")
        sb[b] = row
    result["size_breakdown"] = sb

    # confusion matrix (rows = predicted Bird/Drone/Background, cols = GT Bird/Drone/Background)
    cm = np.zeros((3, 3), int)
    by_img = {}
    for r in recs:
        if r[2] >= args.conf:
            by_img.setdefault(r[0], []).append(r)
    for i, it in enumerate(per_image):
        gt = np.array(it["gt"]).reshape(-1, 5)
        dets = np.array([d for d in it["det"] if d[4] >= args.conf]).reshape(-1, 6)
        used = np.zeros(len(gt), bool)
        if len(dets):
            iou = box_iou(dets[:, :4], gt[:, 1:5]) if len(gt) else np.zeros((len(dets), 0))
            for d in np.argsort(-dets[:, 4]):
                if len(gt):
                    cand = np.where((iou[d] >= 0.5) & ~used)[0]
                else:
                    cand = []
                if len(cand):
                    g = cand[np.argmax(iou[d, cand])]; used[g] = True
                    cm[int(dets[d, 5]), int(gt[g, 0])] += 1
                else:
                    cm[int(dets[d, 5]), 2] += 1
        for g in np.where(~used)[0]:
            cm[2, int(gt[g, 0])] += 1
    result["confusion_matrix"] = {"rows_pred": ["Bird", "Drone", "Background"],
                                  "cols_gt": ["Bird", "Drone", "Background"], "counts": cm.tolist()}

    # timing: batch 1, preprocess excluded, forward + decode + NMS included
    n = min(args.timing_images, len(files)); ts = []
    imgs = [letterbox(cv2.imread(os.path.join(args.images, f)), res)[0] for f in files[:n]]
    for _ in range(5):
        infer(imgs[0], conf_thr=args.conf)
    for x in imgs:
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter(); infer(x, conf_thr=args.conf)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    result["timing"] = {"device": str(dev) if dev.type != "cpu" else f"cpu ({os.cpu_count()} threads)",
                        "batch": 1, "includes": "normalise + forward + decode + NMS (letterbox excluded)",
                        "ms_mean": 1000 * float(np.mean(ts)), "ms_p50": 1000 * float(np.median(ts)),
                        "fps": 1.0 / float(np.mean(ts))}
    result["complexity"] = complexity()

    json.dump(result, open(out / "eval_v2.json", "w"), indent=2)
    json.dump(per_image, open(out / "per_image.json", "w"))
    print(json.dumps({k: result[k] for k in ["overall", "per_class", "size_breakdown", "confusion_matrix", "timing", "complexity"]}, indent=1))


if __name__ == "__main__":
    main()
