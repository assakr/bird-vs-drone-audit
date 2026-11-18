"""
bootstrap_ci.py - 95% confidence intervals for mAP by resampling test images (rev. 4).

Every model is scored on the SAME bootstrap resamples, so differences between models get
paired intervals. Input: per_image.json files written by
    python tools/evaluate_v2.py ... --keep-all-dets      (LCMS-DB or --yolo models)

  python tools/bootstrap_ci.py --runs final=runs/final_full_budget/eval_all/per_image.json ^
        yolov8s-scratch=runs/baselines/yolov8s-scratch_evalall/per_image.json ^
        --compare final:yolov8s-scratch --n 1000

Output: runs/bootstrap_ci.json and a markdown table on screen. AP is computed exactly as in
evaluate_v2.py (same matching and interpolation); with no resampling the point estimates
reproduce evaluate_v2.py.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_v2 import match_image, IOU_THRS

X = np.linspace(0, 1, 101)


def prepare(path):
    """Per class: detections sorted by confidence with their image index and TP flags (all IoU thresholds)."""
    PI = json.load(open(path))
    names = [it["file"] for it in PI]
    dets = {0: [], 1: []}; ngt = np.zeros((len(PI), 2))
    for i, it in enumerate(PI):
        det = np.array(it["det"], np.float32).reshape(-1, 6); gt = np.array(it["gt"], np.float32).reshape(-1, 5)
        tp, _ = match_image(det, gt)
        for c in (0, 1):
            k = det[:, 5] == c
            dets[c] += [(float(det[d, 4]), i, tp[d]) for d in np.where(k)[0]]
            ngt[i, c] = (gt[:, 0] == c).sum()
    out = {"names": names, "ngt": ngt}
    for c in (0, 1):
        dets[c].sort(key=lambda x: -x[0])
        out[c] = (np.array([d[1] for d in dets[c]], int), np.array([d[2] for d in dets[c]], bool).reshape(-1, len(IOU_THRS)))
    return out


def weighted_ap(img_idx, tp, w, n_gt):
    """AP at every IoU threshold for detections already sorted by confidence; w = image weights."""
    if n_gt <= 0:
        return np.full(tp.shape[1], np.nan)
    if len(img_idx) == 0:
        return np.zeros(tp.shape[1])
    wd = w[img_idx][:, None]
    tpc = np.cumsum(wd * tp, 0); fpc = np.cumsum(wd * ~tp, 0)
    rec = tpc / n_gt; prec = tpc / np.maximum(tpc + fpc, 1e-12)
    aps = []
    for t in range(tp.shape[1]):
        nz = wd[:, 0] > 0                      # drop detections from images not drawn
        r, p = rec[nz, t], prec[nz, t]
        mpre = np.concatenate([[1.0], p, [0.0]]); mrec = np.concatenate([[0.0], r, [1.0]])
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        aps.append(np.trapezoid(np.interp(X, mrec, mpre), X) if hasattr(np, "trapezoid") else np.trapz(np.interp(X, mrec, mpre), X))
    return np.array(aps)


def score(P, w):
    a50, a5095, per = [], [], {}
    for c, cn in ((0, "Bird"), (1, "Drone")):
        ap = weighted_ap(*P[c], w, float((w * P["ngt"][:, c]).sum()))
        a50.append(ap[0]); a5095.append(np.nanmean(ap)); per[cn] = ap[0]
    return {"mAP50": float(np.mean(a50)), "mAP50_95": float(np.mean(a5095)), "Bird_AP50": per["Bird"], "Drone_AP50": per["Drone"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=path/to/per_image.json")
    ap.add_argument("--compare", nargs="*", default=[], help="pairs a:b -> CI of (a - b)")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/bootstrap_ci.json")
    a = ap.parse_args()
    runs = dict(r.split("=", 1) for r in a.runs)
    P = {k: prepare(v) for k, v in runs.items()}
    names = list(P.values())[0]["names"]
    for k, p in P.items():
        assert p["names"] == names, f"{k}: per_image.json must cover the same images in the same order"
    n_img = len(names); rng = np.random.default_rng(a.seed)
    point = {k: score(p, np.ones(n_img)) for k, p in P.items()}
    samples = {k: [] for k in P}
    for b in range(a.n):
        w = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img).astype(float)
        for k, p in P.items():
            samples[k].append(score(p, w))
        if (b + 1) % 100 == 0:
            print(f"  {b + 1}/{a.n}", flush=True)
    res = {"n_images": n_img, "n_boot": a.n, "models": {}, "differences": {}}
    for k in P:
        res["models"][k] = {m: {"point": point[k][m], "ci95": list(np.percentile([s[m] for s in samples[k]], [2.5, 97.5]))}
                            for m in point[k]}
    for pair in a.compare:
        x, y = pair.split(":")
        res["differences"][pair] = {}
        for m in point[x]:
            d = np.array([sx[m] - sy[m] for sx, sy in zip(samples[x], samples[y])])
            res["differences"][pair][m] = {"point": point[x][m] - point[y][m], "ci95": list(np.percentile(d, [2.5, 97.5])),
                                           "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    pc = lambda v: f"{100 * v:.1f}"
    print(f"\n| Model | mAP50 [95% CI] | mAP50-95 [95% CI] |\n|---|---|---|")
    for k, r in res["models"].items():
        print(f"| {k} | {pc(r['mAP50']['point'])} [{pc(r['mAP50']['ci95'][0])}, {pc(r['mAP50']['ci95'][1])}] | "
              f"{pc(r['mAP50_95']['point'])} [{pc(r['mAP50_95']['ci95'][0])}, {pc(r['mAP50_95']['ci95'][1])}] |")
    for pair, r in res["differences"].items():
        d = r["mAP50"]; e = r["mAP50_95"]
        print(f"{pair}: mAP50 diff {pc(d['point'])} [{pc(d['ci95'][0])}, {pc(d['ci95'][1])}] p={d['p_two_sided']:.3f}; "
              f"mAP50-95 diff {pc(e['point'])} [{pc(e['ci95'][0])}, {pc(e['ci95'][1])}] p={e['p_two_sided']:.3f}")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
