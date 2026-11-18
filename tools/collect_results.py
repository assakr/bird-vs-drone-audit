"""collect_results.py — turn every eval_v2_test/eval_v2.json + baseline summary into paper-ready tables."""
import csv, json
from pathlib import Path

RUNS = Path("runs")
ORDER = ["A0_plain", "A1_p2", "A2_p2_msfe", "A3_p2_msfe_cbam", "A4_p2_msfe_cbam_bff",
         "A5_full_ciou", "A6_full_wiou", "A7_full_wiou_scaleaware", "final_full_budget", "v2_full"]


def row(name, r):
    pc, o, c, t = r["per_class"], r["overall"], r["complexity"], r["timing"]
    return (f"| {name} | {c['params_M']:.2f} | {(c.get('GFLOPs') or float('nan')):.1f} | {100*o['mAP50']:.1f} | "
            f"{100*o['mAP50_95']:.1f} | {100*pc['Bird']['AP50']:.1f} | {100*pc['Drone']['AP50']:.1f} | "
            f"{100*pc['Drone']['R']:.1f} | {pc['Drone']['FP']+pc['Bird']['FP']} | {t['ms_mean']:.1f} ({t['device']}) |")


lines = ["## LCMS-DB variants (clean test split, evaluate_v2.py)", "",
         "| Run | Params (M) | GFLOPs | mAP50 | mAP50-95 | Bird AP50 | Drone AP50 | Drone R@0.25 | FP@0.25 | ms/img |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for n in ORDER:
    for sub in ["eval_v2_test", "eval_v2"]:
        f = RUNS / n / sub / "eval_v2.json"
        if f.exists():
            lines.append(row(n, json.load(open(f)))); break
# Baselines re-scored with tools/evaluate_v2.py --yolo (identical protocol to the LCMS-DB rows above)
BASE = ["yolov8n", "yolov8s", "yolov8s-p2", "yolo11s", "yolov8s-scratch"]
ev = [(n, RUNS / "baselines" / f"{n}_evalv2" / "eval_v2.json") for n in BASE]
ev = [(n, f) for n, f in ev if f.exists()]
if ev:
    lines += ["", "## Baselines scored with evaluate_v2.py --yolo (same protocol and code as the LCMS-DB rows)", "",
              lines[2], lines[3]]
    for n, f in ev:
        lines.append(row(n + (" (COCO-pretrained)" if not n.endswith("scratch") else " (from scratch)"), json.load(open(f))))

# Rev. 4: same models on test images that have no same-scene training image
un = [(n, RUNS / n / "eval_unseen" / "eval_v2.json") for n in ["final_full_budget", "A7_full_wiou_scaleaware"]]
un += [(n, RUNS / "baselines" / f"{n}_unseen" / "eval_v2.json") for n in BASE]
un = [(n, f) for n, f in un if f.exists()]
if un:
    k = json.load(open(un[0][1]))["split_images"]
    lines += ["", f"## Leakage-free test subset ({k} test images with no same-scene training image; tools/scene_overlap.py)", "",
              lines[2], lines[3]]
    for n, f in un:
        lines.append(row(n, json.load(open(f))))

b = RUNS / "baselines" / "summary.csv"
if b.exists():
    lines += ["", "## Baselines (Ultralytics val, conf 0.001, IoU 0.6, clean test split)", "",
              "| Model | Params (M) | mAP50 | mAP50-95 | Bird AP50 | Drone AP50 | P | R | ms/img |", "|---|---|---|---|---|---|---|---|---|"]
    for r in csv.DictReader(open(b)):
        lines.append(f"| {r['model']} | {r['params_M']} | {r['mAP50']} | {r['mAP50_95']} | {r['Bird_AP50']} | "
                     f"{r['Drone_AP50']} | {r['P']} | {r['R']} | {r['ms_per_img_inference']} |")
(RUNS / "RESULTS_TABLES.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
