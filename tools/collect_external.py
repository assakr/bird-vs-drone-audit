"""collect_external.py - Kaggle test vs external test for every model (rev. 5).
Run from code/lcms_db after tools/run_external_test.bat. Writes runs/EXTERNAL_TABLE.md."""
import json
from pathlib import Path

RUNS = Path("runs")
MODELS = [("LCMS-DB (final)", RUNS / "final_full_budget" / "eval_v2_test", RUNS / "final_full_budget" / "eval_external"),
          ("LCMS-DB A7", RUNS / "A7_full_wiou_scaleaware" / "eval_v2_test", RUNS / "A7_full_wiou_scaleaware" / "eval_external")]
for n, label in [("yolov8s-scratch", "YOLOv8s (scratch)"), ("yolov8s", "YOLOv8s (COCO)"), ("yolov8n", "YOLOv8n (COCO)"),
                 ("yolov8s-p2", "YOLOv8s-P2 (COCO)"), ("yolo11s", "YOLO11s (COCO)")]:
    MODELS.append((label, RUNS / "baselines" / f"{n}_evalv2", RUNS / "baselines" / f"{n}_external"))


def get(d):
    f = d / "eval_v2.json"
    if not f.exists():
        return None
    r = json.load(open(f)); o, pc = r["overall"], r["per_class"]
    fp = pc["Bird"]["FP"] + pc["Drone"]["FP"]
    return [100 * o["mAP50"], 100 * o["mAP50_95"], 100 * pc["Bird"]["AP50"], 100 * pc["Drone"]["AP50"],
            100 * pc["Bird"]["R"], 100 * pc["Drone"]["R"], fp], r["split_images"], r["n_gt"]


out = ["## Kaggle test split (scenes also present in training) vs external test set (other source)", "",
       "| Model | Test set | Images | mAP50 | mAP50-95 | Bird AP50 | Drone AP50 | Bird R@0.25 | Drone R@0.25 | FP@0.25 |",
       "|---|---|---|---|---|---|---|---|---|---|"]
drops = []
for label, kd, ed in MODELS:
    k, e = get(kd), get(ed)
    for name, x in (("Kaggle test", k), ("External", e)):
        if x:
            v, n, _ = x
            out.append(f"| {label} | {name} | {n} | " + " | ".join(f"{t:.1f}" for t in v[:6]) + f" | {v[6]} |")
    if k and e:
        drops.append(f"| {label} | {k[0][0]:.1f} | {e[0][0]:.1f} | {e[0][0] - k[0][0]:+.1f} |")
if drops:
    out += ["", "## Change in mAP50 from the Kaggle test split to the external set", "",
            "| Model | Kaggle test | External | Change |", "|---|---|---|---|"] + drops
(RUNS / "EXTERNAL_TABLE.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
