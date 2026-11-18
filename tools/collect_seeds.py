"""collect_seeds.py - mean ± std over repeated runs (rev. 4 multi-seed study).

Groups every run folder whose name is <base>, <base>_s1, <base>_s2, ... and reports
mean ± sample std of the test metrics written by tools/evaluate_v2.py.
Output: runs/SEEDS_TABLE.md and runs/seeds_summary.json
"""
import json, re, statistics as st
from pathlib import Path

RUNS = Path("runs")
GROUPS = {  # base name -> folder that holds eval_v2.json, relative to the run folder
    "A5_full_ciou": "eval_v2_test", "A6_full_wiou": "eval_v2_test", "A7_full_wiou_scaleaware": "eval_v2_test",
    "final_full_budget": "eval_v2_test",
}
METRICS = [("mAP50", lambda r: r["overall"]["mAP50"]), ("mAP50_95", lambda r: r["overall"]["mAP50_95"]),
           ("Bird_AP50", lambda r: r["per_class"]["Bird"]["AP50"]), ("Drone_AP50", lambda r: r["per_class"]["Drone"]["AP50"])]


def runs_for(base, sub, parent=RUNS, suffix=""):
    out = []
    for d in sorted(parent.iterdir()):
        m = re.fullmatch(re.escape(base) + r"(_s\d+)?" + re.escape(suffix), d.name)
        f = d / sub / "eval_v2.json" if sub else d / "eval_v2.json"
        if m and f.exists():
            out.append((d.name, json.load(open(f))))
    return out


def fmt(vals):
    vals = [100 * v for v in vals]
    if len(vals) == 1:
        return f"{vals[0]:.1f} (n=1)"
    return f"{st.mean(vals):.1f} ± {st.stdev(vals):.1f}"


rows, summary = [], {}
for base, sub in GROUPS.items():
    rs = runs_for(base, sub)
    if rs:
        summary[base] = {k: [f(r) for _, r in rs] for k, f in METRICS} | {"runs": [n for n, _ in rs]}
        rows.append(f"| {base} | {len(rs)} | " + " | ".join(fmt(summary[base][k]) for k, _ in METRICS) + " |")
# YOLOv8s from scratch: runs/baselines/yolov8s-scratch[_sN]_evalv2/eval_v2.json
rs = runs_for("yolov8s-scratch", None, RUNS / "baselines", "_evalv2") if (RUNS / "baselines").exists() else []
if rs:
    summary["yolov8s-scratch"] = {k: [f(r) for _, r in rs] for k, f in METRICS} | {"runs": [n for n, _ in rs]}
    rows.append(f"| yolov8s-scratch | {len(rs)} | " + " | ".join(fmt(summary["yolov8s-scratch"][k]) for k, _ in METRICS) + " |")

lines = ["## Multi-seed results (test split, mean ± sample std over runs)", "",
         "| Config | Runs | mAP50 | mAP50-95 | Bird AP50 | Drone AP50 |", "|---|---|---|---|---|---|"] + rows
if "A6_full_wiou" in summary and "A7_full_wiou_scaleaware" in summary:
    a, b = summary["A6_full_wiou"]["mAP50"], summary["A7_full_wiou_scaleaware"]["mAP50"]
    lines += ["", f"A7 - A6 (assignment effect), mAP50: {100 * (st.mean(b) - st.mean(a)):+.1f} points "
                  f"(min A7 {100 * min(b):.1f} vs max A6 {100 * max(a):.1f})"]
if "A5_full_ciou" in summary and "A6_full_wiou" in summary:
    a, b = summary["A5_full_ciou"]["mAP50"], summary["A6_full_wiou"]["mAP50"]
    lines += [f"A6 - A5 (Wise-IoU vs CIoU), mAP50: {100 * (st.mean(b) - st.mean(a)):+.1f} points"]
(RUNS / "SEEDS_TABLE.md").write_text("\n".join(lines) + "\n")
json.dump(summary, open(RUNS / "seeds_summary.json", "w"), indent=1)
print("\n".join(lines))
