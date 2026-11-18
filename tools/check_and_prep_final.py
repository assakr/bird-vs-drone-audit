"""check_and_prep_final.py — Compare A7 vs A6 and adjust final_full_budget.yaml if needed."""
import json
import sys
from pathlib import Path

a6_file = Path("runs/A6_full_wiou/eval_v2_test/eval_v2.json")
a7_file = Path("runs/A7_full_wiou_scaleaware/eval_v2_test/eval_v2.json")

if not a6_file.exists() or not a7_file.exists():
    print(f"[check_and_prep_final] Warning: {a6_file} or {a7_file} missing. Cannot compare A7 vs A6.")
    sys.exit(0)

a6 = json.load(open(a6_file))
a7 = json.load(open(a7_file))

map50_6 = a6["overall"]["mAP50"]
map50_7 = a7["overall"]["mAP50"]
map_6 = a6["overall"]["mAP50_95"]
map_7 = a7["overall"]["mAP50_95"]

print("=" * 60)
print(f"A6_full_wiou:            mAP@0.5 = {map50_6*100:.2f}%, mAP@0.5:0.95 = {map_6*100:.2f}%")
print(f"A7_full_wiou_scaleaware: mAP@0.5 = {map50_7*100:.2f}%, mAP@0.5:0.95 = {map_7*100:.2f}%")
print("=" * 60)

cfg_path = Path("configs/final_full_budget.yaml")
cfg_text = cfg_path.read_text(encoding="utf-8")

if map50_7 > map50_6 or (map50_7 == map50_6 and map_7 > map_6):
    print(">>> DECISION: A7 WINS against A6 on the clean test split!")
    print(">>> Retaining scale-aware assigner in configs/final_full_budget.yaml.")
else:
    print(">>> DECISION: A7 did NOT beat A6 on the clean test split.")
    print(">>> Reverting configs/final_full_budget.yaml to original assigner as instructed.")
    lines = cfg_text.splitlines()
    new_lines = []
    for line in lines:
        s = line.strip()
        if s.startswith("center_decode:"):
            continue
        if s == "model:":
            continue
        if any(s.startswith(k) for k in ["assigner:", "center_radius:", "level_ranges:", "obj_iou_target:"]):
            continue
        new_lines.append(line)
    cfg_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(">>> Successfully updated configs/final_full_budget.yaml (original assigner restored).")
