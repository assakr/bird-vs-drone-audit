"""collect_rev5.py - tables for the rev. 5 runs (architecture ablation under the scale-aware
assignment, and the decomposition of that assignment). Run from code/lcms_db.
Writes runs/REV5_TABLES.md. Runs that are not finished yet are skipped."""
import json
from pathlib import Path

RUNS = Path("runs")
PAIRS = [("CSP + FPN, P3-P5, CIoU", "A0_plain", "B0_plain_sa"),
         ("+ P2 level", "A1_p2", "B1_p2_sa"),
         ("+ MSFE", "A2_p2_msfe", "B2_p2_msfe_sa"),
         ("+ CBAM", "A3_p2_msfe_cbam", "B3_p2_msfe_cbam_sa"),
         ("+ BFF", "A4_p2_msfe_cbam_bff", "B4_p2_msfe_cbam_bff_sa"),
         ("+ P5 transformer", "A5_full_ciou", "B5_full_ciou_sa"),
         ("+ Wise-IoU", "A6_full_wiou", "A7_full_wiou_scaleaware")]
DECOMP = [("Original assigner (A6)", "A6_full_wiou"),
          ("Scale gating only: one cell per object (C2)", "C2_sa_single_cell"),
          ("Multi-positive only: all levels (C3)", "C3_sa_all_levels"),
          ("Gating + multi-positive, objectness target 1.0 (C1)", "C1_sa_obj1"),
          ("Full scale-aware assignment (A7)", "A7_full_wiou_scaleaware")]


def load(name):
    f = RUNS / name / "eval_v2_test" / "eval_v2.json"
    return json.load(open(f)) if f.exists() else None


def cells(r):
    if r is None:
        return ["-"] * 5
    o, pc, sb = r["overall"], r["per_class"], r["size_breakdown"]
    sd = sb["small"]["Drone_AP50"]
    return [f"{100*o['mAP50']:.1f}", f"{100*o['mAP50_95']:.1f}", f"{100*pc['Bird']['AP50']:.1f}",
            f"{100*pc['Drone']['AP50']:.1f}", f"{100*sd:.1f}"]


H = "mAP50 | mAP50-95 | Bird AP50 | Drone AP50 | Small-drone AP50"
out = ["## Architecture ablation under both assignment rules (clean test split)", "",
       f"| Change | Rule | {H} |", "|---|---|---|---|---|---|---|"]
for desc, a, b in PAIRS:
    out.append(f"| {desc} | original ({a}) | " + " | ".join(cells(load(a))) + " |")
    out.append(f"| | scale-aware ({b}) | " + " | ".join(cells(load(b))) + " |")
out += ["", "## Decomposition of the scale-aware assignment (full architecture, Wise-IoU)", "",
        f"| Variant | {H} |", "|---|---|---|---|---|---|"]
for desc, n in DECOMP:
    out.append(f"| {desc} | " + " | ".join(cells(load(n))) + " |")
(RUNS / "REV5_TABLES.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
