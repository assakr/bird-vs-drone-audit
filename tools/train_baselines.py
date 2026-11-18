"""
train_baselines.py — standard detector baselines on the SAME clean split.

Trains and evaluates (on the clean test split) the comparison models a reviewer
will ask for. Every run uses 640 px input and the same epoch budget.

  yolov8n        3.2 M params   (smallest standard YOLOv8)
  yolov8s       11.2 M params   (same size class as LCMS-DB, 11.8 M)
  yolov8s-p2    YOLOv8s with an extra P2 (stride-4) head — isolates the P2 idea
  yolo11s       current Ultralytics small model
  yolov8s-scratch  YOLOv8s with RANDOM initialisation (no COCO weights). Same training
                   regime as LCMS-DB (from scratch), so the LCMS-DB vs YOLOv8s gap can be
                   split into "pretraining" and "architecture". Not run by default:
                   python tools/train_baselines.py --data ... --models yolov8s-scratch --epochs 100

Requirements:  pip install ultralytics
Usage:         python tools/train_baselines.py --data "D:/PROGRAMMING/dr shimaa/Bird vs Drone kaggel/Dataset_clean/data_clean.yaml"
Output:        runs/baselines/<model>/...  and  runs/baselines/summary.csv
"""
import argparse, csv, time
from pathlib import Path

MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8s-p2": "yolov8s-p2.yaml",   # built from yaml, initialised from yolov8s weights below
    "yolo11s": "yolo11s.pt",
    "yolov8s-scratch": "yolov8s.yaml",  # random init, pretrained=False
}
DEFAULT_MODELS = ["yolov8n", "yolov8s", "yolov8s-p2", "yolo11s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS, choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=0, help="seed 0 keeps the original run names; other seeds add _s<seed>")
    a = ap.parse_args()

    from ultralytics import YOLO
    out = Path("runs/baselines").resolve(); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for base in a.models:
        name = base if a.seed == 0 else f"{base}_s{a.seed}"
        weights_path = out / name / "weights" / "best.pt"
        if not weights_path.exists():
            m = YOLO(MODELS[base])
            if base.endswith("-p2"):
                m = m.load("yolov8s.pt")       # transfer matching weights
            m.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                    project=str(out), name=name, exist_ok=True, seed=a.seed, deterministic=True, plots=True,
                    pretrained=not base.endswith("-scratch"))
            alt_dir = Path("runs/detect") / str(out).replace(":", "") / name
            alt_dir2 = Path("runs/detect") / "runs" / "baselines" / name
            if not weights_path.exists():
                import shutil
                if (alt_dir / "weights" / "best.pt").exists():
                    shutil.move(str(alt_dir), str(out / name))
                elif (alt_dir2 / "weights" / "best.pt").exists():
                    shutil.move(str(alt_dir2), str(out / name))
        best = YOLO(str(weights_path))
        r = best.val(data=a.data, split="test", imgsz=a.imgsz, batch=1, conf=0.001, iou=0.6,
                     device=a.device, project=str(out), name=f"{name}_test", exist_ok=True)
        ap50 = {best.names[i]: float(r.box.ap50[k]) for k, i in enumerate(r.box.ap_class_index)}
        ap5095 = {best.names[i]: float(r.box.ap[k]) for k, i in enumerate(r.box.ap_class_index)}
        params = sum(p.numel() for p in best.model.parameters()) / 1e6
        rows.append({"model": name, "params_M": round(params, 2),
                     "mAP50": round(100 * float(r.box.map50), 2), "mAP50_95": round(100 * float(r.box.map), 2),
                     "Bird_AP50": round(100 * ap50.get("Bird", 0), 2), "Drone_AP50": round(100 * ap50.get("Drone", 0), 2),
                     "Bird_AP50_95": round(100 * ap5095.get("Bird", 0), 2), "Drone_AP50_95": round(100 * ap5095.get("Drone", 0), 2),
                     "P": round(100 * float(r.box.mp), 2), "R": round(100 * float(r.box.mr), 2),
                     "ms_per_img_inference": round(r.speed["inference"], 2), "ms_per_img_nms": round(r.speed["postprocess"], 2)})
        print(rows[-1])
    # merge with an existing summary (rows of models not re-run are kept)
    old = []
    if (out / "summary.csv").exists():
        old = [r for r in csv.DictReader(open(out / "summary.csv")) if r["model"] not in {x["model"] for x in rows}]
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(old + rows)
    print("saved", out / "summary.csv")


if __name__ == "__main__":
    main()
