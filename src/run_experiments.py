"""One-factor-at-a-time configuration sweep for the UVL->YOLOv11 study.

Trains a baseline config and a set of single-feature variations on the face
dataset, evaluates each on the validation split, and writes a combined results
table (CSV + LaTeX). The precision axis is evaluated at inference time
(half=True vs half=False) because MPS cannot train in FP16.

Usage:
    python src/run_experiments.py --epochs 50            # full sweep
    python src/run_experiments.py --epochs 2 --only base # quick smoke test
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset" / "My_YOLO_Dataset" / "data.yaml"
WEIGHTS_DIR = ROOT / "models" / "weights"
RUNS_DIR = ROOT / "runs" / "experiments"
RESULTS_CSV = ROOT / "runs" / "experiments" / "summary.csv"


def device_str() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def ensure_dataset_path(yaml_path: Path) -> None:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str(yaml_path.parent)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


# One-factor-at-a-time configurations around the baseline.
# Each: name, base weights, imgsz, augment(on/off). Precision handled separately.
CONFIGS = [
    {"name": "base_s_640_aug",  "model": "yolo11s.pt", "imgsz": 640, "augment": True,  "axis": "baseline"},
    {"name": "model_n_640_aug", "model": "yolo11n.pt", "imgsz": 640, "augment": True,  "axis": "model_size"},
    {"name": "model_m_640_aug", "model": "yolo11m.pt", "imgsz": 640, "augment": True,  "axis": "model_size"},
    {"name": "input_s_960_aug", "model": "yolo11s.pt", "imgsz": 960, "augment": True,  "axis": "input_size"},
    {"name": "noaug_s_640",     "model": "yolo11s.pt", "imgsz": 640, "augment": False, "axis": "augment"},
]


def train_and_eval(cfg, epochs, device):
    # Prefer local weights if present, else let Ultralytics fetch them.
    local_w = WEIGHTS_DIR / cfg["model"]
    weights = str(local_w) if local_w.exists() else cfg["model"]

    aug = {} if cfg["augment"] else dict(
        mosaic=0.0, mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        translate=0.0, scale=0.0, fliplr=0.0, erasing=0.0,
    )

    model = YOLO(weights)
    t0 = time.time()
    model.train(
        data=str(DATA_YAML), epochs=epochs, imgsz=cfg["imgsz"], batch=16,
        device=device, project=str(RUNS_DIR), name=cfg["name"],
        patience=0, verbose=False, plots=True, exist_ok=True, **aug,
    )
    train_s = time.time() - t0

    m = model.val(data=str(DATA_YAML), imgsz=cfg["imgsz"], device=device,
                  verbose=False, project=str(RUNS_DIR), name=cfg["name"] + "_val",
                  exist_ok=True).box
    return {
        "config": cfg["name"], "axis": cfg["axis"], "model": cfg["model"],
        "imgsz": cfg["imgsz"], "augment": cfg["augment"], "precision": "FP32",
        "epochs": epochs, "train_min": round(train_s / 60, 1),
        "P": round(m.mp, 4), "R": round(m.mr, 4),
        "mAP50": round(m.map50, 4), "mAP50_95": round(m.map, 4),
    }, model


def eval_precision(model, imgsz, device, half):
    """Inference-time precision comparison on the trained baseline."""
    try:
        m = model.val(data=str(DATA_YAML), imgsz=imgsz, device=device, half=half,
                      verbose=False, project=str(RUNS_DIR),
                      name=f"prec_{'fp16' if half else 'fp32'}", exist_ok=True).box
        return {"P": round(m.mp, 4), "R": round(m.mr, 4),
                "mAP50": round(m.map50, 4), "mAP50_95": round(m.map, 4), "ok": True}
    except Exception as e:
        return {"error": str(e)[:200], "ok": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--only", default=None, help="run a single config by name (e.g. base)")
    args = ap.parse_args()

    device = device_str()
    print(f"Device: {device} | epochs: {args.epochs}")
    ensure_dataset_path(DATA_YAML)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    configs = CONFIGS
    if args.only:
        configs = [c for c in CONFIGS if c["name"].startswith(args.only)]

    rows = []
    baseline_model = None
    for cfg in configs:
        print(f"\n===== {cfg['name']} ({cfg['axis']}) =====")
        row, model = train_and_eval(cfg, args.epochs, device)
        print(f"  -> P={row['P']} R={row['R']} mAP50={row['mAP50']} "
              f"mAP50-95={row['mAP50_95']} ({row['train_min']} min)")
        rows.append(row)
        if cfg["axis"] == "baseline":
            baseline_model = model

    # Precision axis: FP16 vs FP32 inference on the baseline model.
    if baseline_model is not None and not args.only:
        print("\n===== precision axis (inference) =====")
        for half in (False, True):
            res = eval_precision(baseline_model, 640, device, half)
            label = "FP16" if half else "FP32"
            if res["ok"]:
                print(f"  {label}: mAP50={res['mAP50']} R={res['R']}")
                rows.append({
                    "config": f"precision_{label.lower()}", "axis": "precision",
                    "model": "yolo11s.pt", "imgsz": 640, "augment": True,
                    "precision": label, "epochs": args.epochs, "train_min": 0.0,
                    "P": res["P"], "R": res["R"], "mAP50": res["mAP50"], "mAP50_95": res["mAP50_95"],
                })
            else:
                print(f"  {label}: FAILED on {device} -> {res['error']}")

    # Write CSV
    if rows:
        with open(RESULTS_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSummary written to {RESULTS_CSV}")
        for r in rows:
            print(r)


if __name__ == "__main__":
    main()
