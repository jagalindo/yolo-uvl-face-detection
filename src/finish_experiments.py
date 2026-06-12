"""Resilient completion of the OFAT config sweep.

Reuses already-trained 50-epoch runs (re-validates their best.pt), trains only
the missing configurations, runs the FP32/FP16 precision axis, and writes the
summary CSV + LaTeX table INCREMENTALLY (crash-safe). Designed to survive the
intermittent MPS TaskAlignedAssigner bug: each config is isolated with retries,
and the 960 config runs at a smaller batch with MPS->CPU op fallback enabled.

Usage:  python src/finish_experiments.py --epochs 50
"""
import argparse
import csv
import os
import time
from pathlib import Path

# Allow unsupported MPS ops to fall back to CPU (mitigates assigner edge cases).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset" / "My_YOLO_Dataset" / "data.yaml"
WEIGHTS_DIR = ROOT / "models" / "weights"
RUNS_DIR = ROOT / "runs" / "experiments"
SUMMARY_CSV = RUNS_DIR / "summary.csv"
SUMMARY_TEX = RUNS_DIR / "summary_table.tex"

FIELDS = ["config", "axis", "model", "imgsz", "augment", "precision",
          "epochs", "train_min", "P", "R", "mAP50", "mAP50_95", "latency_ms", "status"]

CONFIGS = [
    {"name": "base_s_640_aug",  "model": "yolo11s.pt", "imgsz": 640, "augment": True,  "batch": 16, "axis": "baseline"},
    {"name": "model_n_640_aug", "model": "yolo11n.pt", "imgsz": 640, "augment": True,  "batch": 16, "axis": "model_size"},
    {"name": "model_m_640_aug", "model": "yolo11m.pt", "imgsz": 640, "augment": True,  "batch": 16, "axis": "model_size"},
    {"name": "noaug_s_640",     "model": "yolo11s.pt", "imgsz": 640, "augment": False, "batch": 16, "axis": "augment"},
    {"name": "input_s_960_aug", "model": "yolo11s.pt", "imgsz": 960, "augment": True,  "batch": 8,  "axis": "input_size"},
]


def device_str():
    return "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")


def ensure_dataset_path():
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str(DATA_YAML.parent)
    with open(DATA_YAML, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def completed(name, epochs):
    rc = RUNS_DIR / name / "results.csv"
    bw = RUNS_DIR / name / "weights" / "best.pt"
    if not (rc.exists() and bw.exists()):
        return False
    n = sum(1 for _ in open(rc)) - 1
    return n >= epochs


def val_metrics(weights, imgsz, device, half=False, name="val"):
    m = YOLO(str(weights)).val(
        data=str(DATA_YAML), imgsz=imgsz, device=device, half=half, verbose=False,
        project=str(RUNS_DIR), name=name, exist_ok=True, plots=False)
    speed = getattr(m, "speed", {}) or {}
    lat = round(sum(v for k, v in speed.items() if k in ("inference", "preprocess", "postprocess")), 2)
    b = m.box
    return dict(P=round(b.mp, 4), R=round(b.mr, 4),
                mAP50=round(b.map50, 4), mAP50_95=round(b.map, 4), latency_ms=lat)


def aug_kwargs(augment):
    return {} if augment else dict(mosaic=0.0, mixup=0.0, hsv_h=0.0, hsv_s=0.0,
                                   hsv_v=0.0, translate=0.0, scale=0.0, fliplr=0.0, erasing=0.0)


def append_row(row):
    new = not SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def run_config(cfg, epochs, device, retries=2):
    weights = WEIGHTS_DIR / cfg["model"]
    weights = str(weights) if weights.exists() else cfg["model"]
    run_dir = RUNS_DIR / cfg["name"]

    if completed(cfg["name"], epochs):
        print(f"  reuse existing 50-epoch run, re-validating best.pt")
        met = val_metrics(run_dir / "weights" / "best.pt", cfg["imgsz"], device,
                          name=cfg["name"] + "_val")
        return {**base_row(cfg, epochs, 0.0, "reused"), **met}

    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            YOLO(weights).train(
                data=str(DATA_YAML), epochs=epochs, imgsz=cfg["imgsz"], batch=cfg["batch"],
                device=device, project=str(RUNS_DIR), name=cfg["name"], patience=0,
                verbose=False, plots=True, exist_ok=True, **aug_kwargs(cfg["augment"]))
            tmin = round((time.time() - t0) / 60, 1)
            met = val_metrics(run_dir / "weights" / "best.pt", cfg["imgsz"], device,
                              name=cfg["name"] + "_val")
            return {**base_row(cfg, epochs, tmin, "ok"), **met}
        except Exception as e:
            last_err = str(e)[:160]
            print(f"  attempt {attempt}/{retries} FAILED: {last_err}")
    return {**base_row(cfg, epochs, 0.0, f"FAILED: {last_err}")}


def base_row(cfg, epochs, tmin, status):
    return dict(config=cfg["name"], axis=cfg["axis"], model=cfg["model"], imgsz=cfg["imgsz"],
                augment=cfg["augment"], precision="FP32", epochs=epochs, train_min=tmin, status=status)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()
    device = device_str()
    print(f"Device: {device} | MPS fallback: {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")
    ensure_dataset_path()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if SUMMARY_CSV.exists():
        SUMMARY_CSV.unlink()  # fresh aggregate

    base_weights_path = None
    for cfg in CONFIGS:
        print(f"\n===== {cfg['name']} ({cfg['axis']}) =====")
        row = run_config(cfg, args.epochs, device)
        append_row(row)
        print(f"  -> {row.get('status')}: P={row.get('P')} R={row.get('R')} "
              f"mAP50={row.get('mAP50')} mAP50-95={row.get('mAP50_95')}")
        if cfg["axis"] == "baseline" and row.get("status") in ("ok", "reused"):
            base_weights_path = RUNS_DIR / cfg["name"] / "weights" / "best.pt"

    # Precision axis: FP32 vs FP16 inference on the baseline model.
    if base_weights_path and base_weights_path.exists():
        for half in (False, True):
            label = "FP16" if half else "FP32"
            print(f"\n===== precision_{label.lower()} (inference) =====")
            try:
                met = val_metrics(base_weights_path, 640, device, half=half, name=f"prec_{label.lower()}")
                row = dict(config=f"precision_{label.lower()}", axis="precision", model="yolo11s.pt",
                           imgsz=640, augment=True, precision=label, epochs=args.epochs,
                           train_min=0.0, status="ok", **met)
            except Exception as e:
                row = dict(config=f"precision_{label.lower()}", axis="precision", model="yolo11s.pt",
                           imgsz=640, augment=True, precision=label, epochs=args.epochs,
                           train_min=0.0, status=f"FAILED: {str(e)[:120]}")
            append_row(row)
            print(f"  -> {row.get('status')}: mAP50={row.get('mAP50')}")

    # Emit a LaTeX table body from the CSV.
    rows = list(csv.DictReader(open(SUMMARY_CSV)))
    with open(SUMMARY_TEX, "w") as f:
        for r in rows:
            if not r["P"]:
                continue
            f.write(f"{r['axis']} & {r['config']} & {r['model']} & {r['imgsz']} & "
                    f"{r['precision']} & {r['P']} & {r['R']} & {r['mAP50']} & {r['mAP50_95']} \\\\\n")
    print(f"\nDONE. Summary: {SUMMARY_CSV}\nLaTeX body: {SUMMARY_TEX}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
