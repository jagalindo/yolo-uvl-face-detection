"""CLI: validate a UVL feature selection, export it as a JSON artifact, and
evaluate it with REAL metrics (live model.val on the matching trained weights,
falling back to the recorded experiment summary). No metrics are fabricated.

Run:  python src/parse_uvl.py                 # live eval, recorded fallback
      python src/parse_uvl.py --source recorded   # rely on recorded paper results
"""
import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO

import uvl_config as uvl

ROOT = Path(__file__).resolve().parent.parent
UVL_FILE = ROOT / "models" / "yolo_custom_model.uvl"
DATA_YAML = ROOT / "dataset" / "My_YOLO_Dataset" / "data.yaml"
JSON_OUT = ROOT / "results" / "validated_config.json"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def print_uvl_summary():
    print("=" * 58)
    print(f"📖 UVL feature model: {UVL_FILE.name}")
    try:
        import analyze_uvl
        s = analyze_uvl.analyze()
        print(f"  dimensions : {len(s['dimensions'])} -> {', '.join(s['dimensions'])}")
        print(f"  features   : {s['n_features']} ({s['n_leaves']} leaves), max depth {s['max_depth']}")
        print(f"  satisfiable: {s['satisfiable']}, dead features: {s['n_dead']}")
        print(f"  VALID CONFIGURATIONS: {s['n_configs']:,}")
    except Exception as e:
        print(f"  (flamapy analysis unavailable: {e})")
    print("=" * 58)


def run_evaluation(selected, source="live"):
    print_uvl_summary()

    ok, errors = uvl.validate_configuration(selected)
    print("\n🔍 Validating against cross-tree constraints...")
    if not ok:
        for e in errors:
            print(f"  ❌ {e}")
        print("\n❌ Configuration is UNSATISFIABLE — evaluation aborted.")
        return
    print("  ✅ SATISFIABLE — all constraints hold.")

    path = uvl.export_json(selected, JSON_OUT)
    print(f"💾 Exported validated JSON artifact: {path}")

    args = uvl.config_to_yolo_args(selected)
    print(f"⚙️  Translated args: {args}")

    # Real evaluation: live val on the matching trained weights, unless the
    # user asked to rely on the recorded paper results.
    weights = uvl.trained_weights_for(selected)
    metrics = None
    if source == "live" and weights is not None and DATA_YAML.exists():
        try:
            print(f"🚀 Live model.val() on {weights.parent.parent.name}/best.pt ...")
            b = YOLO(str(weights)).val(
                data=str(DATA_YAML), imgsz=args["imgsz"], half=args["half"],
                device=DEVICE, verbose=False, max_det=args["max_det"], workers=0, plots=False).box
            metrics = {"P": b.mp, "R": b.mr, "mAP50": b.map50, "mAP50_95": b.map}
            source = "live model.val()"
        except Exception as e:
            print(f"  (live val unavailable: {str(e)[:90]})")
    if metrics is None:
        m, cfg, note = uvl.measured_metrics(selected)
        if m is None:
            print(f"⚠️ No measured results: {note}")
            return
        metrics, source = m, f"recorded summary.csv [{cfg}] ({note})"

    f1 = (2 * metrics["P"] * metrics["R"] / (metrics["P"] + metrics["R"])
          if metrics["P"] + metrics["R"] > 0 else 0.0)
    print(f"\n📊 Results ({source}):")
    print(f"   P={metrics['P']:.3f}  R={metrics['R']:.3f}  F1={f1:.3f}  "
          f"mAP@50={metrics['mAP50']:.3f}  mAP@50-95={metrics['mAP50_95']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["live", "recorded"], default="live",
                    help="'live' runs model.val(); 'recorded' relies on summary.csv")
    cli = ap.parse_args()
    sample = {
        "ModelScale": "Small", "CSPBlock": "C3k2", "C2PSA": True,
        "Lighting": "LowLight", "Occlusion": "NoOcclusion", "InputSize": "S640",
        "Augmentation": "LightAug", "HistogramEqualization": True, "Optimizer": "AdamW",
        "Precision": "FP16", "DetectionMode": "MultiFace", "Head": "DecoupledHead",
        "NMS": "StandardNMS", "DatasetType": "CustomFace",
    }
    run_evaluation(sample, source=cli.source)
