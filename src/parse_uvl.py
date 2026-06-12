"""Load the canonical UVL feature model and demonstrate translating a selected
feature configuration into a concrete YOLO evaluation run.

The model itself lives in models/yolo_custom_model.uvl (single source of truth,
analyzed with FlamaPy in src/analyze_uvl.py). This script does NOT regenerate it.

Run:  python src/parse_uvl.py
"""
import os
from pathlib import Path

from ultralytics import YOLO
import matplotlib.pyplot as plt

# Project root = parent of the src/ directory that holds this file.
ROOT = Path(__file__).resolve().parent.parent
UVL_FILE_PATH = ROOT / "models" / "yolo_custom_model.uvl"
RESULTS_DIR = ROOT / "results"

# 1. Load the canonical UVL model (authored once, analyzed by src/analyze_uvl.py).
with open(UVL_FILE_PATH, "r", encoding="utf-8") as f:
    uvl_content = f.read()
print(f"Loaded UVL model from: {UVL_FILE_PATH} ({len(uvl_content.splitlines())} lines)")

# 2. Subset of the feature->parameter mapping used to turn a UVL selection into
#    concrete Ultralytics arguments. Feature names match the canonical model.
yolo_feature_model_python = {
    "ModelScale": {"Nano": "yolo11n.pt", "Small": "yolo11s.pt", "Medium": "yolo11m.pt",
                   "Large": "yolo11l.pt", "XLarge": "yolo11x.pt"},
    "InputSize": {"S640": 640, "S960": 960, "S1280": 1280},
    "Precision": {"FP32": False, "FP16": True, "INT8": "Quantized"},
    "Augmentation": {"LightAug": "light", "HeavyAug": "heavy"},
    "DetectionMode": {"SingleFace": 1, "MultiFace": 300},
}


def generate_yolo_config(selected_features):
    """Translate a validated UVL feature selection into Ultralytics arguments."""
    print("\nTranslating UVL features to YOLO configuration...")
    base_weights = yolo_feature_model_python["ModelScale"].get(selected_features.get("ModelScale"), "yolo11s.pt")
    imgsz = yolo_feature_model_python["InputSize"].get(selected_features.get("InputSize"), 640)
    half = yolo_feature_model_python["Precision"].get(selected_features.get("Precision"), False)
    max_det = yolo_feature_model_python["DetectionMode"].get(selected_features.get("DetectionMode"), 300)

    print(f"-> base weights : {base_weights}")
    print(f"-> imgsz        : {imgsz}")
    print(f"-> half (FP16)  : {half}")
    print(f"-> max_det      : {max_det}")

    try:
        model = YOLO(base_weights)
        # coco8 is only a smoke-test target; point this at the real data.yaml to evaluate.
        results = model.val(data="coco8.yaml", imgsz=imgsz, half=bool(half) if half is not True else True,
                            max_det=max_det, device="cpu")
        m = results.box
        p, r, map50 = m.mp, m.mr, m.map50
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        print(f"\n--- Results --- P={p:.3f} R={r:.3f} F1={f1:.3f} mAP@50={map50:.3f}")

        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(["Precision", "Recall", "F1", "mAP@50"], [p, r, f1, map50],
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax.set_ylim(0, 1.1); ax.set_title("YOLO Face Detection Evaluation")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha='center', va='bottom')
        out_png = RESULTS_DIR / "evaluation_results.png"
        plt.savefig(out_png)
        print(f"Chart saved: {out_png}")
    except Exception as e:
        print(f"Error during evaluation: {e}")


if __name__ == "__main__":
    # A sample configuration as would be produced by the UVL/FlamaPy pipeline.
    sample_configuration = {
        "ModelScale": "Small",
        "InputSize": "S960",
        "Precision": "FP16",
        "Augmentation": "LightAug",
        "DetectionMode": "MultiFace",
    }
    generate_yolo_config(sample_configuration)
