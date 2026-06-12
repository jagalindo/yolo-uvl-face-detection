"""Train the custom YOLOv11 face-detection model on the My_YOLO_Dataset.

Run from anywhere:  python src/train.py
Outputs are written to  runs/detect/face_detection_model_v34/.
"""
import multiprocessing
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# Project root = parent of the src/ directory that holds this file.
ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset" / "My_YOLO_Dataset" / "data.yaml"
BASE_WEIGHTS = ROOT / "models" / "weights" / "yolo11s.pt"
RUNS_DIR = ROOT / "runs" / "detect"


def ensure_dataset_path(yaml_path: Path) -> None:
    """Make data.yaml's `path:` an absolute path so Ultralytics resolves it."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str(yaml_path.parent)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def main():
    # Autodetect device: Apple GPU (mps) > NVIDIA GPU (cuda, index 0) > CPU.
    device = "mps" if torch.backends.mps.is_available() else (0 if torch.cuda.is_available() else "cpu")
    print("==========================================================")
    print(f"💻 TRAINING DEVICE: {device} (cuda={torch.cuda.is_available()}, mps={torch.backends.mps.is_available()})")
    print("==========================================================")

    if not DATA_YAML.exists():
        print(f"❌ Error: Dataset config file not found at {DATA_YAML}")
        return
    ensure_dataset_path(DATA_YAML)

    print("⏳ Loading YOLOv11 model...")
    model = YOLO(str(BASE_WEIGHTS))

    print("🚀 Starting model training on custom face dataset...")
    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        project=str(RUNS_DIR),
        name="face_detection_model_v34",
        patience=20,
    )

    best = RUNS_DIR / "face_detection_model_v34" / "weights" / "best.pt"
    print("✅ Training process completed successfully!")
    print(f"🎉 Best trained model: {best}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
