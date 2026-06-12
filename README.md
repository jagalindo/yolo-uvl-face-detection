# YOLOv11 Face Detection driven by a UVL Feature Model

Reproducibility package for the paper. The project connects a **UVL feature
model** (a software product line description of the detector's configuration
space) to a concrete **YOLOv11** face-detection pipeline: feature selections in
the UVL model are translated into Ultralytics training/inference parameters
(input size, precision, confidence/IoU thresholds, preprocessing, lighting
handling, etc.).

## Project structure

```
yolo-uvl-face-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── train.py             # Fine-tune YOLOv11 on the face dataset
│   ├── app.py               # Streamlit UI: UVL options -> YOLO eval + inference
│   ├── detect.py            # Batch inference over assets/ with CLAHE low-light boost
│   ├── parse_uvl.py         # Define/save the UVL model + UVL->YOLO config demo
│   ├── organize_dataset.py  # Build the YOLO dataset layout from raw train/val folders
│   └── download_torch.py    # Resumable download of a CUDA torch wheel (Windows helper)
├── models/
│   ├── yolo_custom_model.uvl # The UVL feature model
│   └── weights/
│       ├── yolo11s.pt        # Base YOLOv11-small weights (training start point)
│       └── best.pt           # Trained face detector (= runs/.../face_detection_model_v34)
├── dataset/
│   └── My_YOLO_Dataset/      # 613 train + 52 val images, 1 class ("face")
│       ├── data.yaml
│       ├── images/{train,val}/
│       └── labels/{train,val}/
├── runs/
│   └── detect/               # Training/validation runs: metrics, curves, plots, weights
└── assets/
    └── input_sample.png      # Default test image for app.py / detect.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`ultralytics` installs a CPU/GPU build of PyTorch automatically. The included
`src/download_torch.py` is only needed if you want a specific CUDA wheel on
Windows (the original 2.4 GB `torch-2.5.1+cu121` wheel was **not** kept in this
package — re-download it with that script if required).

### Hardware / device

All scripts auto-select the device in the order **`mps` (Apple Silicon GPU) →
`cuda` (NVIDIA GPU) → `cpu`**, so they run on Windows, Linux, and macOS.

On macOS there is no CUDA: an Apple-Silicon Mac uses the `mps` GPU backend,
while an Intel Mac falls back to CPU. Inference (`app.py`, `detect.py`) is fast
either way, but **full training (`train.py`, 100 epochs) on CPU/MPS is much
slower than on a CUDA GPU** — use the provided `models/weights/best.pt` if you
only need to run/evaluate the model. `detect.py` also opens an OpenCV preview
window, so install `opencv-python` (not `opencv-python-headless`).

## Usage

All scripts resolve paths relative to the project root, so they can be run from
anywhere:

```bash
# 1. Train the detector (writes runs/detect/face_detection_model_v34/)
python src/train.py

# 2. Interactive UVL -> YOLO evaluation & single-image inference (web UI)
streamlit run src/app.py        # or: python src/app.py

# 3. Batch inference over everything in assets/  (results -> results/)
python src/detect.py

# 4. Regenerate the UVL model file and run the UVL->config translation demo
python src/parse_uvl.py
```

## Dataset

`dataset/My_YOLO_Dataset/` is in standard Ultralytics layout: a single class
(`face`), 613 training and 52 validation images with YOLO-format `.txt` labels.
`data.yaml`'s `path:` is rewritten to an absolute path at runtime by the
training/eval scripts, so it stays portable in version control.

`src/organize_dataset.py` documents how the dataset was assembled from raw
`train/` and `val/` folders (it expects them on the Desktop, as in the original
authoring environment).

## Trained results

Training and validation artifacts (loss/precision/recall curves, PR curves,
confusion matrices, sample batches, `results.csv`, and per-run `weights/`) are
under `runs/detect/`. The headline model used by `app.py`/`detect.py` is
`face_detection_model_v34`; `models/weights/best.pt` is a copy of its best
checkpoint.
