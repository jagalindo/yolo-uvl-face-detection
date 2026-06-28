# YOLOv11 Face Detection driven by a UVL Feature Model

Reproducibility package for the paper *"UVL-Supported Configuration Modeling for
Enhanced YOLOv11 Face Detection."* The project connects a **UVL feature model**
(a software product line description of the detector's configuration space) to a
concrete **YOLOv11** face-detection pipeline: feature selections in the UVL model
are validated against cross-tree constraints, counted/analyzed with flamapy, and
translated into Ultralytics training/inference parameters.

Every reported number is reproducible from this repository: the UVL statistics
are computed live by flamapy (BDD), and the detection metrics come from the
training sweep recorded in `runs/experiments/summary.csv`.

## Project structure

```
yolo-uvl-face-detection/
├── README.md  ·  requirements.txt  ·  .gitignore
├── src/
│   ├── uvl_config.py        # Shared core: constraint validator, UVL->YOLO translation,
│   │                        #   JSON export, real-metrics lookup (no fabricated values)
│   ├── analyze_uvl.py       # flamapy/BDD analysis: satisfiability, counts, sampling
│   ├── parse_uvl.py         # Validate a selection -> export JSON -> evaluate (live|recorded)
│   ├── app.py               # Streamlit UI (2 tabs): detector dashboard + flamapy-based configurator
│   ├── train.py             # Fine-tune one YOLOv11 model on the face dataset
│   ├── run_experiments.py   # One-factor-at-a-time configuration sweep
│   ├── finish_experiments.py# Resilient sweep: reuse done runs, train only what's missing
│   ├── detect.py            # Batch inference over assets/ with CLAHE low-light boost
│   ├── run.py               # Interactive launcher menu for all of the above
│   ├── organize_dataset.py  # Build the YOLO dataset layout from raw train/val folders
│   └── download_torch.py    # Resumable download of a CUDA torch wheel (Windows helper)
├── models/
│   ├── yolo_custom_model.uvl  # The UVL feature model (7 dims, 6 cross-tree constraints)
│   └── weights/
│       ├── yolo11n.pt · yolo11s.pt · yolo11m.pt   # base weights (nano/small/medium)
│       └── best.pt                                # trained face detector (baseline)
├── dataset/
│   └── My_YOLO_Dataset/      # 613 train + 52 val images, 1 class ("face")
│       ├── data.yaml  ·  images/{train,val}/  ·  labels/{train,val}/
├── runs/
│   ├── detect/              # original training/validation runs (curves, plots, weights)
│   ├── experiments/         # config-sweep runs + summary.csv (the paper's results table)
│   └── uvl_analysis.txt     # saved flamapy analysis report
└── assets/
    └── input_sample.png     # default test image for app.py / detect.py
```

## 1. Setup

```bash
cd yolo-uvl-face-detection
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **flamapy / BDD note.** `flamapy-bdd` pulls the `dd` BDD library, which on
> Python 3.9 must be built without pip's isolated build env. If
> `pip install -r requirements.txt` fails on `dd`, run:
> ```bash
> pip install setuptools wheel
> pip install --no-build-isolation flamapy-fm flamapy-sat flamapy-bdd
> ```

**Reference environment for the reported results:** Python 3.9, PyTorch 2.8.0,
Ultralytics 8.4.66, flamapy 2.0.1/2.5.0, on an Apple M5 (MPS GPU).

### Hardware / device

All scripts auto-select the device **`mps` (Apple Silicon) → `cuda` (NVIDIA) →
`cpu`**. Inference and flamapy analysis are fast on any of these. Full training
is much faster on CUDA; on MPS it works but is slow, and **MPS has no FP16
training path** (the precision axis is therefore evaluated at inference time).

## 2. Reproducibility steps

Run from the project root (scripts resolve paths relative to it).

### Step A — Analyze the UVL feature model (flamapy / BDD)

```bash
python src/analyze_uvl.py --sample 3
```

Reproduces the §Methodology numbers. Expected output:

```
Satisfiable (valid)   : True
Total features        : 76        Leaf features : 53
Max tree depth (nodes): 4         Dimensions    : 7
Core features         : 20        Dead features : 0
VALID CONFIGURATIONS  : 2,926,264,320
```

(The full report is also saved in `runs/uvl_analysis.txt`.)

### Step B — Validate a configuration, export JSON, and evaluate

```bash
python src/parse_uvl.py --source recorded   # instant: reads runs/experiments/summary.csv
python src/parse_uvl.py --source live       # runs model.val() on the matching trained weights
```

Prints the UVL summary, validates the selection against the 6 cross-tree
constraints, writes `results/validated_config.json`, and reports real
`P / R / F1 / mAP@50 / mAP@50-95` with the **data source labeled**.

### Step C — Reproduce the configuration sweep (the paper's results table)

```bash
python src/finish_experiments.py --epochs 50
```

One-factor-at-a-time around a baseline (`yolo11s`, 640, FP32, augmentation on),
50 epochs each. It **reuses already-trained 50-epoch runs** and trains only what
is missing, writing `runs/experiments/summary.csv` incrementally (crash-safe).
Full run is ~3–4 h on the M5; if all runs already exist it just re-validates them
in ~1 min. Single-model training alone: `python src/train.py`.

The recorded results (`runs/experiments/summary.csv`) are:

| Axis | Variant | Model | ImgSz | P | R | mAP@50 | mAP@50-95 | Latency |
|------|---------|-------|-------|------|------|--------|-----------|---------|
| baseline    | FP32, aug    | yolo11s | 640 | 0.618 | 0.562 | 0.573 | 0.186 | 24.0 ms |
| model size  | → nano       | yolo11n | 640 | 0.683 | 0.503 | 0.568 | 0.185 | 13.3 ms |
| model size  | → medium     | yolo11m | 640 | 0.622 | 0.477 | 0.496 | 0.157 | 36.8 ms |
| input size  | → 960        | yolo11s | 960 | 0.730 | 0.527 | 0.573 | 0.169 | 41.8 ms |
| augmentation| → disabled   | yolo11s | 640 | 0.497 | 0.308 | 0.283 | 0.072 | 12.9 ms |
| precision   | → FP16       | yolo11s | 640 | 0.623 | 0.562 | 0.573 | 0.185 | n/a\* |

\* FP16 latency is omitted: MPS has no native FP16 inference path (it falls back
to CPU). FP16 **accuracy** is valid and essentially identical to FP32.

### Step D — Interactive app (Streamlit, 2 tabs)

```bash
streamlit run src/app.py        # or: python src/app.py  (auto-launches Streamlit)
```

Opens `http://localhost:8501`:

- **🖥️ Detector Dashboard** — choose features in the sidebar (live constraint
  checking), see the UVL→YOLO translation, **Evaluate** the configuration
  (toggle **Recorded** = instant real numbers from `summary.csv`, or **Live** =
  run `model.val()` now), and run inference on an uploaded image / PDF page (a
  default sample is preloaded).
- **🧬 flamapy-based Configurator** — interactive feature tree, live constraint
  validation, **Run flamapy analysis** (real BDD count), **Sample valid
  configurations**, and a `validated_config.json` download.

### Step E — Batch inference over your own images

```bash
python src/detect.py            # processes everything in assets/, writes results/
```

Applies CLAHE low-light enhancement and opens an OpenCV preview window per image,
so run it in a normal desktop session (not headless), and install
`opencv-python` rather than `opencv-python-headless`.

### Convenience launcher

```bash
python src/run.py               # menu: analyze / validate / app / train / sweep
```

## Quick verification checklist

| Check | Command | Healthy result |
|-------|---------|----------------|
| flamapy pipeline | `python src/analyze_uvl.py` | `VALID CONFIGURATIONS: 2,926,264,320` |
| Validate + evaluate | `python src/parse_uvl.py --source recorded` | `✅ SATISFIABLE` + a metrics line |
| Detector | `streamlit run src/app.py` → Detect faces | bounding boxes drawn |

## Dataset

`dataset/My_YOLO_Dataset/` is standard Ultralytics layout: one class (`face`),
613 train / 52 val images (169 face instances in val) with YOLO `.txt` labels.
`data.yaml`'s `path:` is rewritten to an absolute path at runtime, so it stays
portable in version control. `src/organize_dataset.py` documents how the dataset
was assembled from raw `train/` and `val/` folders.

## How artifacts map to the paper

| Paper claim | Reproduced by | Artifact |
|-------------|---------------|----------|
| 7 dimensions, 76/53 features, 2,926,264,320 valid configs, 0 dead | `src/analyze_uvl.py` | `models/yolo_custom_model.uvl`, `runs/uvl_analysis.txt` |
| 6 cross-tree constraints, satisfiability | `src/analyze_uvl.py`, `src/uvl_config.py` | `models/yolo_custom_model.uvl` |
| Configuration comparison (Table) | `src/finish_experiments.py` | `runs/experiments/summary.csv` |
| Sampled valid configurations | `python src/analyze_uvl.py --sample N` | — |
| Qualitative detection figures | `src/app.py` / `src/detect.py` | `assets/`, `runs/` |
