"""Shared, honest core for UVL-driven configuration.

Single source of truth for: the selectable feature options, the six cross-tree
constraints (identical to models/yolo_custom_model.uvl), translation of a
feature selection into Ultralytics arguments, JSON artifact export, and lookup
of REAL measured metrics from runs/experiments/summary.csv.

No metrics are fabricated here: numbers come either from a live model.val() run
(see the app) or from the recorded experiment summary.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_CSV = ROOT / "runs" / "experiments" / "summary.csv"
WEIGHTS_DIR = ROOT / "models" / "weights"
EXPERIMENTS_DIR = ROOT / "runs" / "experiments"

# Selectable options per UVL dimension (mirrors the feature model).
FEATURES = {
    "ModelScale": ["Nano", "Small", "Medium", "Large", "XLarge"],
    "CSPBlock": ["C3k2", "C2f"],
    "C2PSA": [True, False],
    "Lighting": ["Daylight", "ArtificialLight", "LowLight"],
    "Occlusion": ["NoOcclusion", "PartialOcclusion", "FullOcclusion"],
    "InputSize": ["S640", "S960", "S1280"],
    "Augmentation": ["LightAug", "HeavyAug", "disabled"],
    "HistogramEqualization": [True, False],
    "Optimizer": ["SGD", "Adam", "AdamW", "AutoOpt"],
    "Precision": ["FP32", "FP16", "INT8"],
    "DetectionMode": ["SingleFace", "MultiFace"],
    "Head": ["DecoupledHead", "AnchorFree", "AnchorBased"],
    "NMS": ["StandardNMS", "SoftNMS"],
    "DatasetType": ["COCO", "PascalVOC", "KITTI", "CustomFace", "Synthetic", "Mixed"],
}

SCALE_TO_WEIGHTS = {
    "Nano": "yolo11n.pt", "Small": "yolo11s.pt", "Medium": "yolo11m.pt",
    "Large": "yolo11l.pt", "XLarge": "yolo11x.pt",
}
IMGSZ_MAP = {"S640": 640, "S960": 960, "S1280": 1280}


# --- Cross-tree constraints (identical to the UVL `constraints` block) ---------
# Each: (message, predicate) where predicate(sel) -> True means VIOLATED.
CONSTRAINTS = [
    ("C2PSA => C3k2: the C2PSA attention block requires the C3k2 CSP block",
     lambda s: s.get("C2PSA") and s.get("CSPBlock") != "C3k2"),
    ("FP16 => !HeavyAug: FP16 precision is incompatible with heavy augmentation",
     lambda s: s.get("Precision") == "FP16" and s.get("Augmentation") == "HeavyAug"),
    ("INT8 => MultiFace: INT8 precision requires MultiFace detection mode",
     lambda s: s.get("Precision") == "INT8" and s.get("DetectionMode") != "MultiFace"),
    ("LowLight => HistogramEqualization: low-light capture requires histogram equalization",
     lambda s: s.get("Lighting") == "LowLight" and not s.get("HistogramEqualization")),
    ("S1280 => !Nano: 1280px input is incompatible with the Nano backbone",
     lambda s: s.get("InputSize") == "S1280" and s.get("ModelScale") == "Nano"),
    ("SoftNMS => MultiFace: SoftNMS post-processing requires MultiFace detection mode",
     lambda s: s.get("NMS") == "SoftNMS" and s.get("DetectionMode") != "MultiFace"),
]


def validate_configuration(selected: dict):
    """Return (is_valid, [violation_messages]) for a feature selection."""
    errors = [msg for msg, violated in CONSTRAINTS if violated(selected)]
    return (len(errors) == 0), errors


def config_to_yolo_args(selected: dict) -> dict:
    """Translate a validated feature selection into Ultralytics arguments."""
    return {
        "base_weights": SCALE_TO_WEIGHTS.get(selected.get("ModelScale"), "yolo11s.pt"),
        "imgsz": IMGSZ_MAP.get(selected.get("InputSize"), 640),
        "half": selected.get("Precision") == "FP16",
        "max_det": 1 if selected.get("DetectionMode") == "SingleFace" else 300,
        "augment_disabled": selected.get("Augmentation") == "disabled",
    }


def export_json(selected: dict, path) -> Path:
    """Serialize the validated configuration as a JSON artifact (paper schema)."""
    args = config_to_yolo_args(selected)
    artifact = {**selected, **{k: v for k, v in args.items()}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=4)
    return path


# --- Real measured results -----------------------------------------------------
def matching_experiment(selected: dict):
    """Map a selection to the closest measured OFAT run (config name in summary.csv).

    Returns (config_name, exact) where `exact` is True if the selection differs
    from the baseline in exactly the one factor that run measured.
    """
    if selected.get("Augmentation") == "disabled":
        return "noaug_s_640", True
    if selected.get("ModelScale") == "Nano":
        return "model_n_640_aug", True
    if selected.get("ModelScale") == "Medium":
        return "model_m_640_aug", True
    if selected.get("InputSize") == "S960":
        return "input_s_960_aug", True
    if selected.get("Precision") == "FP16":
        return "precision_fp16", True
    return "base_s_640_aug", True


def measured_metrics(selected: dict):
    """Look up REAL metrics for the closest measured single-factor variant.

    Returns (metrics_dict, config_name, note) or (None, None, reason).
    """
    if not SUMMARY_CSV.exists():
        return None, None, "summary.csv not found (run src/finish_experiments.py)"
    config_name, _ = matching_experiment(selected)
    rows = {r["config"]: r for r in csv.DictReader(open(SUMMARY_CSV))}
    row = rows.get(config_name)
    if not row or not row.get("mAP50"):
        return None, config_name, f"no recorded metrics for '{config_name}'"
    metrics = {k: float(row[k]) for k in ("P", "R", "mAP50", "mAP50_95") if row.get(k)}
    # Count how many factors differ from the baseline selection.
    note = ("measured single-factor variant"
            if config_name != "base_s_640_aug" else "measured baseline")
    return metrics, config_name, note


def trained_weights_for(selected: dict):
    """Path to the trained best.pt of the matching measured run (for live val)."""
    config_name, _ = matching_experiment(selected)
    w = EXPERIMENTS_DIR / config_name / "weights" / "best.pt"
    return w if w.exists() else None
