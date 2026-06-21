"""Streamlit UI: UVL feature-model configurator + YOLOv11 detector/evaluator.

Two tabs:
  1. Detector Dashboard - translate a feature selection into YOLO arguments,
     run REAL validation (live model.val on the matching trained weights, with
     fallback to recorded runs/experiments/summary.csv), and run live inference
     on an uploaded image or PDF page.
  2. FlamaPy IDE Configurator - interactive feature tree with live cross-tree
     constraint checking and the REAL BDD analysis (FlamaPy) of the model.

Run:  streamlit run src/app.py     (or simply:  python src/app.py)
"""
import io
import os
import sys
import time
import json
from pathlib import Path

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uvl_config as uvl

try:
    import fitz  # PyMuPDF, optional - enables PDF page input
except Exception:
    fitz = None


# --- Auto-launch Streamlit if run as a plain script ---------------------------
if not st.runtime.exists():
    import subprocess
    print("🚀 Launching Streamlit interface...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", sys.argv[0]])
    except KeyboardInterrupt:
        print("\n👋 Streamlit interface stopped by user.")
    sys.exit()

st.set_page_config(page_title="YOLOv11 + UVL Face Detection", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = str(ROOT / "dataset" / "My_YOLO_Dataset" / "data.yaml")
DEFAULT_MODEL = str(ROOT / "models" / "weights" / "best.pt")
DEFAULT_IMAGE = str(ROOT / "assets" / "input_sample.png")
FALLBACK_WEIGHTS = str(ROOT / "models" / "weights" / "yolo11s.pt")
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_cached_yolo(model_path):
    """Load YOLO weights once and reuse across reruns."""
    return YOLO(model_path)


@st.cache_resource
def cached_uvl_stats():
    """Run the real FlamaPy/BDD analysis once and cache the plain stats."""
    import analyze_uvl
    r = analyze_uvl.analyze()
    return {k: v for k, v in r.items() if not k.startswith("_")}


def update_dataset_path(yaml_path):
    """Make data.yaml's `path:` absolute so Ultralytics resolves it."""
    if not yaml_path or not os.path.exists(yaml_path):
        return
    try:
        yaml_path = os.path.abspath(yaml_path)
        dataset_dir = os.path.dirname(yaml_path).replace("\\", "/")
        with open(yaml_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        found = False
        for i, line in enumerate(lines):
            if line.strip().startswith("path:"):
                lines[i] = f"path: {dataset_dir}\n"
                found = True
                break
        if not found:
            lines.insert(0, f"path: {dataset_dir}\n")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error updating dataset path: {e}")


# --- Sidebar: the UVL feature tree (writes into st.session_state) --------------
st.sidebar.header("🛠️ UVL Feature Tree")
DEFAULTS = {
    "ModelScale": "Small", "CSPBlock": "C3k2", "C2PSA": True,
    "Lighting": "Daylight", "Occlusion": "NoOcclusion", "InputSize": "S640",
    "Augmentation": "LightAug", "HistogramEqualization": False, "Optimizer": "AdamW",
    "Precision": "FP32", "DetectionMode": "MultiFace", "Head": "DecoupledHead",
    "NMS": "StandardNMS", "DatasetType": "CustomFace",
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def feature_select(label, key, group):
    st.sidebar.markdown(f"**{group}**")
    opts = uvl.FEATURES[key]
    if opts == [True, False]:
        st.session_state[key] = st.sidebar.checkbox(label, value=st.session_state[key], key=f"w_{key}")
    else:
        st.session_state[key] = st.sidebar.selectbox(
            label, opts, index=opts.index(st.session_state[key]), key=f"w_{key}")


feature_select("ModelScale", "ModelScale", "1. Backbone")
feature_select("CSPBlock", "CSPBlock", "")
feature_select("C2PSA attention block", "C2PSA", "")
feature_select("LightingConditions", "Lighting", "2. Image Capture")
feature_select("Occlusion", "Occlusion", "")
feature_select("InputSize", "InputSize", "3. Data Processing")
feature_select("Augmentation", "Augmentation", "")
feature_select("Histogram Equalization", "HistogramEqualization", "")
feature_select("Optimizer", "Optimizer", "4. Training")
feature_select("Precision", "Precision", "")
feature_select("Head", "Head", "5. Detection")
feature_select("FaceDetectionMode", "DetectionMode", "")
feature_select("NMS", "NMS", "6. Post-processing")
feature_select("DatasetType", "DatasetType", "7. Evaluation")

st.sidebar.subheader("Inference thresholds")
conf_threshold = st.sidebar.slider("Confidence", 0.01, 1.00, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU (NMS)", 0.10, 0.90, 0.45, 0.05)

st.sidebar.subheader("📂 Paths")
custom_dataset_path = st.sidebar.text_input("Dataset YAML", value=DEFAULT_YAML)
custom_model_path = st.sidebar.text_input("Trained model (.pt)", value=DEFAULT_MODEL)

selected = {k: st.session_state[k] for k in DEFAULTS}
args = uvl.config_to_yolo_args(selected)
is_valid, errors = uvl.validate_configuration(selected)

if is_valid:
    st.sidebar.success("✅ Configuration satisfies all 6 cross-tree constraints")
else:
    st.sidebar.error("❌ Constraint violation(s):")
    for e in errors:
        st.sidebar.caption(f"• {e}")


tab_dash, tab_flama = st.tabs(["🖥️ Detector Dashboard", "🧬 FlamaPy IDE Configurator"])

# ============================== TAB 1: DASHBOARD ==============================
with tab_dash:
    st.title("🖥️ YOLOv11 Face Detection & UVL Configuration Engine")
    st.markdown("Translate the UVL feature selection into YOLOv11 arguments, then evaluate and run inference.")

    st.subheader("⚙️ UVL → YOLO translation")
    st.code(
        f"base_weights = {args['base_weights']!r}\n"
        f"imgsz        = {args['imgsz']}\n"
        f"half (FP16)  = {args['half']}\n"
        f"max_det      = {args['max_det']}  # {selected['DetectionMode']}\n"
        f"augment      = {'disabled' if args['augment_disabled'] else 'enabled'}\n"
        f"conf={conf_threshold}, iou={iou_threshold}, device={DEVICE!r}",
        language="python")

    st.subheader("📊 Model evaluation (real measured results)")
    eval_mode = st.radio(
        "Evaluation source",
        ["📊 Recorded results (paper Table 2 / summary.csv — instant)",
         "🔬 Live evaluation (run model.val on the trained weights)"],
        horizontal=True,
        help="Both use real data. 'Recorded' reads the values measured during our "
             "experiment sweep (fast, no GPU). 'Live' re-runs model.val() on the "
             "matching trained weights now.")
    use_recorded = eval_mode.startswith("📊")

    if not is_valid:
        st.warning("Fix the constraint violations in the sidebar before evaluating.")
    elif st.button("🚀 Evaluate this configuration"):
        weights = uvl.trained_weights_for(selected)
        metrics, source = None, ""
        # Live validation on the matching trained weights (unless recorded mode).
        if not use_recorded and weights is not None and os.path.exists(custom_dataset_path):
            with st.spinner(f"Running live model.val() on {weights.parent.parent.name}/best.pt ..."):
                try:
                    update_dataset_path(custom_dataset_path)
                    b = load_cached_yolo(str(weights)).val(
                        data=custom_dataset_path, imgsz=args["imgsz"], half=args["half"],
                        device=DEVICE, verbose=False, max_det=args["max_det"],
                        conf=conf_threshold, iou=iou_threshold, workers=0, plots=False).box
                    metrics = {"P": b.mp, "R": b.mr, "mAP50": b.map50, "mAP50_95": b.map}
                    source = f"live model.val() on {weights.parent.parent.name}/best.pt"
                except Exception as e:
                    st.info(f"Live validation unavailable ({str(e)[:80]}); using recorded results.")
        # Recorded real measurements from the experiment summary (default / fallback).
        if metrics is None:
            m, cfg, note = uvl.measured_metrics(selected)
            if m is None:
                st.error(f"No measured results available: {note}")
                st.stop()
            metrics, source = m, f"recorded summary.csv [{cfg}] ({note})"

        st.success(f"✅ Source: {source}")
        f1 = (2 * metrics["P"] * metrics["R"] / (metrics["P"] + metrics["R"])
              if metrics["P"] + metrics["R"] > 0 else 0.0)
        c = st.columns(5)
        c[0].metric("Precision", f"{metrics['P']:.3f}")
        c[1].metric("Recall", f"{metrics['R']:.3f}")
        c[2].metric("F1", f"{f1:.3f}")
        c[3].metric("mAP@50", f"{metrics['mAP50']:.3f}")
        c[4].metric("mAP@50-95", f"{metrics['mAP50_95']:.3f}")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        vals = [metrics["P"], metrics["R"], f1, metrics["mAP50"], metrics["mAP50_95"]]
        bars = ax.bar(["P", "R", "F1", "mAP@50", "mAP@50-95"], vals,
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax.set_ylim(0, 1.05)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=9)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("📸 Inference on your own image or PDF")
    types = ["jpg", "jpeg", "png"] + (["pdf"] if fitz else [])
    uploaded = st.file_uploader(f"Upload an image{' or PDF' if fitz else ''}", type=types)

    original_image = None
    if uploaded is not None:
        if uploaded.name.lower().endswith(".pdf") and fitz:
            doc = fitz.open(stream=uploaded.read(), filetype="pdf")
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=150)
            original_image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            st.image(original_image, caption="First PDF page", use_container_width=True)
        else:
            original_image = Image.open(uploaded).convert("RGB")
            st.image(original_image, caption="Uploaded image", use_container_width=True)
    elif os.path.exists(DEFAULT_IMAGE):
        original_image = Image.open(DEFAULT_IMAGE).convert("RGB")
        st.info("Loaded default test image (assets/input_sample.png)")
        st.image(original_image, caption="Default image", use_container_width=True)

    if original_image is not None and st.button("🔍 Detect faces"):
        with st.spinner("Detecting..."):
            try:
                processed = original_image
                brightness = float(np.mean(np.array(original_image.convert("L"))))
                if brightness < 100:
                    st.info(f"💡 Low light ({brightness:.0f}/255): auto-enhancing.")
                    processed = ImageEnhance.Contrast(
                        ImageEnhance.Brightness(processed).enhance(1.8)).enhance(1.3)
                else:
                    st.success(f"☀️ Good lighting ({brightness:.0f}/255).")
                model_path = (custom_model_path if os.path.exists(custom_model_path) else FALLBACK_WEIGHTS)
                model = load_cached_yolo(model_path)
                results = model.predict(
                    source=processed, imgsz=args["imgsz"], half=args["half"], device=DEVICE,
                    max_det=args["max_det"], conf=conf_threshold, iou=iou_threshold, verbose=False)
                st.image(results[0].plot(), caption=f"Detection ({Path(model_path).name})",
                         use_container_width=True)
                boxes = results[0].boxes
                if len(boxes) > 0:
                    confs = [float(b.conf[0]) * 100 for b in boxes]
                    st.success(f"🧑 {len(boxes)} face(s), avg confidence {np.mean(confs):.1f}%")
                else:
                    st.warning("No faces detected.")
            except Exception as e:
                st.error(f"Inference error: {e}")

# ========================= TAB 2: FLAMAPY CONFIGURATOR =======================
with tab_flama:
    st.title("🧬 FlamaPy IDE Feature Configurator")
    st.markdown("Live cross-tree constraint checking plus the **real** FlamaPy/BDD analysis of `yolo_custom_model.uvl`.")

    col_cfg, col_analysis = st.columns([1.1, 1])

    with col_cfg:
        st.subheader("Current selection")
        st.json(selected)
        if is_valid:
            st.success("SATISFIABLE — all 6 cross-tree constraints hold.")
        else:
            st.error("UNSATISFIABLE — conflicts:")
            for e in errors:
                st.caption(f"• {e}")
        # JSON artifact download (paper schema).
        artifact = {**selected, **args}
        st.download_button("💾 Download validated_config.json",
                           data=json.dumps(artifact, indent=4),
                           file_name="validated_config.json", mime="application/json")

    with col_analysis:
        st.subheader("FlamaPy / BDD analysis")
        st.caption("Computed live from the UVL model (not hardcoded).")
        if st.button("⚡ Run FlamaPy analysis"):
            with st.spinner("Parsing UVL and compiling BDD..."):
                try:
                    s = cached_uvl_stats()
                    st.metric("Valid configurations", f"{s['n_configs']:,}")
                    a, b = st.columns(2)
                    a.metric("Satisfiable", str(s["satisfiable"]))
                    b.metric("Dead features", s["n_dead"])
                    a.metric("Total features", s["n_features"])
                    b.metric("Leaf features", s["n_leaves"])
                    a.metric("Dimensions", len(s["dimensions"]))
                    b.metric("Max depth", s["max_depth"])
                    st.caption(f"Dimensions: {', '.join(s['dimensions'])}")
                except Exception as e:
                    st.error(f"FlamaPy analysis failed: {e}")

        n = st.number_input("Random valid configurations to sample", 1, 20, 3)
        if st.button("🎲 Sample valid configurations"):
            with st.spinner("Uniform BDD sampling..."):
                try:
                    import analyze_uvl
                    for i, cfg in enumerate(analyze_uvl.sample(int(n)), 1):
                        st.caption(f"[{i}] {', '.join(cfg)}")
                except Exception as e:
                    st.error(f"Sampling failed: {e}")
