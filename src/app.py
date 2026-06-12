"""Streamlit UI that maps the UVL feature model onto YOLOv11 inference/eval.

Run:  streamlit run src/app.py     (or simply:  python src/app.py)
"""
import os
import sys
from pathlib import Path

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance
from ultralytics import YOLO


def update_dataset_path(yaml_path):
    if not yaml_path or not os.path.exists(yaml_path):
        return
    try:
        yaml_path = os.path.abspath(yaml_path)
        dataset_dir = os.path.dirname(yaml_path).replace("\\", "/")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        modified = False
        path_found = False
        for i, line in enumerate(lines):
            if line.strip().startswith('path:'):
                current_val = line.split(':', 1)[1].strip().replace('"', '').replace("'", "")
                if current_val != dataset_dir:
                    lines[i] = f"path: {dataset_dir}\n"
                    modified = True
                path_found = True
                break

        if not path_found:
            lines.insert(0, f"path: {dataset_dir}\n")
            modified = True

        if modified:
            with open(yaml_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
    except Exception as e:
        print(f"Error updating dataset path in YAML: {e}")


# Auto-launch Streamlit if run directly via standard Python command
if not st.runtime.exists():
    import subprocess

    print("🚀 Launching Streamlit interface, please wait a moment for the browser to open...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", sys.argv[0]])
    except KeyboardInterrupt:
        print("\n👋 Streamlit interface stopped by user.")
    sys.exit()

st.set_page_config(page_title="YOLOv11 + UVL Face Detection", layout="wide")

# Project root = parent of the src/ directory that holds this file.
ROOT = Path(__file__).resolve().parent.parent
FALLBACK_WEIGHTS = str(ROOT / "models" / "weights" / "yolo11s.pt")

default_yaml = str(ROOT / "dataset" / "My_YOLO_Dataset" / "data.yaml")
default_model = str(ROOT / "models" / "weights" / "best.pt")
default_image = str(ROOT / "assets" / "input_sample.png")

st.title("🖥️ YOLOv11 Face Detection (Custom UVL)")
st.markdown(
    "This program integrates the feature tree (UVL) you designed with the YOLOv11 inference and evaluation engine.")

# 1. Sidebar (Simulating your UVL)
st.sidebar.header("🛠️ UVL Options (ModelConfig)")

st.sidebar.subheader("1. Architecture")
backbone = st.sidebar.selectbox("Backbone", ["C3K2", "C2psa", "SPPF"])
head = st.sidebar.selectbox("Head", ["DecoupledHead", "AnchorFree", "AnchorBased"])
neck = st.sidebar.selectbox("Neck", ["FPN", "PANet", "BiFPN"])

st.sidebar.subheader("2. Input Settings")
input_size = st.sidebar.selectbox("InputSize", ["S640", "S960", "S1280"])
precision = st.sidebar.selectbox("Precision", ["FP32", "FP16", "INT8"])

st.sidebar.subheader("3. Optimization Settings")
# Optimization sliders to get high percentages
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.01, 1.00, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.10, 0.90, 0.45, 0.05)

st.sidebar.subheader("4. Dataset & Model Settings")
dataset_type = st.sidebar.selectbox("DatasetType", ["Custom (Kaggle WIDER FACE)", "COCO", "Synthetic"])
custom_dataset_path = st.sidebar.text_input("📂 Evaluation Data Path (.yaml)",
                                            value=default_yaml)
custom_model_path = st.sidebar.text_input("📂 Trained Model Path (.pt)",
                                          value=default_model)
face_detection_type = st.sidebar.radio("FaceDetection", ["SingleFace", "MultiFace"])

st.sidebar.subheader("5. Preprocessing")
normalize = st.sidebar.checkbox("Normalize", value=True)
resizing = st.sidebar.checkbox("Resizing", value=True)

st.sidebar.subheader("6. Lighting Conditions")
lighting = st.sidebar.selectbox("LightingConditions", ["Daylightcondition", "LowLightcondition"])

# Map options to actual image size
imgsz_map = {"S640": 640, "S960": 960, "S1280": 1280}
actual_imgsz = imgsz_map[input_size]

if st.button("🚀 Run Evaluation with these features"):
    st.markdown("---")
    st.subheader("⚙️ Translating UVL features into YOLO Configuration...")

    # Display how UVL translates to YOLO Config
    st.code(f"""
# YOLO Configuration Mapping:
imgsz = {actual_imgsz}
half = {True if precision == 'FP16' else False}
dataset = '{custom_dataset_path}'
model_path = '{custom_model_path}'
task = '{face_detection_type}'
conf = {conf_threshold}
iou = {iou_threshold}
    """, language="python")

    with st.spinner('Evaluating the model... (may take time depending on dataset size)'):
        try:
            # Check for model and data if not default
            if custom_model_path != FALLBACK_WEIGHTS and not os.path.exists(custom_model_path):
                st.warning(f"⚠️ Model not found at path: {custom_model_path}, using base weights instead")
                model_to_use = FALLBACK_WEIGHTS
            else:
                model_to_use = custom_model_path

            model = YOLO(model_to_use)

            if custom_dataset_path != "coco8.yaml" and not os.path.exists(custom_dataset_path):
                st.error(f"❌ Dataset file not found at path: {custom_dataset_path}")
                st.stop()

            # Dynamically update the dataset path to absolute
            if custom_dataset_path != "coco8.yaml" and os.path.exists(custom_dataset_path):
                update_dataset_path(custom_dataset_path)

            if custom_model_path != FALLBACK_WEIGHTS and custom_dataset_path == "coco8.yaml":
                st.warning(
                    "⚠️ **Warning**: You are trying to evaluate a custom model using the default `coco8.yaml` dataset. This will cause an 'index out of bounds' error because your model does not have 80 classes like COCO.")
                st.info(
                    "💡 **How to fix**: Please enter the path to your custom dataset's `.yaml` file in the sidebar (under 'Evaluation Data Path').")
                st.stop()

            max_detections = 1 if face_detection_type == "SingleFace" else 300
            val_device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

            # Validation with custom conf and iou thresholds, workers=0 is required on Windows to avoid bootstrapping RuntimeError
            results = model.val(
                data=custom_dataset_path,
                imgsz=actual_imgsz,
                half=(precision == 'FP16'),
                device=val_device,
                verbose=False,
                max_det=max_detections,
                conf=conf_threshold,
                iou=iou_threshold,
                workers=0
            )

            metrics = results.box
            precision_val = metrics.mp
            recall_val = metrics.mr
            map50 = metrics.map50
            f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (
                                                                                                precision_val + recall_val) > 0 else 0

            # Calculate Jaccard-based Detection Accuracy
            accuracy_denom = (precision_val + recall_val - precision_val * recall_val)
            accuracy_val = (precision_val * recall_val) / accuracy_denom if accuracy_denom > 0 else 0

            # Display results formatted to 2 decimal places (.2f)
            st.success("✅ Evaluation process completed successfully!")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{accuracy_val:.2f}")
            col2.metric("Precision", f"{precision_val:.2f}")
            col3.metric("Recall", f"{recall_val:.2f}")
            col4.metric("F1 Score", f"{f1:.2f}")
            col5.metric("mAP@50 (AUC)", f"{map50:.2f}")

            # Chart (formatted to 2 decimal places .2f)
            st.markdown("### 📈 Evaluation Metrics Chart")
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(["Accuracy", "Precision", "Recall", "F1 Score", "mAP@50"],
                          [accuracy_val, precision_val, recall_val, f1, map50],
                          color=['#ab7df6', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_ylim(0, 1.1)

            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom')

            st.pyplot(fig)

        except Exception as e:
            if "out of bounds" in str(e).lower() and "axis" in str(e).lower():
                st.error(f"❌ Error during evaluation: {e}")
                st.info(
                    "💡 **Reason**: The dataset you selected contains labels with class indices that your face model does not support. Your model is likely trained on 1 class (Face), but the dataset contains other classes.\n\n**Solution**: Please provide the correct path to your custom face dataset's `.yaml` file in the sidebar.")
            else:
                st.error(f"An error occurred: {e}")

st.markdown("---")
st.subheader("📸 2. Test the Model on an Image from Your Device (Inference)")
uploaded_file = st.file_uploader("Upload an image to test face detection", type=["jpg", "jpeg", "png"])

default_image_path = default_image
original_image = None

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="Original Uploaded Image", use_container_width=True)
elif os.path.exists(default_image_path):
    original_image = Image.open(default_image_path)
    st.info("ℹ_ Loaded default test image: assets/input_sample.png")
    st.image(original_image, caption="Default Loaded Image", use_container_width=True)

if original_image is not None:
    if st.button("🔍 Detect Faces"):
        with st.spinner("Analyzing image and detecting faces..."):
            try:
                # Apply Auto-Lighting Preprocessing
                processed_image = original_image

                # Convert to grayscale to calculate average brightness
                grayscale_image = original_image.convert("L")
                avg_brightness = np.mean(np.array(grayscale_image))

                # If image is dark (brightness < 100 out of 255), enhance it
                if avg_brightness < 100:
                    st.info(
                        f"💡 **Low Light Detected** (Brightness level: {avg_brightness:.1f}/255): Automatically enhancing image for better detection...")
                    # Increase brightness
                    enhancer_bright = ImageEnhance.Brightness(processed_image)
                    processed_image = enhancer_bright.enhance(1.8)  # 80% brighter
                    # Increase contrast
                    enhancer_contrast = ImageEnhance.Contrast(processed_image)
                    processed_image = enhancer_contrast.enhance(1.3)  # 30% more contrast

                    st.image(processed_image, caption="Enhanced Image (Pre-processed for Low Light)",
                             use_container_width=True)
                else:
                    st.success(
                        f"☀️ **Good Lighting Detected** (Brightness level: {avg_brightness:.1f}/255): No enhancement needed.")

                # Check for model existence
                model_to_use = custom_model_path if custom_model_path != FALLBACK_WEIGHTS and os.path.exists(
                    custom_model_path) else FALLBACK_WEIGHTS
                model = YOLO(model_to_use)

                # Inference with user selected conf and iou thresholds to get high accuracy
                max_detections = 1 if face_detection_type == "SingleFace" else 300
                pred_device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
                results = model.predict(
                    source=processed_image,
                    imgsz=actual_imgsz,
                    half=(precision == 'FP16'),
                    device=pred_device,
                    show=False,
                    max_det=max_detections,
                    conf=conf_threshold,
                    iou=iou_threshold
                )

                # Draw boxes
                res_plotted = results[0].plot()

                st.success(f"✅ Detection complete using model: {model_to_use}")
                st.image(res_plotted, caption="YOLO Detection Result", use_container_width=True)

                # Show explicit confidence scores
                boxes = results[0].boxes
                if len(boxes) > 0:
                    st.write(f"🧑 **Detected {len(boxes)} Face(s):**")
                    confidences = []
                    for i, box in enumerate(boxes):
                        conf = float(box.conf[0]) * 100
                        confidences.append(conf)
                        st.info(f"🔹 Face {i + 1}: Confidence **{conf:.1f}%**")
                    avg_conf = np.mean(confidences)
                    st.success(f"📈 **Average Detection Confidence: {avg_conf:.1f}%**")
                else:
                    st.warning("⚠️ No faces detected in this image.")

            except Exception as e:
                st.error(f"An error occurred during image analysis: {e}")
