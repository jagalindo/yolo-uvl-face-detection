"""Batch face detection over the images in assets/, using the trained model.

Reads the UVL feature model, applies CLAHE low-light enhancement, runs YOLOv11
inference and writes annotated results to results/.

Run:  python src/detect.py
"""
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Project root = parent of the src/ directory that holds this file.
ROOT = Path(__file__).resolve().parent.parent
UVL_PATH = ROOT / "models" / "yolo_custom_model.uvl"
IMAGES_FOLDER = ROOT / "assets"
OUTPUT_FOLDER = ROOT / "results"
BEST_WEIGHTS = ROOT / "models" / "weights" / "best.pt"
FALLBACK_WEIGHTS = ROOT / "models" / "weights" / "yolo11s.pt"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 2. Read UVL Model Safely (Without requiring broken uvlparser package)
print("📘 Reading UVL model features...")
if os.path.exists(UVL_PATH):
    try:
        with open(UVL_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print("🔹 Features found in UVL:")
        for line in lines:
            stripped = line.strip()
            # Basic parsing to print features
            if stripped and not stripped.startswith("namespace") and not stripped.startswith(
                    "features") and stripped != "mandatory" and stripped != "alternative" and stripped != "optional" and stripped != "alternative":
                print(f"   - {stripped}")
    except Exception as e:
        print(f"⚠️ Could not parse UVL details: {e}")
else:
    print(f"⚠️ UVL file not found at: {UVL_PATH}")

# 3. Load the best trained YOLO model
if not os.path.exists(BEST_WEIGHTS):
    print(f"⚠️ Custom model not found at {BEST_WEIGHTS}. Falling back to {FALLBACK_WEIGHTS.name}")
    model_name = str(FALLBACK_WEIGHTS)
else:
    print(f"🚀 Loading custom trained model: {BEST_WEIGHTS}")
    model_name = str(BEST_WEIGHTS)

model = YOLO(model_name)

# 4. Device Selection: Apple GPU (mps) > NVIDIA GPU (cuda) > CPU
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Using device: {device}")


# 5. Image Preprocessing helper for Low Light (to boost detection accuracy)
def preprocess_low_light(img_bgr):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast in dark images."""
    # Calculate average brightness
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)

    if avg_brightness < 90:
        print(
            f"💡 Low light detected (Brightness: {avg_brightness:.1f}/255). Automatically enhancing image for high accuracy...")
        # Convert to LAB color space
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        # Merge back
        limg = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Apply slight bilateral filter to reduce sensor noise while preserving face edges
        denoised_bgr = cv2.bilateralFilter(enhanced_bgr, d=7, sigmaColor=50, sigmaSpace=50)
        return denoised_bgr
    return img_bgr


# 6. Process all images in the folder
image_files = [f for f in os.listdir(IMAGES_FOLDER)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print("⚠️ No images found in the specified folder.")
else:
    print(f"🔍 Found {len(image_files)} image(s). Starting face detection...")

    for img_name in image_files:
        img_path = os.path.join(IMAGES_FOLDER, img_name)
        print(f"\n📸 Processing: {img_name}")

        # Read image using OpenCV
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"❌ Could not read image: {img_name}")
            continue

        # Apply our high-accuracy low-light preprocessing
        processed_bgr = preprocess_low_light(img_bgr)

        # Run prediction
        # Using optimal conf=0.15 and iou=0.40 thresholds to detect small/blurry/far faces
        results = model.predict(
            source=processed_bgr,
            device=device,
            conf=0.15,
            iou=0.40,
            save=True,
            project=str(OUTPUT_FOLDER),
            name="detections",
            verbose=False
        )

        # Print detection stats
        boxes = results[0].boxes
        num_faces = len(boxes)
        print(f"🎯 Detected {num_faces} Face(s)")
        if num_faces > 0:
            confidences = [float(box.conf[0]) * 100 for box in boxes]
            print(f"📈 Average Detection Confidence: {np.mean(confidences):.2f}%")
            for i, conf in enumerate(confidences):
                print(f"   🔹 Face {i + 1}: {conf:.2f}%")

        # Plot result and display window
        annotated_frame = results[0].plot()

        # Scale down for display if image is too large for the screen
        h, w = annotated_frame.shape[:2]
        if w > 1000 or h > 800:
            scale = min(1000 / w, 800 / h)
            annotated_frame = cv2.resize(annotated_frame, (int(w * scale), int(h * scale)))

        cv2.imshow("YOLOv11 Face Detection Result", annotated_frame)

        # Wait 1.5 seconds per image (press any key to skip)
        key = cv2.waitKey(1500)
        if key == 27:  # ESC key to stop
            break

    cv2.destroyAllWindows()
    print("\n✅ Finished processing all images.")
    print(f"📁 Results saved in: {OUTPUT_FOLDER}")
