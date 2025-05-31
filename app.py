# app.py

import streamlit as st
import numpy as np
import cv2
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from grad_cam import make_gradcam_heatmap
import os

# ─────────────────────────────────────────────────────────────────────────────
# 1) APP LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("Brain Tumor Detection")
st.write("Upload an MRI scan and get both the predicted class and Grad‐CAM heatmap.")

# ─────────────────────────────────────────────────────────────────────────────
# 2) LOAD THE MODEL ONCE (CACHE IT)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource  # cache the loaded model so it isn’t reloaded on every run
def load_tumor_model():
    # Make sure the path matches your repo structure: "models/brain_tumor.h5"
    return load_model(os.path.join("models", "brain_tumor.h5"))

model = load_tumor_model()

# We need the last convolutional layer name used for Grad-CAM:
# If you used "conv2d_1" in inference_api.py, keep the same here:
LAST_CONV = "conv2d_1"

# ─────────────────────────────────────────────────────────────────────────────
# 3) DEFINE HELPER FUNCTIONS FOR OOD CHECKS & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def looks_like_mri(bgr: np.ndarray) -> bool:
    """
    Quick “out‐of‐distribution” check to see if the image is a grayscale MRI
    of a brain. Returns True if it looks like an MRI, False otherwise.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull) if hull is not None else 0
    solidity = float(area) / hull_area if hull_area else 0
    # Real brain MRIs tend to have high solidity and occupy a decent fraction of the image
    return (solidity > 0.9) and (area > 0.2 * bgr.shape[0] * bgr.shape[1])

def is_grayscale(bgr: np.ndarray, tol: float = 10.0) -> bool:
    """
    Return True if the uploaded image is nearly grayscale (i.e. MRI)
    by measuring channel differences. False for full-color photos.
    """
    diffs = np.abs(bgr[..., 0].astype(int) - bgr[..., 1].astype(int))
    diffs += np.abs(bgr[..., 1].astype(int) - bgr[..., 2].astype(int))
    return diffs.mean() < tol

def preprocess_for_model(bgr: np.ndarray):
    """
    Resize BGR→(128,128), scale to [0,1], and return a (1,128,128,3) tensor.
    Also return the original BGR resized (for Grad-CAM overlay).
    """
    resized = cv2.resize(bgr, (128, 128))
    arr = resized.astype("float32") / 255.0
    return np.expand_dims(arr, axis=0), resized

# ─────────────────────────────────────────────────────────────────────────────
# 4) DEFINE YOUR CLASS NAMES
# ─────────────────────────────────────────────────────────────────────────────
# In your inference_api.py, you used TRAIN_DIR = "dataset/Training" to derive classes,
# but here we’ll just hard‐code if you know them. For example:
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
# Adjust the order if your model was trained differently (just match index→label).

# ─────────────────────────────────────────────────────────────────────────────
# 5) STREAMLIT FILE UPLOADER AND PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Choose an MRI image…", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Read the image bytes from Streamlit’s uploader
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("❌ Could not decode the image. Make sure it’s a valid PNG/JPEG file.")
    else:
        # 1) Out‐of‐distribution (OOD) checks
        if not looks_like_mri(bgr):
            st.warning("⚠️ Does not look like a brain MRI. Please upload a proper scan.")
        elif not is_grayscale(bgr):
            st.warning("⚠️ The image seems to be in color. Brain MRIs are typically grayscale.")
        else:
            # 2) Preprocess and predict
            img_tensor, proc_bgr = preprocess_for_model(bgr)
            probs = model.predict(img_tensor)[0]
            idx = int(np.argmax(probs))
            label = CLASSES[idx]
            confidence = float(probs[idx])

            st.success(f"**Prediction:** {label}  \n**Confidence:** {confidence * 100:.2f}%")

            # 3) Generate Grad-CAM
            heatmap = make_gradcam_heatmap(img_tensor, model, LAST_CONV, idx)
            heatmap = cv2.resize(heatmap, (proc_bgr.shape[1], proc_bgr.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(proc_bgr, 0.6, heatmap, 0.4, 0)

            # 4) Display the Grad-CAM overlay in Streamlit
            st.subheader("Grad‐CAM Heatmap Overlay")
            st.image(overlay, channels="BGR", use_column_width=True)

