import streamlit as st
import numpy as np
import cv2
import os
import requests
from tensorflow.keras.models import load_model
from grad_cam import make_gradcam_heatmap

# ─────────────────────────────────────────────────────────────────────────────
# 1) PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("Brain Tumor Detection")
st.write("Upload an MRI scan and get both the predicted class and Grad-CAM heatmap.")

# ─────────────────────────────────────────────────────────────────────────────
# 2) MODEL DOWNLOAD & LOADING
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR  = "models"
MODEL_NAME = "brain_tumor.h5"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Replace with your actual Google Drive file ID for brain_tumor.h5
FILE_ID   = "1B7TylG3svQ3JieoIXAnq1640tILLwKVW"
DRIVE_URL = f"https://docs.google.com/uc?export=download&id={FILE_ID}"

@st.cache_resource
def fetch_and_load_model():
    """
    If the .h5 is not already in /models/, download it from Google Drive.
    Then load it once (cached) so that Streamlit does not reload on every user action.
    """
    # Ensure the models/ directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Download if missing
    if not os.path.isfile(MODEL_PATH):
        with st.spinner("📥 Downloading model from Google Drive…"):
            resp = requests.get(DRIVE_URL, stream=True)
            resp.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

    # Load and return the Keras model
    model = load_model(MODEL_PATH)
    return model

model = fetch_and_load_model()
# Name of your last convolutional layer for Grad-CAM
LAST_CONV = "conv2d_1"

# ─────────────────────────────────────────────────────────────────────────────
# 3) CLASS ORDER (must exactly match training’s flow_from_directory order)
# ─────────────────────────────────────────────────────────────────────────────
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]

# ─────────────────────────────────────────────────────────────────────────────
# 4) HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def looks_like_mri(bgr: np.ndarray) -> bool:
    """Rough contour‐based check: returns True if image has a large, mostly solid shape."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    solidity = float(area) / hull_area if hull_area else 0
    # Real MRIs tend to fill a decent fraction of the frame and be quite solid
    return (solidity > 0.9) and (area > 0.2 * bgr.shape[0] * bgr.shape[1])

def is_grayscale(bgr: np.ndarray, tol: float = 10.0) -> bool:
    """
    Check if the image is effectively grayscale by comparing channel differences.
    Returns True if the channels are nearly identical.
    """
    diffs = np.abs(bgr[..., 0].astype(int) - bgr[..., 1].astype(int))
    diffs += np.abs(bgr[..., 1].astype(int) - bgr[..., 2].astype(int))
    return diffs.mean() < tol

def preprocess_for_model(bgr: np.ndarray):
    """
    Convert BGR→GRAY, resize to (128,128), normalize to [0,1], and expand dims 
    to produce a (1,128,128,1) tensor for model.predict.
    Returns:
      img_tensor: shape (1, 128, 128, 1) ready for model.predict
      proc_gray:  shape (128, 128) - the resized grayscale for Grad-CAM overlay
    """
    # 1) Convert to single‐channel grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)       # shape: (H, W)
    # 2) Resize exactly to (128, 128)
    resized_gray = cv2.resize(gray, (128, 128))        # shape: (128, 128)
    # 3) Normalize pixel values to [0,1]
    arr = resized_gray.astype("float32") / 255.0        # shape: (128, 128)
    # 4) Expand dims → (128,128,1) then → (1,128,128,1)
    arr = np.expand_dims(arr, axis=-1)                  # shape: (128,128,1)
    img_tensor = np.expand_dims(arr, axis=0)            # shape: (1,128,128,1)
    return img_tensor, resized_gray

# ─────────────────────────────────────────────────────────────────────────────
# 5) STREAMLIT UI: UPLOAD, INFERENCE, GRAD-CAM
# ─────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Choose an MRI image…", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Read file‐bytes into a BGR image
    raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    if bgr is None:
        st.error("❌ Could not decode the image. Please upload a valid PNG/JPEG.")
    else:
        # 1) Out‐of‐distribution: does it even look like a brain MRI?
        if not looks_like_mri(bgr):
            st.warning("⚠️ This does not look like a brain MRI. Upload a proper scan.")
        # 2) Check if it’s effectively grayscale
        elif not is_grayscale(bgr):
            st.warning("⚠️ The image appears to be in color. Brain MRIs are typically grayscale.")
        else:
            # 3) Correct preprocessing → (1,128,128,1) tensor
            img_tensor, proc_gray = preprocess_for_model(bgr)

            # 4) Run inference
            probs = model.predict(img_tensor, verbose=0)[0]
            idx = int(np.argmax(probs))
            label = CLASSES[idx]
            confidence = float(probs[idx]) * 100
            st.success(f"**Prediction:** {label}   \n**Confidence:** {confidence:.2f}%")

            # 5) Generate Grad-CAM heatmap
            heatmap = make_gradcam_heatmap(img_tensor, model, LAST_CONV, idx)
            heatmap = cv2.resize(heatmap, (proc_gray.shape[1], proc_gray.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # 6) Overlay on top of the grayscale scan
            proc_bgr = cv2.cvtColor(proc_gray, cv2.COLOR_GRAY2BGR)   # shape: (128,128,3)
            overlay = cv2.addWeighted(proc_bgr, 0.6, heatmap, 0.4, 0)

            st.subheader("Grad-CAM Heatmap Overlay")
            st.image(overlay, channels="BGR", use_column_width=True)

