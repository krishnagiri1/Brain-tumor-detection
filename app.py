import streamlit as st
import numpy as np
import cv2
import os
import requests
from tensorflow.keras.models import load_model
from grad_cam import make_gradcam_heatmap

# ─────────────────────────────────────────────────────────────────────────────
# 1) STREAMLIT PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("Brain Tumor Detection")
st.write("Upload an MRI scan and get both the predicted class and Grad-CAM heatmap.")

# ─────────────────────────────────────────────────────────────────────────────
# 2) MODEL DOWNLOAD & LOADING (from Google Drive)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR  = "models"
MODEL_NAME = "brain_tumor.h5"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Replace with your own Google Drive file ID for brain_tumor.h5
FILE_ID   = "1B7TylG3svQ3JieoIXAnq1640tILLwKVW"
DRIVE_URL = f"https://docs.google.com/uc?export=download&id={FILE_ID}"

@st.cache_resource
def fetch_and_load_model():
    """
    If `models/brain_tumor.h5` is not present locally, download it from Google Drive.
    Then load it once (this is cached), so Streamlit does not re‐load on every interaction.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.isfile(MODEL_PATH):
        with st.spinner("📥 Downloading model from Google Drive…"):
            resp = requests.get(DRIVE_URL, stream=True)
            resp.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

    model = load_model(MODEL_PATH)
    return model

model = fetch_and_load_model()
LAST_CONV = "conv2d"  # Adjust this if your last conv layer has a different name

# ─────────────────────────────────────────────────────────────────────────────
# 3) CLASS NAMES (must match exactly the order used during training)
# ─────────────────────────────────────────────────────────────────────────────
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]

# ─────────────────────────────────────────────────────────────────────────────
# 4) HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def looks_like_mri(bgr: np.ndarray) -> bool:
    """
    Very rough check: convert to grayscale, threshold, look for a large, mostly‐solid contour.
    Returns True if it “looks” like a brain MRI. Otherwise returns False.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    solidity = float(area) / hull_area if hull_area else 0
    return (solidity > 0.9) and (area > 0.2 * bgr.shape[0] * bgr.shape[1])

def is_grayscale(bgr: np.ndarray, tol: float = 10.0) -> bool:
    """
    Check if the three channels differ by less than `tol` on average.
    If so, it’s effectively grayscale. Returns True if it’s nearly grayscale.
    """
    diffs = np.abs(bgr[..., 0].astype(int) - bgr[..., 1].astype(int))
    diffs += np.abs(bgr[..., 1].astype(int) - bgr[..., 2].astype(int))
    return diffs.mean() < tol

def preprocess_for_model(bgr: np.ndarray):
    """
    Convert the OpenCV BGR image → RGB, resize to (128,128), normalize to [0,1],
    and return:
      - img_tensor: a NumPy array shape (1,128,128,3) for model.predict
      - proc_rgb:   the resized RGB image (shape (128,128,3)) for Grad‐CAM overlay
    """
    # 1) Convert BGR → RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)        # shape: (H, W, 3)
    # 2) Resize exactly to (128, 128)
    resized_rgb = cv2.resize(rgb, (128, 128))         # shape: (128, 128, 3)
    # 3) Normalize pixel values to [0,1]
    arr = resized_rgb.astype("float32") / 255.0       # shape: (128, 128, 3)
    # 4) Expand dims → shape (1, 128, 128, 3)
    img_tensor = np.expand_dims(arr, axis=0)          # shape: (1, 128, 128, 3)
    return img_tensor, resized_rgb

# ─────────────────────────────────────────────────────────────────────────────
# 5) STREAMLIT USER INTERFACE: FILE UPLOAD, INFERENCE, GRAD‐CAM DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Choose an MRI image…", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Read as raw bytes → BGR (OpenCV uses BGR by default)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if bgr is None:
        st.error("❌ Could not decode the image. Please upload a valid PNG/JPEG.")
    else:
        # 1) Optional: check if it looks roughly like a brain MRI
        if not looks_like_mri(bgr):
            st.warning("⚠️ This does not look like a brain MRI. Please upload a proper scan.")
        # 2) Another check: it should be essentially grayscale
        elif not is_grayscale(bgr):
            st.warning("⚠️ The image appears to be in color. Brain MRIs are usually grayscale.")
        else:
            # 3) Preprocess into (1,128,128,3) tensor
            img_tensor, proc_rgb = preprocess_for_model(bgr)

            # 4) Run inference
            probs = model.predict(img_tensor, verbose=0)[0]
            idx = int(np.argmax(probs))
            label = CLASSES[idx]
            confidence = float(probs[idx]) * 100
            st.success(f"**Prediction:** {label}   \n**Confidence:** {confidence:.2f}%")

            # 5) Generate Grad‐CAM heatmap
            heatmap = make_gradcam_heatmap(img_tensor, model, LAST_CONV, idx)
            heatmap = cv2.resize(heatmap, (proc_rgb.shape[1], proc_rgb.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # 6) Overlay the heatmap onto the original RGB (converted back to BGR for display)
            proc_bgr = cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2BGR)  # shape: (128,128,3)
            overlay = cv2.addWeighted(proc_bgr, 0.6, heatmap, 0.4, 0)

            st.subheader("Grad-CAM Heatmap Overlay")
            st.image(overlay, channels="BGR", use_column_width=True)


