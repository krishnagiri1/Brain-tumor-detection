import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from grad_cam import make_gradcam_heatmap

# ─────────────────────────────────────────────────────────────────────────────
# 1) Set up page and title
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("Brain Tumor Detection")
st.write("Upload an MRI scan and get both the predicted class and Grad-CAM heatmap.")

# ─────────────────────────────────────────────────────────────────────────────
# 2) Load model once (cache with the older @st.cache decorator if you pinned
#    Streamlit < 1.18; if you're using Streamlit ≥ 1.18, you can use @st.cache_resource)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache(allow_output_mutation=True)
def load_tumor_model():
    return load_model(os.path.join("models", "brain_tumor.h5"))

model = load_tumor_model()
LAST_CONV = "conv2d_1"   # keep this the same as your original Grad-CAM code

# ─────────────────────────────────────────────────────────────────────────────
# 3) Dynamically derive class names in alphabetical order
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_DIR = "dataset/Training"
CLASSES = sorted(
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
)
# Now CLASSES might be ["glioma","meningioma","no_tumor","pituitary"] exactly as flow_from_directory used.

# ─────────────────────────────────────────────────────────────────────────────
# 4) Helper: OOD check and preprocess
# ─────────────────────────────────────────────────────────────────────────────
def looks_like_mri(bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    solidity = float(area) / hull_area if hull_area else 0
    return (solidity > 0.9) and (area > 0.2 * bgr.shape[0] * bgr.shape[1])

def is_grayscale(bgr: np.ndarray, tol: float = 10.0) -> bool:
    diffs = np.abs(bgr[..., 0].astype(int) - bgr[..., 1].astype(int))
    diffs += np.abs(bgr[..., 1].astype(int) - bgr[..., 2].astype(int))
    return diffs.mean() < tol

def preprocess_for_model(bgr: np.ndarray):
    # 1) Convert BGR→RGB (if your model was trained on RGB)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # 2) Resize to (128,128) if that was your training size
    resized = cv2.resize(rgb, (128, 128))
    arr = resized.astype("float32") / 255.0
    # 3) Expand dims to (1,128,128,3)
    return np.expand_dims(arr, axis=0), resized

# ─────────────────────────────────────────────────────────────────────────────
# 5) Streamlit UI: File uploader & inference flow
# ─────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Choose an MRI image…", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Read bytes as OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if bgr is None:
        st.error("❌ Could not decode the image. Please upload a valid PNG/JPEG.")
    else:
        # 1) OOD checks
        if not looks_like_mri(bgr):
            st.warning("⚠️ This does not look like a brain MRI. Upload a proper scan.")
        elif not is_grayscale(bgr):
            st.warning("⚠️ The image appears to be in full color. Brain MRIs are typically grayscale.")
        else:
            # 2) Preprocess and run inference
            img_tensor, proc_rgb = preprocess_for_model(bgr)
            probs = model.predict(img_tensor)[0]
            idx = int(np.argmax(probs))
            label = CLASSES[idx]
            confidence = float(probs[idx]) * 100.0

            st.success(f"**Prediction:** {label}  \n**Confidence:** {confidence:.2f}%")

            # 3) Grad-CAM
            heatmap = make_gradcam_heatmap(img_tensor, model, LAST_CONV, idx)
            heatmap = cv2.resize(heatmap, (proc_rgb.shape[1], proc_rgb.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Convert the RGB image back to BGR for consistent coloring in OpenCV overlay
            proc_bgr = cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(proc_bgr, 0.6, heatmap, 0.4, 0)

            st.subheader("Grad-CAM Heatmap Overlay")
            st.image(overlay, channels="BGR", use_column_width=True)


