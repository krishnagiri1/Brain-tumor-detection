# inference_api.py
import io
import base64
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from grad_cam import make_gradcam_heatmap

app = FastAPI()

# --- CONFIG ---
MODEL_PATH = "models/brain_tumor.h5"
LAST_CONV  = "conv2d_1"
TRAIN_DIR  = "dataset/Training"

# Derive class names from your data folders
import os
CLASSES = sorted(
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
)

# Load your tumor model once
model = load_model(MODEL_PATH)


def preprocess_image_bytes(data: bytes):
    """
    Load raw bytes into a (1,128,128,3) array and return both the array
    and the original PIL image for later overlay.
    """
    img = load_img(io.BytesIO(data), target_size=(128, 128))
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0), img


def is_grayscale(img_array: np.ndarray, tol: float = 10.0) -> bool:
    """
    Quick check: True if the uploaded image is nearly grayscale,
    False if it has strong color channels (e.g. a landscape photo).
    """
    # Sum absolute differences between channels per pixel
    diffs = np.abs(img_array[..., 0].astype(int) - img_array[..., 1].astype(int))
    diffs += np.abs(img_array[..., 1].astype(int) - img_array[..., 2].astype(int))
    return diffs.mean() < tol


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # 0. Read the upload once
    contents = await file.read()
    if not contents:
        raise HTTPException(400, detail="Empty file")

    # 1. Quick color vs grayscale check
    try:
        raw = np.frombuffer(contents, dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("OpenCV failed to decode image")
    except Exception:
        raise HTTPException(400, detail="Invalid image file")

    if not is_grayscale(bgr):
        return {
            "prediction": "no_brain_detected",
            "confidence": None,
            "cam_image": None,
        }

    # 2. Preprocess for the tumor model
    try:
        img_tensor, pil_img = preprocess_image_bytes(contents)
    except Exception:
        raise HTTPException(400, detail="Failed to preprocess image")

    # 3. Tumor classification
    probs = model.predict(img_tensor)[0]
    idx = int(np.argmax(probs))
    label = CLASSES[idx]
    confidence = float(probs[idx])

    # 4. Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_tensor, model, LAST_CONV, pred_index=idx)

    # 5. Overlay heatmap on the original image (in-memory)
    #    Convert PIL -> OpenCV BGR
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)

    # 6. Encode overlay to JPEG + Base64
    success, encoded = cv2.imencode(".jpg", overlay)
    if not success:
        raise HTTPException(500, detail="Failed to encode Grad-CAM image")
    cam_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

    return {
        "prediction": label,
        "confidence": confidence,
        "cam_image": cam_b64,
    }
