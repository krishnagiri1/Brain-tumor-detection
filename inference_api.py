# inference_api.py
import io, base64, os
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from tensorflow.keras.models import load_model
from grad_cam import make_gradcam_heatmap

app = FastAPI()
MODEL_PATH = "models/brain_tumor.h5"
LAST_CONV  = "conv2d_1"
TRAIN_DIR  = "dataset/Training"
CLASSES = sorted(d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d)))
model = load_model(MODEL_PATH)

def looks_like_mri(bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) else 0
    return solidity > 0.9 and area > 0.2 * bgr.shape[0] * bgr.shape[1]

def is_grayscale(bgr: np.ndarray, tol: float=10.0) -> bool:
    diffs = np.abs(bgr[...,0].astype(int) - bgr[...,1].astype(int))
    diffs += np.abs(bgr[...,1].astype(int) - bgr[...,2].astype(int))
    return diffs.mean() < tol

def preprocess_for_model(bgr: np.ndarray):
    # resize, normalize, shape=(1,128,128,3)
    resized = cv2.resize(bgr, (128,128))
    arr = resized.astype("float32") / 255.0
    return np.expand_dims(arr, 0), resized  # return tensor and resized BGR for CAM overlay

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    # decode once
    raw = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Invalid image file")
    # OOD checks
    if not looks_like_mri(bgr) or not is_grayscale(bgr):
        return {
            "prediction": "No brain scan detected",
            "confidence": None,
            "cam_image": None
        }
    # prepare for model
    img_tensor, proc_bgr = preprocess_for_model(bgr)
    # classify
    probs = model.predict(img_tensor)[0]
    idx = int(np.argmax(probs))
    label = CLASSES[idx]
    confidence = float(probs[idx])
    # Grad-CAM
    heatmap = make_gradcam_heatmap(img_tensor, model, LAST_CONV, idx)
    heatmap = cv2.resize(heatmap, (proc_bgr.shape[1], proc_bgr.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(proc_bgr, 0.6, heatmap, 0.4, 0)
    # encode
    success, enc = cv2.imencode(".jpg", overlay)
    if not success:
        raise HTTPException(500, "Failed to encode CAM image")
    cam_b64 = base64.b64encode(enc.tobytes()).decode()

    return {
        "prediction": label,
        "confidence": confidence,
        "cam_image": cam_b64
    }

@app.get("/predict")
def predict_get():
    raise HTTPException(405, "Use POST /predict with form-data")
