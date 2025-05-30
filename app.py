import streamlit as st
import requests
import base64

# Page config
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.title("Brain Tumor Detection")
st.write("Upload an MRI scan and get both the predicted class and Grad-CAM heatmap.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    files = {"file": uploaded_file.getvalue()}
    try:
        response = requests.post("http://127.0.0.1:8000/predict/", files=files).json()
    except Exception as e:
        st.error(f"Failed to fetch prediction: {e}")
        st.stop()

    pred = response.get("prediction")
    conf = response.get("confidence")
    cam  = response.get("cam_image")

    st.subheader("Prediction")
    if pred == "no_brain_detected":
        st.write("❌ **No brain scan detected.** Please upload a valid MRI image.")
    else:
        st.write(f"**Class:** {pred}")
        # Only show confidence if it's a number
        if conf is not None:
            st.write(f"**Confidence:** {conf:.2%}")

    # Only show heatmap if it exists
    if cam:
        st.subheader("Grad-CAM Heatmap")
        img_data = base64.b64decode(cam)
        st.image(img_data, use_container_width=True)
