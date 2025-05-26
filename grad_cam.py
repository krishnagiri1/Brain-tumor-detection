# grad_cam.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2
import os

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 1. Create a model that maps the input image to the activations
    #    of the last conv layer and the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    # 2. Compute gradient of the top predicted class
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    # 3. Pool gradients over spatial locations
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    # 4. Multiply each channel by “how important this channel is”
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    # 5. Normalize to [0,1]
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # Superimpose
    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    # Save to disk
    cv2.imwrite(cam_path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
    return superimposed

def demo(img_path, model_path, last_conv_layer):
    # 1. Load & preprocess image
    img = load_img(img_path, target_size=(128, 128))
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    # 2. Load model
    model = load_model(model_path)
    # 3. Generate heatmap
    heatmap = make_gradcam_heatmap(x, model, last_conv_layer)
    # 4. Save overlay
    base, ext = os.path.splitext(os.path.basename(img_path))
    out_path = f"cam_{base}.jpg"
    overlay = save_and_display_gradcam(img_path, heatmap, out_path)
    print(f"Saved Grad-CAM overlay to {out_path}")
    return overlay
