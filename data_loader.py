# data_loader.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

def load_data(data_dir, img_size=(128, 128), test_size=0.2, random_state=42):
    X, y = [], []
    # Automatically pick up all class-folders
    classes = sorted(entry for entry in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, entry)))
    label_map = {cls_name: idx for idx, cls_name in enumerate(classes)}
    print("Labels:", label_map)   # sanity check

    for cls_name, cls_idx in label_map.items():
        cls_folder = os.path.join(data_dir, cls_name)
        for fname in os.listdir(cls_folder):
            path = os.path.join(cls_folder, fname)
            img = load_img(path, target_size=img_size)
            arr = img_to_array(img) / 255.0
            X.append(arr); y.append(cls_idx)

    X = np.array(X); y = np.array(y)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
