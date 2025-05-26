from data_loader import load_data
from model_builder import build_model
import tensorflow as tf
import yaml


# train.py
import yaml
from data_loader import load_data
from model_builder import build_model

# 1. Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# 2. Load data
X_train, X_test, y_train, y_test = load_data(cfg["train_dir"], tuple(cfg["input_shape"]))

# 3. Build and compile model
model = build_model(tuple(cfg["input_shape"]), cfg["n_classes"])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 4. Train
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=cfg["batch_size"],
    epochs=cfg["epochs"]
)

# 5. Save model
model.save(cfg["model_path"])
