# model_builder.py
from tensorflow.keras import layers, Model

def build_model(input_shape=(128, 128, 3), n_classes=2):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inputs, outputs)
