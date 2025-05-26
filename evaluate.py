# evaluate.py
from data_loader import load_data
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import yaml

def main():
    # 1. Load config
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # 2. Load data
    X_train, X_test, y_train, y_test = load_data(cfg["train_dir"], tuple(cfg["input_shape"]))

    # 3. Load the trained model
    model = load_model(cfg["model_path"])

    # 4. Make predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # 5. Print metrics
    classes = sorted([d for d in __import__("os").listdir(cfg["train_dir"]) 
                      if __import__("os").path.isdir(__import__("os").path.join(cfg["train_dir"], d))])
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=classes))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
