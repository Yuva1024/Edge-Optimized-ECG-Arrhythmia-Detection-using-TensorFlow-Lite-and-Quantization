import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Paths
DATA_PATH = "data\ecg.csv"
MODEL_DIR = "models/ecg_saved_model"

def load_data():
    print("Loading ECG dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int32)

    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def main():
    X_train, X_test, y_train, y_test = load_data()

    print("Building model...")
    model = build_model(input_shape=X_train.shape[1], num_classes=5)

    print("Training model...")
    model.fit(X_train, y_train, epochs=5, batch_size=32)

    print("Evaluating model...")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")

    print("Saving model in SavedModel format...")
    os.makedirs("models", exist_ok=True)
    model.export(MODEL_DIR)

    print("SavedModel exported successfully!")

if __name__ == "__main__":
    main()
