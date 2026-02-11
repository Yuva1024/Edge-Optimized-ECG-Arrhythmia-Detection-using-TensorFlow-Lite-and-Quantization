import os
import tensorflow as tf

MODEL_DIR = "models/ecg_saved_model"
TFLITE_PATH = "models/ecg_model.tflite"
TFLITE_QUANT_PATH = "models/ecg_model_quant.tflite"

def convert_model():
    print("Converting SavedModel → TFLite...")

    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)

    # Normal conversion
    tflite_model = converter.convert()
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    print("Normal TFLite model saved:", TFLITE_PATH)

    # Quantized conversion
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quant_model = converter.convert()

    with open(TFLITE_QUANT_PATH, "wb") as f:
        f.write(quant_model)

    print("Quantized TFLite model saved:", TFLITE_QUANT_PATH)

def main():
    os.makedirs("models", exist_ok=True)
    convert_model()

if __name__ == "__main__":
    main()
