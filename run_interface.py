import numpy as np
import tensorflow as tf
import pandas as pd

DATA_PATH = "data/ecg.csv"
MODEL_PATH = "models/ecg_model_quant.tflite"

def main():
    print("Loading quantized TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Loading sample ECG input...")
    df = pd.read_csv(DATA_PATH)

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int32)

    sample = X[0]
    sample_label = y[0]

    sample = np.expand_dims(sample, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], sample)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output)

    print("Actual Label:", sample_label)
    print("Predicted Label:", predicted_class)

if __name__ == "__main__":
    main()
