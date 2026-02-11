import os
import time
import tensorflow as tf

NORMAL_MODEL = "models/ecg_model.tflite"
QUANT_MODEL = "models/ecg_model_quant.tflite"

def benchmark(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    start = time.time()
    for _ in range(100):
        interpreter.invoke()
    end = time.time()

    avg_latency = (end - start) / 100
    size = os.path.getsize(model_path)

    return avg_latency, size

def main():
    print("Benchmarking models...\n")

    normal_latency, normal_size = benchmark(NORMAL_MODEL)
    quant_latency, quant_size = benchmark(QUANT_MODEL)

    print("Normal Model:")
    print(" Size:", normal_size, "bytes")
    print(" Avg Latency:", normal_latency, "seconds\n")

    print("Quantized Model:")
    print(" Size:", quant_size, "bytes")
    print(" Avg Latency:", quant_latency, "seconds\n")

if __name__ == "__main__":
    main()
