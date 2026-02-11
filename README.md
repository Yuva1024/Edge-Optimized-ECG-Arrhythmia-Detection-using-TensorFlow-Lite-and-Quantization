# Edge-Optimized-ECG-Arrhythmia-Detection-using-TensorFlow-Lite-and-Quantization

This project demonstrates an edge-deployable ECG heartbeat classification pipeline using:

- TensorFlow
- TensorFlow Lite
- Post-training quantization
- TFLite Interpreter inference
- Latency + size benchmarking

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Yuva1024/Edge-Optimized-ECG-Arrhythmia-Detection-using-TensorFlow-Lite-and-Quantization.git
cd Edge-Optimized-ECG-Arrhythmia-Detection-using-TensorFlow-Lite-and-Quantization
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 1: Train the ECG Classification Model

Run:

```bash
python train.py
```

This will:

  - Load the ECG dataset

  - Train a lightweight neural network classifier

  - Save the model in TensorFlow SavedModel format

Saved model output:

```bash
models/ecg_saved_model/
```

  

### Step 2: Convert to TensorFlow Lite + Quantization

Run:
```bash
python convert_to_tflite.py
```

This generates:

  - ecg_model.tflite (standard model)

  - ecg_model_quant.tflite (INT8 quantized model)

Output files:

```bash
models/ecg_model.tflite
models/ecg_model_quant.tflite
```

### Step 3: Run Inference with TensorFlow Lite Interpreter
Run:

```bash
python run_interface.py
```
This will:

  - Load the quantized TFLite model

  - Run inference on a sample ECG heartbeat

  - Print predicted vs actual label

Example output:

```bash
Actual Label: 0
Predicted Label: 0
```

### Step 4: Benchmark Model Size and Latency
Run:

```bash
python benchmark.py
```
This compares:

  - Normal vs Quantized model size

  - Average inference latency

Example output:

```bash
Normal Model:
 Size: 120000 bytes
 Avg Latency: 0.015 sec

Quantized Model:
 Size: 75000 bytes
 Avg Latency: 0.009 sec
```



## Key Learning

  - Model export using SavedModel

  - TFLite conversion pipeline

  - INT8 quantization trade-offs

  - Efficient edge inference execution







