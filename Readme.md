---

### Video Preprocessing for MViT

For training the **Multiscale Vision Transformer (MViT)** model, videos are preprocessed into fixed-length clips with a consistent spatial and temporal format. Each video is decoded using a video reader and uniformly sampled to **16 frames per clip** across the full temporal duration. If a video contains fewer than 16 frames, the last frame is repeated to pad the clip, ensuring a fixed temporal dimension. Frames are converted to RGB format (grayscale videos are expanded to three channels), normalized to the ([0,1]) range, and rearranged into the tensor shape **(C, T, H, W)** as required by MViT. Optional spatial transforms such as resizing, normalization, and horizontal flipping are applied consistently across all frames in a clip to preserve temporal coherence. This preprocessing pipeline ensures that all video samples are temporally aligned, channel-consistent, and compatible with the input requirements of the MViT architecture.

---

## Video Preprocessing Pipeline (Bullet-Point Summary)

* **Video decoding**

  * Videos are read using a video reader and loaded entirely into memory for frame indexing.

* **Temporal sampling**

  * Each video is uniformly sampled to a fixed number of frames (`frames_per_clip`, e.g., 16).
  * If a video contains fewer frames than required, the last frame is repeated to pad the clip.

* **Channel consistency**

  * All videos are converted to **RGB format**.
  * Grayscale videos (single-channel) are expanded to three channels.

* **Normalization**

  * Pixel values are scaled from `[0, 255]` to `[0, 1]`.

* **Tensor formatting**

  * Frames are rearranged from `(T, H, W, C)` to **`(C, T, H, W)`**, which is the required input format for MViT.

* **Spatial transformations (optional)**

  * Consistent spatial transforms (resize, normalization, horizontal flip) are applied uniformly across all frames in a clip to preserve temporal coherence.

* **Robust error handling**

  * If a video fails to load, the dataset safely skips it and loads the next available sample.

---

## Input Tensor Format Explanation: `(C, T, H, W)`

The MViT model expects video clips in the **channel-first, time-aware** tensor format `(C, T, H, W)`:

```
(C, T, H, W)
 │   │   │   └─ Frame width  (e.g., 224 pixels)
 │   │   └───── Frame height (e.g., 224 pixels)
 │   └───────── Temporal dimension (number of frames, e.g., 16)
 └───────────── Channels (RGB → 3)
```

### Example

For a video clip sampled to **16 frames** at **224×224 resolution**:

```
Input Tensor Shape: (3, 16, 224, 224)
```

* **C = 3** → RGB color channels
* **T = 16** → Temporally sampled frames
* **H = 224** → Frame height
* **W = 224** → Frame width

This format allows MViT to:

* Capture **spatiotemporal relationships**
* Apply **multiscale attention across time and space**
* Efficiently process video clips as structured 4D tensors

---

# How to Run the MViT Scripts

This repository provides scripts for **training**, **evaluation**, **model export**, and **real-time inference** using a video-based **Multiscale Vision Transformer (MViT)** model.

---

## 1. Environment Setup

### 1.1 Create a Virtual Environment

From the project root directory:

```bash
python3 -m venv virenv
source virenv/bin/activate
```

> You should see `(virenv)` in your terminal once activated.

---

### 1.2 Install Dependencies

Install all required packages using the provided `requirements.txt` file:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Training the Model

### Script

```text
train_MViT.py
```

### Description

* Trains an MViT model using the specified dataset.
* Uses **MLflow** to log:

  * Training & validation accuracy
  * Training & validation loss
  * Optimizer, learning rate, weight decay, dropout, epochs, etc.
* Automatically **saves the best model** based on **lowest validation loss**.
* The best model is stored as an MLflow artifact under the corresponding run.

### Command

```bash
python train_MViT.py
```

After training:

* Note the **MLflow Run ID** (printed in the console or visible in the MLflow UI).
* This Run ID is required for testing, model export, and inference.

---

## 3. Evaluating the Model on the Test Set

### Script

```text
test_MViT.py
```

### Description

* Loads the **best saved model** from MLflow artifacts.
* Evaluates performance on the test dataset.
* Reports **test accuracy** and other evaluation metrics.

### Command

```bash
python test_MViT.py --run_id <MLFLOW_RUN_ID>
```

Example:

```bash
python test_MViT.py --run_id 106e4008a54a4baa92886afbb6457432
```

---

## 4. Converting MLflow Model to `.pt` File

### Script

```text
mlflow2model.py
```

### Description

* Extracts the trained model from MLflow artifacts.
* Converts and saves it as a **PyTorch `.pt` file**.
* Useful for deployment, inference, or integration with other pipelines.

### Command

```bash
python mlflow2model.py --run_id <MLFLOW_RUN_ID> --output_path model.pt
```

Example:

```bash
python mlflow2model.py --run_id 106e4008a54a4baa92886afbb6457432 --output_path best_mvit.pt
```

---

## 5. Real-Time Inference Using RTSP Stream

### Script

```text
rtsp_stream_optimised.py
```

### Description

* Loads the trained MViT model.
* Performs **real-time action recognition** on an RTSP video stream.
* Useful for validating real-world performance beyond offline test accuracy.

## RTSP Inference Configuration

The real-time inference script (`rtsp_stream_optimised.py`) is configured through a central parameter block.
Before running inference, the following parameters **must be updated to match the trained model and deployment setup**.

```python
# -----------------------------
# CONFIGURATION
# -----------------------------
RTSP_URL = "rtsp://admin:admin%40123@192.168.0.110:554/stream1"
BATCH_SIZE = 64
TARGET_FPS = 25
OUTPUT_FOLDER = "/home/smartan5070/Downloads/SlowfastTrainer-main/unseen_test/cropped_frames"
TRAIN_ROOT = train_datapath
MODEL_PATH = "/home/smartan5070/Downloads/SlowfastTrainer-main/downloaded_models/31_class_model_acc80_17_12.pt"
NUM_CLASSES = 31
K = 5
```

---

## Parameter Explanation

* **RTSP_URL**

  * RTSP stream address of the live camera feed.
  * Must be updated when switching cameras or deployment environments.

* **BATCH_SIZE**

  * Number of video clips processed together during inference.
  * Higher values improve throughput but increase GPU memory usage.

* **TARGET_FPS**

  * Frames per second sampled from the RTSP stream.
  * Should align with the temporal assumptions used during training.

* **OUTPUT_FOLDER**

  * Directory used to temporarily store extracted or cropped frames.
  * Ensure the path exists and has write permissions.

* **TRAIN_ROOT**

  * Path to the training dataset root.
  * Used to load class mappings (`class_to_idx`) to ensure label consistency.

* **MODEL_PATH**

  * Path to the exported `.pt` model file (converted from MLflow).
  * **This must match the trained model architecture and number of classes.**

* **NUM_CLASSES**

  * Number of action classes the model was trained on.
  * Must exactly match the value used during training.



### Command

```bash
python rtsp_stream_optimised.py

Example:

```bash
python rtsp_stream_optimised.py
```

---

## 6. MLflow Tracking (Optional but Recommended)

To view training logs, metrics, and artifacts:

```bash
mlflow ui
```

Then open:

```
http://localhost:5000
```

From the MLflow UI, you can:

* Compare experiments
* Visualize training/validation curves
* Retrieve Run IDs
* Download model artifacts

---

## Summary Workflow

```text
1. Create virtual environment
2. Install dependencies
3. Train model (train_MViT.py)
4. Evaluate model (test_MViT.py)
5. Export model to .pt (mlflow2model.py)
6. Run real-time inference (rtsp_stream_optimised.py)
```

---
