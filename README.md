# Real-Time Human Behavior Detection

This project detects a person's behavior from webcam pose motion. It uses a pose
estimator to extract body landmarks from each video frame, builds a rolling
30-frame sequence, and classifies that sequence with a Keras LSTM model.

The default live pipeline uses Ultralytics YOLO pose (`yolo11n-pose.pt`). YOLO
returns 17 COCO keypoints, and `utils/pose_utils.py` maps those points into
33 BlazePose-compatible landmark slots. Each frame becomes a 99-value vector:

```text
33 landmarks * 3 values (x, y, z) = 99 features
```

The LSTM model therefore expects input with this shape:

```text
(batch, 30, 99)
```

## Behavior Classes

The current classifier uses five behavior classes:

| Class | Meaning | Inference status |
| --- | --- | --- |
| `desk_work` | working at a desk, typing, mouse use, screen-focused work | WORKING |
| `on_phone` | phone use, phone near ear, texting posture | NOT WORKING |
| `idle_sitting` | sitting still without active desk work | NOT WORKING |
| `consuming` | eating or drinking motion | NOT WORKING |
| `falling` | falling, fallen, or lying/transition pose | NOT WORKING |

Legacy seven-class labels are mapped into this five-class taxonomy during data
loading and training:

| Old label | New label |
| --- | --- |
| `typing` | `desk_work` |
| `reading` | `desk_work` |
| `eating` | `consuming` |
| `drinking` | `consuming` |

Inference only displays `WORKING` when the confident smoothed prediction is
`desk_work`. The other four classes display `NOT WORKING`. Low-confidence
predictions, missing pose landmarks, or incomplete frame windows display
`UNKNOWN` or `uncertain`.

## Project Files

```text
collect_data.py              Collect real webcam pose sequences into data/X.npy and data/y.npy
train_model.py               Train the LSTM behavior classifier
inference.py                 Run real-time webcam inference
generate_synthetic_data.py   Generate synthetic data for smoke tests only
utils/pose_utils.py          Shared pose extraction, landmark mapping, and drawing utilities
train_yolo_pose.py           Optional YOLO pose training helper, not required for behavior training
requirements.txt             Python dependencies
data/X.npy                   Collected pose sequence dataset
data/y.npy                   Collected behavior labels
models/pose_lstm.keras       Trained LSTM classifier
models/label_encoder.pkl     sklearn LabelEncoder matching the model output order
models/training_curves.png   Training accuracy/loss plot
models/confusion_matrix.png  Held-out test confusion matrix from training
```

## Environment Setup

Use Python 3.10, 3.11, or 3.12 for the local environment. This workspace has
been verified with Python 3.12.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python` is not found on your machine, use `.venv/bin/python` in the commands
below.

The first YOLO pose run may download `yolo11n-pose.pt`.

## Current Artifact Contract

The model and encoder must match each other. The expected contract is:

```text
models/pose_lstm.keras input:  (None, 30, 99)
models/pose_lstm.keras output: (None, 5)
models/label_encoder.pkl:      sklearn LabelEncoder with 5 classes
```

Verify the local artifacts with:

```bash
python - <<'PY'
import pickle
import tensorflow as tf

model = tf.keras.models.load_model("models/pose_lstm.keras")
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("Model:", model.input_shape, "->", model.output_shape)
print("Classes:", label_encoder.classes_)
PY
```

Expected classes:

```text
consuming
desk_work
falling
idle_sitting
on_phone
```

## Collect Real Webcam Data

Use this step when you need to create or improve the training dataset.

```bash
python collect_data.py --source 0
```

If camera `0` is not correct, try another index:

```bash
python collect_data.py --source 1
```

The collection window shows the detected skeleton, current class, window fill,
landmark count, and saved sample counts.

Key bindings:

```text
1 = desk_work
2 = on_phone
3 = idle_sitting
4 = consuming
5 = falling
q = quit after every class reaches the minimum sample count
```

How to save one sample:

1. Perform the target behavior in front of the camera.
2. Wait until the overlay shows `Window: 30/30`.
3. Press the matching number key.
4. The current 30-frame pose window is appended to the dataset.

By default, the collector requires at least 150 samples per class before `q`
can quit. The saved dataset files are:

```text
data/X.npy  shape: (N, 30, 99)
data/y.npy  shape: (N,)
```

Collect varied examples for better inference:

- Change lighting, distance, and body angle.
- Use natural motion instead of holding one frozen pose.
- Capture both left-hand and right-hand versions when relevant.
- For `falling`, simulate safely. Do not perform dangerous falls.
- Keep the same pose backend for collection and inference. This project uses
  YOLO pose by default for both.

## Train Locally

Train the LSTM classifier from `data/X.npy` and `data/y.npy`:

```bash
python train_model.py --epochs 100
```

Training does the following:

- Loads `data/X.npy` and `data/y.npy`
- Maps legacy labels into the five-class taxonomy
- Validates shape `(N, 30, 99)`
- Requires all five behavior classes
- Uses a stratified train/test split
- Applies balanced class weights
- Saves the trained model, label encoder, training curves, and confusion matrix

Output files:

```text
models/pose_lstm.keras
models/label_encoder.pkl
models/training_curves.png
models/confusion_matrix.png
```

## Train on Google Colab

Use Colab if local training is slow. The webcam collection should still be done
locally, then the collected `X.npy` and `y.npy` files can be uploaded to Colab.

### Cell 1: Clone

```python
!git clone https://github.com/saimumadi00-sketch/human-pose-detection.git
%cd human-pose-detection
```

### Cell 2: Install

```python
!pip install -q -r requirements.txt
```

### Cell 3: Upload Data

```python
from google.colab import files
uploaded = files.upload()
```

Upload:

```text
X.npy
y.npy
```

### Cell 4: Move Data

```python
!mkdir -p data
!mv X.npy data/X.npy
!mv y.npy data/y.npy
```

### Cell 5: Check Data

```python
import numpy as np

X = np.load("data/X.npy")
y = np.load("data/y.npy", allow_pickle=True).astype(str)

print("X shape:", X.shape)
print("y shape:", y.shape)

labels, counts = np.unique(y, return_counts=True)
print(dict(zip(labels, counts)))
```

Expected `X` shape:

```text
(N, 30, 99)
```

Expected labels:

```text
consuming
desk_work
falling
idle_sitting
on_phone
```

### Cell 6: Train

```python
!python train_model.py --epochs 100
```

### Cell 7: Download Results

```python
!zip -j trained_behavior_model.zip models/pose_lstm.keras models/label_encoder.pkl models/training_curves.png models/confusion_matrix.png

from google.colab import files
files.download("trained_behavior_model.zip")
```

After downloading, replace these local files with the Colab-trained versions:

```text
models/pose_lstm.keras
models/label_encoder.pkl
models/training_curves.png
models/confusion_matrix.png
```

## Run Inference

Run webcam inference locally:

```bash
python inference.py --source 0 --debug-predictions
```

If camera `0` is not correct:

```bash
python inference.py --source 1 --debug-predictions
```

Inference behavior:

- Opens a webcam/video source with OpenCV
- Extracts pose landmarks from each frame
- Requires at least 8 visible landmarks by default
- Builds a rolling 30-frame sequence
- Runs the LSTM once the window is full
- Smooths labels with a 5-prediction majority vote
- Treats confidence below `0.45` as uncertain
- Draws the pose skeleton and WORKING / NOT WORKING / UNKNOWN banner
- Prints debug predictions once per second when `--debug-predictions` is used

Press `q` in the video window to quit.

Useful inference options:

```bash
python inference.py --source 0 --debug-predictions
python inference.py --source 0 --min-prediction-confidence 0.60
python inference.py --source 0 --min-valid-landmarks 12
python inference.py --source path/to/video.mp4 --debug-predictions
```

## Synthetic Data

`generate_synthetic_data.py` is only for pipeline smoke tests. It creates
simple artificial pose sequences for the same five classes:

```bash
python generate_synthetic_data.py --samples-per-class 5 \
  --x-path /tmp/synthetic_X.npy \
  --y-path /tmp/synthetic_y.npy
```

Do not use synthetic data as the final training dataset for webcam inference.
Real collected webcam data is required for reliable behavior classification.

## Optional YOLO Pose Training

`train_yolo_pose.py` is a helper for training an Ultralytics YOLO pose model on a
COCO-Pose style dataset. It is not required for the behavior classifier workflow.
The normal project workflow uses the pretrained `yolo11n-pose.pt` pose model and
trains only the LSTM behavior classifier.

## Troubleshooting

If the pose skeleton is wrong:

- Improve lighting and make sure the full upper body is visible.
- Try a different camera index with `--source 1`.
- Increase `--min-detection-confidence` if false detections appear.
- Keep the camera steady and reduce background clutter.

If behavior predictions are wrong:

- Run with `--debug-predictions` and inspect the top probabilities.
- Verify `models/pose_lstm.keras` and `models/label_encoder.pkl` came from the
  same training run.
- Confirm the encoder has exactly the five expected classes.
- Collect more real samples for weak classes.
- Keep class counts reasonably balanced.
- Retrain after adding data.
- Use the same pose backend for collection and inference.

If inference says `No pose` or stays `UNKNOWN`:

- Check that the video window shows a visible skeleton.
- Lower `--min-valid-landmarks` only if the skeleton is mostly correct.
- Make sure the person is within the camera frame.

## Submission Checklist

Before presenting or submitting:

```bash
python - <<'PY'
import pickle
import numpy as np
import tensorflow as tf

X = np.load("data/X.npy")
y = np.load("data/y.npy", allow_pickle=True).astype(str)
model = tf.keras.models.load_model("models/pose_lstm.keras")
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("X:", X.shape)
print("y:", y.shape)
print("Model:", model.input_shape, "->", model.output_shape)
print("Classes:", label_encoder.classes_)
print("Counts:", dict(zip(*np.unique(y, return_counts=True))))
PY
```

Expected:

```text
X shape is (N, 30, 99)
y has N labels
model output has 5 classes
label_encoder.pkl has the same 5 classes as inference.py
models/pose_lstm.keras exists
models/label_encoder.pkl exists
models/training_curves.png exists
models/confusion_matrix.png exists
```
