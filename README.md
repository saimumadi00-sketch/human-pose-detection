# Real-Time Human Behavior Detection

This project uses Ultralytics YOLO pose (`yolo11n-pose.pt`) to extract human pose
landmarks, maps the 17 COCO keypoints into 33 BlazePose-compatible slots, and
classifies 30-frame pose windows with a Keras LSTM model.

## Classes

The current behavior classifier uses five merged classes:

- `desk_work` - merged from typing and reading
- `on_phone`
- `idle_sitting`
- `consuming` - merged from eating and drinking
- `falling`

`desk_work` is displayed as WORKING during inference. The other four classes are
displayed as NOT WORKING.

## Setup

Use Python 3.13 from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first YOLO run may download `yolo11n-pose.pt`.

## Step 1: Collect Data

```bash
python collect_data.py
```

Collector key bindings:

- `1` = `desk_work`
- `2` = `on_phone`
- `3` = `idle_sitting`
- `4` = `consuming`
- `5` = `falling`

Wait for the window counter to reach `30/30`, then press the matching class key
to save one sample. The collector requires at least 150 samples per class before
`Q` is allowed to quit, and it prints a per-class count table on exit.

Saved files:

- `data/X.npy` with shape `(N, 30, 99)`
- `data/y.npy` with shape `(N,)`

## Step 2: Train

```bash
python train_model.py --epochs 100
```

Training saves:

- `models/pose_lstm.keras`
- `models/label_encoder.pkl`
- `models/training_curves.png`
- `models/confusion_matrix.png`

The `models/` folder needs `pose_lstm.keras` after training before inference can
run successfully.

## Step 3: Infer

```bash
python inference.py --source 0
```

Inference maintains a rolling 30-frame window, smooths predictions with a
five-prediction majority vote, overlays the pose skeleton, and shows a large
WORKING / NOT WORKING / UNKNOWN status banner. Press `Q` to quit.

## Optional Synthetic Generator

`generate_synthetic_data.py` is kept only for quick pipeline smoke tests. The
real training workflow should use webcam data collected with `collect_data.py`.
