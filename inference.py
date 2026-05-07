"""Run real-time behavior classification from webcam using trained LSTM model."""

from __future__ import annotations

import argparse
import os
import pickle
import time
from collections import Counter, deque
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
import tensorflow as tf

try:
    from .utils.pose_utils import (
        DEFAULT_TASK_MODEL_PATH,
        DEFAULT_YOLO_POSE_MODEL,
        LANDMARK_VECTOR_SIZE,
        count_visible_landmarks,
        create_pose_estimator,
        draw_pose_landmarks,
        extract_landmarks,
        process_bgr_frame,
    )
except ImportError:
    from utils.pose_utils import (
        DEFAULT_TASK_MODEL_PATH,
        DEFAULT_YOLO_POSE_MODEL,
        LANDMARK_VECTOR_SIZE,
        count_visible_landmarks,
        create_pose_estimator,
        draw_pose_landmarks,
        extract_landmarks,
        process_bgr_frame,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
WORKING_CLASSES = {"desk_work"}
NOT_WORKING_CLASSES = {"on_phone", "idle_sitting", "consuming", "falling"}
EXPECTED_CLASSES = WORKING_CLASSES | NOT_WORKING_CLASSES
WORKING_STATUS_TEXT = "● WORKING"
NOT_WORKING_STATUS_TEXT = "✗ NOT WORKING"
UNKNOWN_STATUS_TEXT = "? UNKNOWN"


def parse_source(source: str):
    """Convert source string into int camera index when numeric."""
    return int(source) if source.isdigit() else source


def format_top_predictions(probs: np.ndarray, classes, top_k: int = 3) -> str:
    """Format top model probabilities for terminal debugging."""
    top_ids = np.argsort(probs)[-top_k:][::-1]
    return ", ".join(f"{classes[idx]}={probs[idx] * 100:.1f}%" for idx in top_ids)


def load_label_classes(encoder_path: Path) -> np.ndarray:
    """Load class labels from the current lightweight format or old LabelEncoder files."""
    with open(encoder_path, "rb") as f:
        label_data = pickle.load(f)

    if isinstance(label_data, dict):
        classes = label_data.get("classes")
        if classes is None:
            classes = label_data.get("classes_")
    elif hasattr(label_data, "classes_"):
        classes = label_data.classes_
    elif isinstance(label_data, (list, tuple, np.ndarray)):
        classes = label_data
    else:
        classes = None

    if classes is None:
        raise ValueError(f"Could not load class labels from: {encoder_path}")

    classes = np.asarray(classes).astype(str)
    if classes.size == 0:
        raise ValueError(f"No class labels found in: {encoder_path}")
    return classes


def validate_behavior_classes(classes: np.ndarray) -> None:
    """Ensure inference is running against the merged five-class classifier."""
    class_set = set(classes.astype(str).tolist())
    if class_set != EXPECTED_CLASSES:
        missing = sorted(EXPECTED_CLASSES - class_set)
        extra = sorted(class_set - EXPECTED_CLASSES)
        parts = []
        if missing:
            parts.append(f"missing: {', '.join(missing)}")
        if extra:
            parts.append(f"unexpected: {', '.join(extra)}")
        detail = "; ".join(parts)
        raise ValueError(
            "Inference expects the five merged behavior classes "
            f"({', '.join(sorted(EXPECTED_CLASSES))}); {detail}. "
            "Collect real data and retrain the LSTM before running inference."
        )


def smooth_prediction(prediction_history: deque[str], latest_label: str) -> str:
    """Return the majority/plurality label, falling back to latest on ties/no repeats."""
    prediction_history.append(latest_label)
    counts = Counter(prediction_history)
    top_count = max(counts.values())
    top_labels = [label for label, count in counts.items() if count == top_count]
    if top_count == 1 or len(top_labels) > 1:
        return latest_label
    return top_labels[0]


def work_status_for_label(label: str | None, is_confident: bool) -> tuple[str, tuple[int, int, int]]:
    """Map behavior label to banner text and BGR color."""
    if is_confident and label in WORKING_CLASSES:
        return WORKING_STATUS_TEXT, (35, 150, 50)
    if is_confident and label in NOT_WORKING_CLASSES:
        return NOT_WORKING_STATUS_TEXT, (35, 35, 190)
    return UNKNOWN_STATUS_TEXT, (110, 110, 110)


def draw_status_banner(frame: np.ndarray, status: str, color: tuple[int, int, int]) -> None:
    """Draw a large working-status banner at the top of the frame."""
    height, width = frame.shape[:2]
    banner_height = max(62, min(90, height // 7))
    cv2.rectangle(frame, (0, 0), (width, banner_height), color, -1)

    icon_center = (30, banner_height // 2 + 1)
    if status == WORKING_STATUS_TEXT:
        cv2.circle(frame, icon_center, 10, (245, 245, 245), -1)
        text = "WORKING"
        text_origin = (54, banner_height // 2 + 13)
    elif status == NOT_WORKING_STATUS_TEXT:
        cv2.line(frame, (20, icon_center[1] - 10), (40, icon_center[1] + 10), (245, 245, 245), 4)
        cv2.line(frame, (40, icon_center[1] - 10), (20, icon_center[1] + 10), (245, 245, 245), 4)
        text = "NOT WORKING"
        text_origin = (54, banner_height // 2 + 13)
    else:
        text = "? UNKNOWN"
        text_origin = (18, banner_height // 2 + 13)

    cv2.putText(
        frame,
        text,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.22,
        (255, 255, 255),
        3,
    )


def validate_model_contract(model: tf.keras.Model, classes: np.ndarray, window: int) -> None:
    """Fail early when inference inputs do not match the trained model."""
    input_shape = model.input_shape
    output_shape = model.output_shape
    if not isinstance(input_shape, tuple) or len(input_shape) != 3:
        raise ValueError(f"Expected model input shape (None, window, features), got {input_shape}.")

    model_window = input_shape[1]
    feature_dim = input_shape[2]
    if model_window is not None and model_window != window:
        raise ValueError(
            f"Model expects window={model_window}, but inference was started with --window {window}."
        )
    if feature_dim != LANDMARK_VECTOR_SIZE:
        raise ValueError(
            f"Model expects {feature_dim} features, but pose extraction produces "
            f"{LANDMARK_VECTOR_SIZE}."
        )

    if not isinstance(output_shape, tuple) or output_shape[-1] != len(classes):
        raise ValueError(
            f"Model output shape {output_shape} does not match {len(classes)} encoder classes."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time behavior detection inference.")
    parser.add_argument("--source", type=str, default="0", help="Webcam index or video path/URL.")
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Number of consecutive frames required for prediction.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=SCRIPT_DIR / "models" / "pose_lstm.keras",
        help="Path to trained Keras model.",
    )
    parser.add_argument(
        "--encoder-path",
        type=Path,
        default=SCRIPT_DIR / "models" / "label_encoder.pkl",
        help="Path to saved class-label pickle.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum detection confidence.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum tracking confidence.",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="MediaPipe BlazePose model complexity.",
    )
    parser.add_argument(
        "--pose-task-model",
        type=Path,
        default=DEFAULT_TASK_MODEL_PATH,
        help=(
            "Path to pose_landmarker_lite.task for Python 3.13 MediaPipe Tasks. "
            "Ignored when mp.solutions.pose is available."
        ),
    )
    parser.add_argument(
        "--pose-backend",
        choices=["auto", "mediapipe", "mediapipe-tasks", "yolo"],
        default="yolo",
        help=(
            "Pose backend. 'auto' uses classic MediaPipe when available and "
            "YOLO on Python 3.13."
        ),
    )
    parser.add_argument(
        "--yolo-pose-model",
        default=DEFAULT_YOLO_POSE_MODEL,
        help="Ultralytics YOLO pose model for the YOLO backend.",
    )
    parser.add_argument(
        "--yolo-device",
        default="cpu",
        help="Device for YOLO pose inference, for example 'cpu' or '0'.",
    )
    parser.add_argument(
        "--min-valid-landmarks",
        type=int,
        default=8,
        help=(
            "Minimum visible pose landmarks required before adding a frame to "
            "the prediction window."
        ),
    )
    parser.add_argument(
        "--min-prediction-confidence",
        type=float,
        default=0.45,
        help="Show uncertain below this top-class probability.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Number of recent predictions used for majority-vote smoothing.",
    )
    parser.add_argument(
        "--debug-predictions",
        action="store_true",
        help="Print pose quality and top predictions once per second.",
    )
    return parser.parse_args()


def validate_runtime_args(args: argparse.Namespace) -> None:
    """Validate inference options before loading models or opening video sources."""
    if args.window <= 0:
        raise ValueError("--window must be > 0.")
    if args.smooth_window <= 0:
        raise ValueError("--smooth-window must be > 0.")
    max_landmarks = LANDMARK_VECTOR_SIZE // 3
    if not 1 <= args.min_valid_landmarks <= max_landmarks:
        raise ValueError(f"--min-valid-landmarks must be between 1 and {max_landmarks}.")
    if not 0.0 <= args.min_prediction_confidence <= 1.0:
        raise ValueError("--min-prediction-confidence must be between 0 and 1.")
    if not 0.0 <= args.min_detection_confidence <= 1.0:
        raise ValueError("--min-detection-confidence must be between 0 and 1.")
    if not 0.0 <= args.min_tracking_confidence <= 1.0:
        raise ValueError("--min-tracking-confidence must be between 0 and 1.")


def main() -> None:
    args = parse_args()
    validate_runtime_args(args)

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.encoder_path.exists():
        raise FileNotFoundError(f"Class-label file not found: {args.encoder_path}")

    # Load trained model and class labels.
    model = tf.keras.models.load_model(args.model_path)
    classes = load_label_classes(args.encoder_path)
    validate_behavior_classes(classes)
    validate_model_contract(model, classes, args.window)

    sequence_buffer = deque(maxlen=args.window)
    prediction_history = deque(maxlen=args.smooth_window)
    prediction_label = "N/A"
    prediction_confidence = 0.0
    status_text = "UNKNOWN"
    status_color = (110, 110, 110)
    last_debug_at = 0.0

    mp_pose, mp_drawing, mp_drawing_styles, pose = create_pose_estimator(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        model_complexity=args.model_complexity,
        pose_task_model_path=args.pose_task_model,
        pose_backend=args.pose_backend,
        yolo_pose_model=args.yolo_pose_model,
        yolo_device=args.yolo_device,
    )

    print(f"Loaded model: {args.model_path}")
    print(f"Loaded class labels: {args.encoder_path}")
    print(f"Model input/output: {model.input_shape} -> {model.output_shape}")
    print(f"Classes: {', '.join(str(class_name) for class_name in classes)}")
    print(f"Pose backend object: {pose.__class__.__name__}")

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open video source: {args.source}. "
            "If this is a webcam, check that it is connected and visible with "
            "`ls /dev/video*`, or try another source such as --source 1."
        )

    print("Inference started. Press 'q' to quit.")
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Frame read failed. Exiting.")
                break

            # Run pose estimation and update rolling feature window.
            results = process_bgr_frame(frame, pose)
            valid_landmark_count = count_visible_landmarks(results)
            top_predictions = None
            if valid_landmark_count >= args.min_valid_landmarks:
                landmarks = extract_landmarks(results)
                sequence_buffer.append(landmarks)
            else:
                sequence_buffer.clear()
                prediction_history.clear()
                prediction_label = "No pose"
                prediction_confidence = 0.0
                status_text, status_color = work_status_for_label(None, False)

            # Predict once the rolling sequence reaches the required window length.
            if len(sequence_buffer) == args.window:
                input_tensor = np.expand_dims(np.array(sequence_buffer, dtype=np.float32), axis=0)
                probs = model.predict(input_tensor, verbose=0)[0]
                class_idx = int(np.argmax(probs))
                prediction_confidence = float(probs[class_idx])
                raw_label = str(classes[class_idx])
                smoothed_label = smooth_prediction(prediction_history, raw_label)
                is_confident = prediction_confidence >= args.min_prediction_confidence
                if prediction_confidence < args.min_prediction_confidence:
                    prediction_label = f"uncertain ({smoothed_label})"
                else:
                    prediction_label = smoothed_label
                status_text, status_color = work_status_for_label(smoothed_label, is_confident)
                top_predictions = format_top_predictions(probs, classes)

            if args.debug_predictions:
                now = time.monotonic()
                if now - last_debug_at >= 1.0:
                    debug_parts = [
                        f"valid_landmarks={valid_landmark_count}",
                        f"window={len(sequence_buffer)}/{args.window}",
                        f"label={prediction_label}",
                    ]
                    if top_predictions is not None:
                        debug_parts.append(f"top={top_predictions}")
                    print(f"debug: {', '.join(debug_parts)}")
                    last_debug_at = now

            # Draw pose skeleton and prediction text overlays.
            draw_pose_landmarks(frame, results, mp_pose, mp_drawing, mp_drawing_styles)
            draw_status_banner(frame, status_text, status_color)
            cv2.putText(
                frame,
                f"Prediction: {prediction_label}",
                (10, 102),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Confidence: {prediction_confidence * 100:.2f}%",
                (10, 134),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Window: {len(sequence_buffer)}/{args.window}  Landmarks: {valid_landmark_count}",
                (10, 166),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Behavior Detection (Q to quit)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        pose.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
