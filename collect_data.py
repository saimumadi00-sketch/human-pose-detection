"""Collect real YOLO pose sequences from webcam and save them for training."""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np

try:
    from .utils.pose_utils import (
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
        DEFAULT_YOLO_POSE_MODEL,
        LANDMARK_VECTOR_SIZE,
        count_visible_landmarks,
        create_pose_estimator,
        draw_pose_landmarks,
        extract_landmarks,
        process_bgr_frame,
    )


DEFAULT_CLASSES = [
    "desk_work",
    "on_phone",
    "idle_sitting",
    "consuming",
    "falling",
]
LEGACY_LABEL_MAP = {
    "typing": "desk_work",
    "reading": "desk_work",
    "eating": "consuming",
    "drinking": "consuming",
}
MIN_SAMPLES_PER_CLASS = 150
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_source(source: str):
    """Convert camera source to int when possible, otherwise keep as path/URL."""
    return int(source) if source.isdigit() else source


def duplicate_labels(labels: List[str]) -> List[str]:
    """Return labels that appear more than once while preserving first duplicate order."""
    seen = set()
    duplicates = []
    for label in labels:
        if label in seen and label not in duplicates:
            duplicates.append(label)
        seen.add(label)
    return duplicates


def build_key_map(classes: List[str]) -> Dict[str, str]:
    """Map keyboard keys ('1', '2', ...) to class labels."""
    if not classes:
        raise ValueError("--classes cannot be empty.")
    duplicates = duplicate_labels(classes)
    if duplicates:
        labels = ", ".join(duplicates)
        raise ValueError(f"--classes cannot contain duplicates: {labels}.")
    if len(classes) > 9:
        raise ValueError("Up to 9 classes are supported for numeric key bindings.")
    return {str(i + 1): label for i, label in enumerate(classes)}


def normalize_label(label: str) -> str:
    """Map old seven-class labels into the current five-class taxonomy."""
    label = str(label)
    return LEGACY_LABEL_MAP.get(label, label)


def count_by_class(labels: np.ndarray, classes: List[str]) -> Dict[str, int]:
    """Count saved examples for each configured class."""
    return {class_name: int(np.sum(labels == class_name)) for class_name in classes}


def load_existing_dataset(
    x_path: Path, y_path: Path, window: int, allowed_classes: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Load an existing dataset if available, mapping legacy labels when needed."""
    if x_path.exists() != y_path.exists():
        raise FileNotFoundError(
            f"Expected both dataset files to exist together: {x_path} and {y_path}"
        )

    if x_path.exists() and y_path.exists():
        x_data = np.load(x_path).astype(np.float32)
        y_data = np.load(y_path, allow_pickle=True).astype(str)
        y_data = np.array([normalize_label(label) for label in y_data], dtype=str)
        if x_data.ndim != 3 or x_data.shape[1:] != (window, LANDMARK_VECTOR_SIZE):
            raise ValueError(
                f"Existing X shape {x_data.shape} does not match required "
                f"(N, {window}, {LANDMARK_VECTOR_SIZE})."
            )
        if len(x_data) != len(y_data):
            raise ValueError(f"Existing dataset mismatch: X={len(x_data)}, y={len(y_data)}.")
        unknown_labels = sorted(set(y_data.tolist()) - set(allowed_classes))
        if unknown_labels:
            labels = ", ".join(unknown_labels)
            allowed = ", ".join(allowed_classes)
            raise ValueError(
                f"Existing dataset contains labels outside --classes: {labels}. "
                f"Allowed labels: {allowed}."
            )
        return x_data, y_data

    x_data = np.empty((0, window, LANDMARK_VECTOR_SIZE), dtype=np.float32)
    y_data = np.empty((0,), dtype=str)
    return x_data, y_data


def save_dataset(x_data: np.ndarray, y_data: np.ndarray, x_path: Path, y_path: Path) -> None:
    """Persist dataset arrays to disk."""
    x_path.parent.mkdir(parents=True, exist_ok=True)
    y_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(x_path, x_data)
    np.save(y_path, y_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect real pose sequences for behavior classes.")
    parser.add_argument("--source", type=str, default="0", help="Webcam index or video source.")
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Number of consecutive frames per saved sample.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Behavior classes to capture. Key bindings are assigned as 1..N.",
    )
    parser.add_argument(
        "--x-path",
        type=Path,
        default=SCRIPT_DIR / "data" / "X.npy",
        help="Output path for X sequences.",
    )
    parser.add_argument(
        "--y-path",
        type=Path,
        default=SCRIPT_DIR / "data" / "y.npy",
        help="Output path for y labels.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="YOLO pose detection confidence threshold.",
    )
    parser.add_argument(
        "--yolo-pose-model",
        default=DEFAULT_YOLO_POSE_MODEL,
        help="Ultralytics YOLO pose model.",
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
        help="Minimum visible pose landmarks required before adding a frame to the window.",
    )
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=MIN_SAMPLES_PER_CLASS,
        help="Minimum saved samples per class required before Q can quit.",
    )
    return parser.parse_args()


def validate_runtime_args(args: argparse.Namespace) -> None:
    """Validate collection options before camera and model initialization."""
    if args.window <= 0:
        raise ValueError("--window must be > 0.")
    max_landmarks = LANDMARK_VECTOR_SIZE // 3
    if not 1 <= args.min_valid_landmarks <= max_landmarks:
        raise ValueError(f"--min-valid-landmarks must be between 1 and {max_landmarks}.")
    if not 0.0 <= args.min_detection_confidence <= 1.0:
        raise ValueError("--min-detection-confidence must be between 0 and 1.")
    if args.min_samples_per_class < 0:
        raise ValueError("--min-samples-per-class must be >= 0.")


def minimums_met(counts: Dict[str, int], min_samples: int) -> bool:
    """Return True when every class has enough examples to end collection."""
    return all(count >= min_samples for count in counts.values())


def missing_minimum_text(counts: Dict[str, int], min_samples: int) -> str:
    """Format the classes that still need more samples."""
    missing = [
        f"{label} {count}/{min_samples}"
        for label, count in counts.items()
        if count < min_samples
    ]
    return ", ".join(missing)


def print_count_table(counts: Dict[str, int], min_samples: int) -> None:
    """Print final per-class sample counts."""
    print("\nPer-class sample counts:")
    print("------------------------")
    for label, count in counts.items():
        status = "OK" if count >= min_samples else "NEEDS MORE"
        print(f"{label:<14} {count:>5} / {min_samples:<5} {status}")


def draw_collection_overlay(
    frame: np.ndarray,
    key_map: Dict[str, str],
    counts: Dict[str, int],
    active_label: str,
    window_fill: int,
    window_size: int,
    valid_landmark_count: int,
    min_samples: int,
    warning: str | None,
) -> None:
    """Draw live class, window, and per-class count information."""
    height, width = frame.shape[:2]
    panel_height = min(height, 178 + 28 * len(key_map))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    cv2.putText(
        frame,
        f"Class: {active_label}",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Window: {window_fill}/{window_size}  Landmarks: {valid_landmark_count}",
        (12, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "Saved samples:",
        (12, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
    )

    y = 126
    for key, label in key_map.items():
        count = counts[label]
        color = (0, 220, 0) if count >= min_samples else (0, 180, 255)
        cv2.putText(
            frame,
            f"{key} = {label:<12} {count}/{min_samples}",
            (28, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
        )
        y += 28

    if warning:
        cv2.rectangle(frame, (0, height - 46), (width, height), (0, 120, 255), -1)
        cv2.putText(
            frame,
            warning[:96],
            (12, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )


def main() -> None:
    args = parse_args()
    validate_runtime_args(args)

    key_map = build_key_map(args.classes)
    x_data, y_data = load_existing_dataset(
        args.x_path,
        args.y_path,
        args.window,
        args.classes,
    )
    sequence_buffer = deque(maxlen=args.window)
    active_label = args.classes[0]
    warning_message: str | None = None
    warning_until = 0.0

    mp_pose, mp_drawing, mp_drawing_styles, pose = create_pose_estimator(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_detection_confidence,
        model_complexity=1,
        pose_backend="yolo",
        yolo_pose_model=args.yolo_pose_model,
        yolo_device=args.yolo_device,
    )

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        pose.close()
        raise RuntimeError(
            f"Failed to open video source: {args.source}. "
            "If this is a webcam, check that it is connected and visible with "
            "`ls /dev/video*`, or try another source such as --source 1."
        )

    print("Real data collection started with YOLO pose.")
    print(f"Press number keys to save the current {args.window}-frame window:")
    for key, label in key_map.items():
        print(f"  {key} -> {label}")
    print(
        f"Press 'q' to quit after every class has at least "
        f"{args.min_samples_per_class} samples."
    )

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Frame read failed. Exiting.")
                break

            results = process_bgr_frame(frame, pose)
            valid_landmark_count = count_visible_landmarks(results)
            if valid_landmark_count >= args.min_valid_landmarks:
                sequence_buffer.append(extract_landmarks(results))
            else:
                sequence_buffer.clear()

            counts = count_by_class(y_data, args.classes)
            now = time.monotonic()
            visible_warning = warning_message if warning_until > now else None

            draw_pose_landmarks(frame, results, mp_pose, mp_drawing, mp_drawing_styles)
            draw_collection_overlay(
                frame=frame,
                key_map=key_map,
                counts=counts,
                active_label=active_label,
                window_fill=len(sequence_buffer),
                window_size=args.window,
                valid_landmark_count=valid_landmark_count,
                min_samples=args.min_samples_per_class,
                warning=visible_warning,
            )
            cv2.imshow("Collect Real Behavior Data (Q to quit)", frame)

            key_code = cv2.waitKey(1) & 0xFF
            key_char = chr(key_code) if key_code != 255 else ""

            if key_char.lower() == "q":
                counts = count_by_class(y_data, args.classes)
                if minimums_met(counts, args.min_samples_per_class):
                    break
                warning_message = "Collect more before quitting: " + missing_minimum_text(
                    counts, args.min_samples_per_class
                )
                warning_until = time.monotonic() + 3.0
                print(warning_message)
                continue

            if key_char in key_map:
                active_label = key_map[key_char]
                if len(sequence_buffer) < args.window:
                    warning_message = (
                        f"Need {args.window} valid frames before saving "
                        f"{active_label}: {len(sequence_buffer)}/{args.window}"
                    )
                    warning_until = time.monotonic() + 2.0
                    print(warning_message)
                    continue

                new_sequence = np.array(sequence_buffer, dtype=np.float32)[None, ...]
                new_label = np.array([active_label], dtype=str)
                x_data = np.concatenate([x_data, new_sequence], axis=0)
                y_data = np.concatenate([y_data, new_label], axis=0)
                save_dataset(x_data, y_data, args.x_path, args.y_path)

                counts = count_by_class(y_data, args.classes)
                warning_message = f"Saved {active_label}: {counts[active_label]} total"
                warning_until = time.monotonic() + 1.25
                print(f"Saved sample #{len(y_data)} as '{active_label}'.")

    finally:
        save_dataset(x_data, y_data, args.x_path, args.y_path)
        cap.release()
        pose.close()
        cv2.destroyAllWindows()

    final_counts = count_by_class(y_data, args.classes)
    print_count_table(final_counts, args.min_samples_per_class)
    print(f"\nFinished. Dataset shapes: X={x_data.shape}, y={y_data.shape}")
    print(f"Saved to: {args.x_path} and {args.y_path}")


if __name__ == "__main__":
    main()
