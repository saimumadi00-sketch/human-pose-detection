"""Generate synthetic pose sequence data compatible with the training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_CLASSES = [
    "desk_work",
    "on_phone",
    "idle_sitting",
    "consuming",
    "falling",
]
SUPPORTED_CLASSES = frozenset(DEFAULT_CLASSES)
SCRIPT_DIR = Path(__file__).resolve().parent
NUM_LANDMARKS = 33
LANDMARK_DIMS = 3
LANDMARK_VECTOR_SIZE = NUM_LANDMARKS * LANDMARK_DIMS

# Core BlazePose landmark ids used to create behavior-like motion.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic (X.npy, y.npy) pose sequence dataset."
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help=f"Class labels to synthesize. Supported: {', '.join(DEFAULT_CLASSES)}.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=120,
        help="Number of sequences to generate for each class.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Frames per sequence.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.008,
        help="Gaussian noise standard deviation added to each landmark coordinate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--x-path",
        type=Path,
        default=SCRIPT_DIR / "data" / "X.npy",
        help="Output path for X dataset.",
    )
    parser.add_argument(
        "--y-path",
        type=Path,
        default=SCRIPT_DIR / "data" / "y.npy",
        help="Output path for y labels.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing dataset instead of overwriting.",
    )
    return parser.parse_args()


def create_base_pose(rng: np.random.Generator) -> np.ndarray:
    """Create a plausible neutral 33x3 skeleton in normalized coordinate space."""
    pose = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)

    # Generic body cloud (for landmarks not explicitly set below).
    pose[:, 0] = 0.50 + rng.normal(0.0, 0.03, size=NUM_LANDMARKS)
    pose[:, 1] = 0.58 + rng.normal(0.0, 0.06, size=NUM_LANDMARKS)
    pose[:, 2] = rng.normal(0.0, 0.02, size=NUM_LANDMARKS)

    # Key joints arranged in an upright stance.
    key = {
        NOSE: (0.50, 0.18, -0.10),
        LEFT_SHOULDER: (0.44, 0.35, 0.02),
        RIGHT_SHOULDER: (0.56, 0.35, 0.02),
        LEFT_ELBOW: (0.39, 0.48, 0.03),
        RIGHT_ELBOW: (0.61, 0.48, 0.03),
        LEFT_WRIST: (0.37, 0.62, 0.05),
        RIGHT_WRIST: (0.63, 0.62, 0.05),
        LEFT_HIP: (0.46, 0.58, 0.03),
        RIGHT_HIP: (0.54, 0.58, 0.03),
        LEFT_KNEE: (0.46, 0.76, 0.04),
        RIGHT_KNEE: (0.54, 0.76, 0.04),
        LEFT_ANKLE: (0.46, 0.93, 0.06),
        RIGHT_ANKLE: (0.54, 0.93, 0.06),
    }
    for idx, coords in key.items():
        pose[idx] = np.array(coords, dtype=np.float32)

    return pose


def apply_global_variation(
    pose: np.ndarray,
    tx: float,
    ty: float,
    scale: float,
) -> np.ndarray:
    """Apply translation and scale around image center to simulate person variation."""
    out = pose.copy()
    out[:, 0] = 0.5 + (out[:, 0] - 0.5) * scale + tx
    out[:, 1] = 0.5 + (out[:, 1] - 0.5) * scale + ty
    return out


def _apply_seated_posture(pose: np.ndarray, phase: float, slouch: float = 0.0) -> None:
    """Fold the lower body into a seated pose used by desk-based classes."""
    pose[[LEFT_HIP, RIGHT_HIP], 1] += 0.08 + slouch
    pose[[LEFT_KNEE, RIGHT_KNEE], 1] -= 0.06
    pose[[LEFT_ANKLE, RIGHT_ANKLE], 1] -= 0.03
    pose[[LEFT_KNEE, RIGHT_KNEE], 0] += 0.025 * np.sin(phase)
    pose[[LEFT_ANKLE, RIGHT_ANKLE], 0] += 0.035 * np.sin(phase + np.pi / 2.0)
    pose[[NOSE, LEFT_SHOULDER, RIGHT_SHOULDER], 1] += slouch


def _move_joint_toward(pose: np.ndarray, joint: int, target_xy: tuple[float, float], amount: float) -> None:
    """Blend one landmark toward a normalized image-space target."""
    pose[joint, 0] += (target_xy[0] - pose[joint, 0]) * amount
    pose[joint, 1] += (target_xy[1] - pose[joint, 1]) * amount


def apply_behavior_motion(
    pose: np.ndarray,
    class_name: str,
    frame_idx: int,
    window: int,
) -> np.ndarray:
    """Inject class-specific motion pattern into one frame."""
    out = pose.copy()
    t = frame_idx / max(window - 1, 1)
    phase = 2.0 * np.pi * t

    # Minor baseline sway in all classes.
    out[:, 0] += 0.004 * np.sin(phase)
    out[:, 1] += 0.003 * np.sin(2.0 * phase)

    if class_name == "consuming":
        _apply_seated_posture(out, phase, slouch=0.010)
        bite = 0.5 - 0.5 * np.cos(3.0 * phase)
        _move_joint_toward(out, RIGHT_ELBOW, (0.58, 0.45), 0.55)
        _move_joint_toward(out, RIGHT_WRIST, (0.60, 0.56), 0.35)
        _move_joint_toward(out, RIGHT_WRIST, (0.52, 0.30), 0.60 * bite)
        out[NOSE, 1] += 0.015 * bite
        out[LEFT_WRIST, 1] += 0.025 * np.sin(2.0 * phase)

    elif class_name == "falling":
        # Smooth transition from upright to down/tilted pose.
        drop = 1.0 / (1.0 + np.exp(-11.0 * (t - 0.55)))
        out[:, 1] += 0.28 * drop
        out[:, 2] += 0.22 * drop
        out[[LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE], 0] -= 0.10 * drop
        out[[RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE], 0] += 0.04 * drop
        out[NOSE, 1] += 0.16 * drop

    elif class_name == "idle_sitting":
        _apply_seated_posture(out, phase, slouch=0.005)
        out[[LEFT_WRIST, RIGHT_WRIST], 1] += 0.025
        out[[LEFT_WRIST, RIGHT_WRIST], 0] += 0.008 * np.sin(phase)
        out[[LEFT_SHOULDER, RIGHT_SHOULDER], 1] += 0.006 * np.sin(phase)

    elif class_name == "on_phone":
        _apply_seated_posture(out, phase, slouch=0.006)
        _move_joint_toward(out, RIGHT_ELBOW, (0.60, 0.40), 0.70)
        _move_joint_toward(out, RIGHT_WRIST, (0.62, 0.28), 0.88)
        out[RIGHT_WRIST, 0] += 0.010 * np.sin(5.0 * phase)
        out[RIGHT_WRIST, 1] += 0.006 * np.cos(4.0 * phase)
        out[NOSE, 0] += 0.012
        out[LEFT_WRIST, 1] += 0.035

    elif class_name == "desk_work":
        _apply_seated_posture(out, phase, slouch=0.014)
        key_tap = np.sin(8.0 * phase)
        _move_joint_toward(out, LEFT_WRIST, (0.46, 0.57), 0.80)
        _move_joint_toward(out, RIGHT_WRIST, (0.54, 0.57), 0.80)
        _move_joint_toward(out, LEFT_ELBOW, (0.43, 0.50), 0.35)
        _move_joint_toward(out, RIGHT_ELBOW, (0.57, 0.50), 0.35)
        out[LEFT_WRIST, 1] += 0.014 * key_tap
        out[RIGHT_WRIST, 1] -= 0.014 * key_tap
        out[[LEFT_WRIST, RIGHT_WRIST], 0] += 0.010 * np.sin(6.0 * phase)
        out[NOSE, 1] += 0.020

    else:
        supported = ", ".join(DEFAULT_CLASSES)
        raise ValueError(f"Unsupported class_name '{class_name}'. Supported: {supported}.")

    return out


def generate_sequence(
    class_name: str,
    window: int,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one sequence with shape (window, 99)."""
    base_pose = create_base_pose(rng)
    tx = rng.uniform(-0.05, 0.05)
    ty = rng.uniform(-0.04, 0.04)
    scale = rng.uniform(0.92, 1.08)

    sequence = np.zeros((window, LANDMARK_VECTOR_SIZE), dtype=np.float32)
    for i in range(window):
        frame_pose = apply_global_variation(base_pose, tx=tx, ty=ty, scale=scale)
        frame_pose = apply_behavior_motion(frame_pose, class_name, i, window)
        frame_pose += rng.normal(0.0, noise_std, size=frame_pose.shape).astype(np.float32)

        # Keep coordinates bounded to normalized pose-like range.
        frame_pose[:, 0] = np.clip(frame_pose[:, 0], 0.0, 1.0)
        frame_pose[:, 1] = np.clip(frame_pose[:, 1], 0.0, 1.0)
        frame_pose[:, 2] = np.clip(frame_pose[:, 2], -1.0, 1.0)
        sequence[i] = frame_pose.reshape(-1)

    return sequence


def count_by_label(labels: Iterable[str]) -> dict[str, int]:
    """Count labels for quick generation summary."""
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


def main() -> None:
    args = parse_args()
    if args.samples_per_class <= 0:
        raise ValueError("--samples-per-class must be > 0.")
    if args.window <= 0:
        raise ValueError("--window must be > 0.")
    if args.noise_std < 0.0:
        raise ValueError("--noise-std must be >= 0.")
    if not args.classes:
        raise ValueError("--classes cannot be empty.")
    duplicate_classes = [
        label for label, count in count_by_label(args.classes).items() if count > 1
    ]
    if duplicate_classes:
        duplicates = ", ".join(duplicate_classes)
        raise ValueError(f"--classes cannot contain duplicates: {duplicates}.")
    unsupported_classes = sorted(set(args.classes) - SUPPORTED_CLASSES)
    if unsupported_classes:
        supported = ", ".join(DEFAULT_CLASSES)
        unsupported = ", ".join(unsupported_classes)
        raise ValueError(f"Unsupported synthetic classes: {unsupported}. Supported: {supported}.")

    args.x_path.parent.mkdir(parents=True, exist_ok=True)
    args.y_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    generated_sequences: list[np.ndarray] = []
    generated_labels: list[str] = []

    for class_name in args.classes:
        for _ in range(args.samples_per_class):
            generated_sequences.append(
                generate_sequence(
                    class_name=class_name,
                    window=args.window,
                    noise_std=args.noise_std,
                    rng=rng,
                )
            )
            generated_labels.append(class_name)

    x_new = np.stack(generated_sequences, axis=0).astype(np.float32)
    y_new = np.array(generated_labels, dtype=str)

    # Shuffle full synthetic set for randomized class ordering.
    indices = rng.permutation(len(y_new))
    x_new = x_new[indices]
    y_new = y_new[indices]

    if args.append and args.x_path.exists() and args.y_path.exists():
        x_old = np.load(args.x_path).astype(np.float32)
        y_old = np.load(args.y_path, allow_pickle=True).astype(str)
        if x_old.ndim != 3 or x_old.shape[1:] != (args.window, LANDMARK_VECTOR_SIZE):
            raise ValueError(
                f"Existing X shape mismatch: expected (N, {args.window}, {LANDMARK_VECTOR_SIZE}), "
                f"found {x_old.shape}."
            )
        if len(x_old) != len(y_old):
            raise ValueError(f"Existing dataset mismatch: X={len(x_old)}, y={len(y_old)}.")
        unsupported_old_classes = sorted(set(y_old.tolist()) - SUPPORTED_CLASSES)
        if unsupported_old_classes:
            unsupported = ", ".join(unsupported_old_classes)
            raise ValueError(
                f"Existing dataset contains unsupported labels: {unsupported}. "
                "Regenerate the dataset instead of appending."
            )
        x_data = np.concatenate([x_old, x_new], axis=0)
        y_data = np.concatenate([y_old, y_new], axis=0)
    else:
        x_data = x_new
        y_data = y_new

    np.save(args.x_path, x_data)
    np.save(args.y_path, y_data)

    print(f"Saved X: {args.x_path} | shape={x_data.shape}")
    print(f"Saved y: {args.y_path} | shape={y_data.shape}")
    print(f"Class counts: {count_by_label(y_data.tolist())}")


if __name__ == "__main__":
    main()
