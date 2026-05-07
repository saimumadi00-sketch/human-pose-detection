"""Train an LSTM classifier on pose landmark sequences."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Dense, Dropout, Input, LSTM

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_CLASSES = [
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
EXPECTED_WINDOW = 30
EXPECTED_FEATURE_DIM = 99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM model for behavior detection.")
    parser.add_argument(
        "--x-path",
        type=Path,
        default=SCRIPT_DIR / "data" / "X.npy",
        help="Path to X.npy",
    )
    parser.add_argument(
        "--y-path",
        type=Path,
        default=SCRIPT_DIR / "data" / "y.npy",
        help="Path to y.npy",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=SCRIPT_DIR / "models" / "pose_lstm.keras",
        help="Output path for trained Keras model.",
    )
    parser.add_argument(
        "--encoder-path",
        type=Path,
        default=SCRIPT_DIR / "models" / "label_encoder.pkl",
        help="Output path for class-label pickle.",
    )
    parser.add_argument(
        "--curves-path",
        type=Path,
        default=SCRIPT_DIR / "models" / "training_curves.png",
        help="Output path for training curves image.",
    )
    parser.add_argument(
        "--cm-path",
        type=Path,
        default=SCRIPT_DIR / "models" / "confusion_matrix.png",
        help="Output path for confusion matrix image.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="Restore best validation weights after this many non-improving epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--reduce-lr-patience",
        type=int,
        default=7,
        help="Reduce learning rate after this many non-improving validation-loss epochs.",
    )
    parser.add_argument(
        "--reduce-lr-factor",
        type=float,
        default=0.5,
        help="Learning-rate reduction factor used by ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=1e-6,
        help="Minimum learning rate used by ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Validation split fraction from training data.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def build_model(window: int, feature_dim: int, num_classes: int) -> tf.keras.Model:
    """Create the LSTM architecture requested for sequence classification."""
    model = Sequential(
        [
            Input(shape=(window, feature_dim)),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.4),
            Bidirectional(LSTM(64)),
            Dropout(0.4),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def normalize_behavior_labels(y: np.ndarray) -> np.ndarray:
    """Merge legacy seven-class labels into the current five-class taxonomy."""
    return np.array([LEGACY_LABEL_MAP.get(str(label), str(label)) for label in y], dtype=str)


def validate_dataset(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    validation_split: float,
) -> None:
    """Validate dataset shape and split settings before training starts."""
    if x.ndim != 3:
        raise ValueError(f"Expected X with shape (N, window, features), got {x.shape}")
    if x.shape[1:] != (EXPECTED_WINDOW, EXPECTED_FEATURE_DIM):
        raise ValueError(
            f"Expected X shape (N, {EXPECTED_WINDOW}, {EXPECTED_FEATURE_DIM}), got {x.shape}."
        )
    if len(x) != len(y):
        raise ValueError(f"Mismatched sample counts: X={len(x)}, y={len(y)}")
    if len(x) < 10:
        raise ValueError("Dataset too small. Collect more sequences before training.")
    if not np.isfinite(x).all():
        raise ValueError("Dataset X contains NaN or infinite values.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")
    if not 0.0 <= validation_split < 1.0:
        raise ValueError("--validation-split must be between 0 and 1.")


def validate_class_distribution(
    y_encoded: np.ndarray,
    class_names: np.ndarray,
    test_size: float,
) -> None:
    """Validate stratified split requirements and produce helpful errors."""
    counts = np.bincount(y_encoded, minlength=len(class_names))
    too_small = [class_names[idx] for idx, count in enumerate(counts) if count < 2]
    if too_small:
        labels = ", ".join(str(label) for label in too_small)
        raise ValueError(f"Each class needs at least 2 samples for stratified split: {labels}")

    test_count = int(np.ceil(len(y_encoded) * test_size))
    if test_count < len(class_names):
        raise ValueError(
            f"--test-size creates only {test_count} test samples for {len(class_names)} classes. "
            "Increase --test-size or collect more data."
        )

    train_count = len(y_encoded) - test_count
    if train_count < len(class_names):
        raise ValueError(
            f"--test-size leaves only {train_count} training samples for {len(class_names)} classes. "
            "Decrease --test-size or collect more data."
        )


def validate_training_args(args: argparse.Namespace) -> None:
    """Validate scalar training options before any expensive model work."""
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be >= 0.")
    if args.reduce_lr_patience < 0:
        raise ValueError("--reduce-lr-patience must be >= 0.")
    if not 0.0 < args.reduce_lr_factor < 1.0:
        raise ValueError("--reduce-lr-factor must be between 0 and 1.")
    if args.min_learning_rate <= 0.0:
        raise ValueError("--min-learning-rate must be > 0.")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be > 0.")


def validate_behavior_classes(y: np.ndarray) -> None:
    """Require the merged five-class taxonomy expected by inference."""
    class_set = set(y.tolist())
    expected = set(FINAL_CLASSES)
    unknown = sorted(class_set - expected)
    missing = [label for label in FINAL_CLASSES if label not in class_set]
    if unknown:
        raise ValueError(
            "Dataset contains unsupported labels after merging: "
            f"{', '.join(unknown)}. Expected only: {', '.join(FINAL_CLASSES)}."
        )
    if missing:
        raise ValueError(
            "Dataset is missing required classes after merging: "
            f"{', '.join(missing)}. Collect real samples for all five classes."
        )


def plot_training_curves(history: tf.keras.callbacks.History, output_path: Path) -> None:
    """Plot and save accuracy/loss training curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    if "val_accuracy" in history.history:
        axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: np.ndarray, output_path: Path
) -> None:
    """Compute and save confusion matrix visualization."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_training_args(args)

    # Ensure deterministic behavior where possible.
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Load sequence tensors and labels.
    if not args.x_path.exists() or not args.y_path.exists():
        raise FileNotFoundError(f"Missing dataset files: {args.x_path} and/or {args.y_path}")

    x = np.load(args.x_path).astype(np.float32)
    y = np.load(args.y_path, allow_pickle=True).astype(str)
    y = normalize_behavior_labels(y)

    validate_dataset(x, y, args.test_size, args.validation_split)
    validate_behavior_classes(y)

    # Encode labels to integers and then one-hot vectors.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    validate_class_distribution(y_encoded, label_encoder.classes_, args.test_size)
    num_classes = len(label_encoder.classes_)

    # Split data with stratification so each class is represented consistently.
    x_train, x_test, y_train_ids, y_test_ids = train_test_split(
        x,
        y_encoded,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_encoded,
    )
    y_train = tf.keras.utils.to_categorical(y_train_ids, num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(y_test_ids, num_classes=num_classes)
    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_train_ids,
    )
    class_weights = {
        class_idx: float(weight) for class_idx, weight in enumerate(class_weight_values)
    }

    window = x.shape[1]
    feature_dim = x.shape[2]
    model = build_model(window, feature_dim, num_classes)

    # Compile model using requested optimizer/loss/metric settings.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Train with a validation split from the training set.
    callbacks = []
    if args.early_stopping_patience > 0:
        if args.validation_split <= 0.0:
            raise ValueError("--early-stopping-patience requires --validation-split > 0.")
        callbacks.append(
            EarlyStopping(
                monitor="val_accuracy",
                patience=args.early_stopping_patience,
                restore_best_weights=True,
            )
        )
    if args.reduce_lr_patience > 0:
        if args.validation_split <= 0.0:
            raise ValueError("--reduce-lr-patience requires --validation-split > 0.")
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=args.reduce_lr_factor,
                patience=args.reduce_lr_patience,
                min_lr=args.min_learning_rate,
            )
        )

    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Evaluate on held-out test data.
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    y_pred_ids = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print("\nClassification Report:")
    print(
        classification_report(
            y_test_ids,
            y_pred_ids,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    # Save model and class-label artifacts for inference.
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.encoder_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    encoder_path = args.encoder_path
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)

    # Save visual diagnostics.
    plot_training_curves(history, args.curves_path)
    plot_confusion_matrix(
        y_true=y_test_ids,
        y_pred=y_pred_ids,
        class_names=label_encoder.classes_,
        output_path=args.cm_path,
    )

    print(f"Saved model: {args.model_path}")
    print(f"Saved class labels: {args.encoder_path}")
    print(f"Saved training curves: {args.curves_path}")
    print(f"Saved confusion matrix: {args.cm_path}")


if __name__ == "__main__":
    main()
