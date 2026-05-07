"""Train an Ultralytics YOLO pose model on the COCO-Pose dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA = SCRIPT_DIR / "datasets" / "coco-pose.yaml"
DEFAULT_DATASETS_DIR = PROJECT_ROOT / "datasets"
DEFAULT_PROJECT = SCRIPT_DIR / "runs" / "pose"
DEFAULT_YOLO_POSE_MODEL = "yolo11n-pose.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Ultralytics YOLO pose on the COCO-Pose dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Dataset YAML. Defaults to this project's COCO-Pose YAML.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_YOLO_POSE_MODEL,
        help="Pose model checkpoint or YAML architecture to train.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers.")
    parser.add_argument(
        "--device",
        default=None,
        help="Training device, for example '0', '0,1', 'cpu', or 'mps'.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_PROJECT,
        help="Directory where Ultralytics training runs are saved.",
    )
    parser.add_argument(
        "--name",
        default="coco-pose-yolo11n",
        help="Run name inside the project directory.",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=DEFAULT_DATASETS_DIR,
        help="Ultralytics datasets_dir for downloads and dataset lookup.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the model checkpoint when supported.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow reusing an existing run directory.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Epochs to wait for improvement before early stopping.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate YOLO training options before handing them to Ultralytics."""
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0.")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be > 0.")
    if args.batch_size == 0:
        raise ValueError("--batch-size cannot be 0.")
    if args.workers < 0:
        raise ValueError("--workers must be >= 0.")
    if args.patience < 0:
        raise ValueError("--patience must be >= 0.")
    if not args.name.strip():
        raise ValueError("--name cannot be empty.")


def load_ultralytics():
    try:
        from ultralytics import YOLO, settings
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is required for COCO-Pose training. "
            "Install it with: pip install -r behavior_detection/requirements.txt"
        ) from exc
    return YOLO, settings


def configure_datasets_dir(settings: Any, datasets_dir: Path) -> None:
    """Point Ultralytics at this project's dataset cache directory."""
    datasets_dir.mkdir(parents=True, exist_ok=True)
    settings.update({"datasets_dir": str(datasets_dir.resolve())})


def resolve_data_arg(data: Path) -> str:
    """Resolve local YAML paths while still allowing Ultralytics built-in names."""
    if data.exists():
        return str(data.resolve())

    if data.name == str(data) and data.suffix in {".yaml", ".yml"}:
        return str(data)

    raise FileNotFoundError(f"Dataset YAML not found: {data}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    YOLO, settings = load_ultralytics()
    configure_datasets_dir(settings, args.datasets_dir)
    data = resolve_data_arg(args.data)

    train_kwargs: dict[str, Any] = {
        "data": data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch_size,
        "workers": args.workers,
        "project": str(args.project),
        "name": args.name,
        "resume": args.resume,
        "exist_ok": args.exist_ok,
        "patience": args.patience,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    print(f"Training {args.model} with COCO-Pose config: {args.data}")
    print(f"Ultralytics datasets_dir: {args.datasets_dir.resolve()}")

    model = YOLO(args.model)
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
