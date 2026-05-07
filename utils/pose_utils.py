"""Shared pose-estimation utilities for behavior detection scripts."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


NUM_LANDMARKS = 33
LANDMARK_DIMS = 3
LANDMARK_VECTOR_SIZE = NUM_LANDMARKS * LANDMARK_DIMS
SCRIPT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TASK_MODEL_PATH = SCRIPT_DIR / "models" / "pose_landmarker_lite.task"
DEFAULT_YOLO_POSE_MODEL = "yolo11n-pose.pt"

POSE_CONNECTIONS = frozenset(
    {
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),
        (9, 10),
        (11, 12),
        (11, 13),
        (13, 15),
        (15, 17),
        (15, 19),
        (15, 21),
        (17, 19),
        (12, 14),
        (14, 16),
        (16, 18),
        (16, 20),
        (16, 22),
        (18, 20),
        (11, 23),
        (12, 24),
        (23, 24),
        (23, 25),
        (24, 26),
        (25, 27),
        (26, 28),
        (27, 29),
        (28, 30),
        (29, 31),
        (30, 32),
        (27, 31),
        (28, 32),
    }
)

COCO_TO_BLAZEPOSE = {
    0: 0,  # nose
    1: 2,  # left eye
    2: 5,  # right eye
    3: 7,  # left ear
    4: 8,  # right ear
    5: 11,  # left shoulder
    6: 12,  # right shoulder
    7: 13,  # left elbow
    8: 14,  # right elbow
    9: 15,  # left wrist
    10: 16,  # right wrist
    11: 23,  # left hip
    12: 24,  # right hip
    13: 25,  # left knee
    14: 26,  # right knee
    15: 27,  # left ankle
    16: 28,  # right ankle
}


class _TaskPoseResults:
    """Adapter that matches the small part of mp.solutions results we use."""

    def __init__(self, landmarks):
        self.pose_landmarks = SimpleNamespace(landmark=landmarks) if landmarks else None


class _TaskPoseEstimator:
    """MediaPipe Tasks pose estimator with a .process(frame) method."""

    def __init__(self, mp_module, landmarker):
        self._mp = mp_module
        self._landmarker = landmarker

    def process(self, rgb_frame: np.ndarray) -> _TaskPoseResults:
        rgb_frame = np.ascontiguousarray(rgb_frame)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(image)
        landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
        return _TaskPoseResults(landmarks)

    def close(self) -> None:
        self._landmarker.close()


class _YoloPoseEstimator:
    """Ultralytics YOLO pose estimator with a .process(frame) method."""

    def __init__(
        self,
        model_path: str | Path,
        min_detection_confidence: float,
        device: str,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required for the YOLO pose backend. "
                "Install behavior_detection/requirements.txt, then run again."
            ) from exc

        self._model = YOLO(str(model_path))
        self._conf = min_detection_confidence
        self._device = device

    def process(self, rgb_frame: np.ndarray) -> _TaskPoseResults:
        results = self._model.predict(
            rgb_frame,
            conf=self._conf,
            device=self._device,
            verbose=False,
        )
        if not results:
            return _TaskPoseResults(None)

        result = results[0]
        if result.keypoints is None or result.keypoints.xyn is None:
            return _TaskPoseResults(None)

        keypoints = result.keypoints.xyn
        if len(keypoints) == 0:
            return _TaskPoseResults(None)

        person_idx = 0
        if result.boxes is not None and result.boxes.conf is not None:
            box_conf = result.boxes.conf.detach().cpu().numpy()
            if len(box_conf) > 0:
                person_idx = int(np.argmax(box_conf))

        xyn = keypoints[person_idx].detach().cpu().numpy()
        if result.keypoints.conf is not None:
            conf = result.keypoints.conf[person_idx].detach().cpu().numpy()
        else:
            conf = np.ones(len(xyn), dtype=np.float32)

        landmarks = [
            SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=0.0)
            for _ in range(NUM_LANDMARKS)
        ]
        for coco_idx, blaze_idx in COCO_TO_BLAZEPOSE.items():
            if coco_idx >= len(xyn):
                continue
            score = float(conf[coco_idx]) if coco_idx < len(conf) else 1.0
            if score <= 0.0:
                continue
            x, y = xyn[coco_idx]
            landmarks[blaze_idx] = SimpleNamespace(
                x=float(x),
                y=float(y),
                z=0.0,
                visibility=score,
            )

        if len(landmarks) != NUM_LANDMARKS:
            raise RuntimeError(
                f"YOLO pose mapping produced {len(landmarks)} landmarks; "
                f"expected {NUM_LANDMARKS}."
            )
        return _TaskPoseResults(landmarks)

    def close(self) -> None:
        return None


def create_pose_estimator(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_complexity: int = 1,
    pose_task_model_path: Path | str | None = None,
    pose_backend: str = "auto",
    yolo_pose_model: Path | str = DEFAULT_YOLO_POSE_MODEL,
    yolo_device: str = "cpu",
):
    """
    Initialize pose estimation and return commonly used drawing handles.

    Backends:
      - auto: classic MediaPipe when mp.solutions is available, otherwise YOLO.
      - mediapipe: classic mp.solutions.pose only.
      - mediapipe-tasks: newer MediaPipe Tasks API.
      - yolo: Ultralytics YOLO pose.

    Returns:
        Tuple of (mp_pose, mp_drawing, mp_drawing_styles, pose_estimator).
    """
    if not 0.0 <= min_detection_confidence <= 1.0:
        raise ValueError("min_detection_confidence must be between 0 and 1.")
    if not 0.0 <= min_tracking_confidence <= 1.0:
        raise ValueError("min_tracking_confidence must be between 0 and 1.")
    if model_complexity not in {0, 1, 2}:
        raise ValueError("model_complexity must be 0, 1, or 2.")

    backend = pose_backend.lower()
    allowed_backends = {"auto", "mediapipe", "mediapipe-tasks", "yolo"}
    if backend not in allowed_backends:
        raise ValueError(f"Unknown pose backend '{pose_backend}'. Use one of {allowed_backends}.")

    if backend == "auto" and sys.version_info >= (3, 13):
        mp_pose = SimpleNamespace(POSE_CONNECTIONS=POSE_CONNECTIONS)
        pose = _YoloPoseEstimator(
            model_path=yolo_pose_model,
            min_detection_confidence=min_detection_confidence,
            device=yolo_device,
        )
        return mp_pose, None, None, pose

    mp = None
    if backend in {"auto", "mediapipe", "mediapipe-tasks"}:
        try:
            import mediapipe as mp
        except ImportError as exc:
            if backend == "mediapipe":
                raise RuntimeError(
                    "MediaPipe is required for the classic MediaPipe backend. "
                    "Install behavior_detection/requirements.txt, then run again."
                ) from exc

    if backend == "auto" and (mp is None or not hasattr(mp, "solutions")):
        mp_pose = SimpleNamespace(POSE_CONNECTIONS=POSE_CONNECTIONS)
        pose = _YoloPoseEstimator(
            model_path=yolo_pose_model,
            min_detection_confidence=min_detection_confidence,
            device=yolo_device,
        )
        return mp_pose, None, None, pose

    if backend == "yolo":
        mp_pose = SimpleNamespace(POSE_CONNECTIONS=POSE_CONNECTIONS)
        pose = _YoloPoseEstimator(
            model_path=yolo_pose_model,
            min_detection_confidence=min_detection_confidence,
            device=yolo_device,
        )
        return mp_pose, None, None, pose

    if mp is not None and hasattr(mp, "solutions"):
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        return mp_pose, mp_drawing, mp_drawing_styles, pose

    if backend == "mediapipe":
        version = getattr(mp, "__version__", "unknown") if mp is not None else "missing"
        raise RuntimeError(
            "The classic MediaPipe backend needs mp.solutions.pose, but the "
            f"installed mediapipe package ({version}) does not provide it. "
            "Use '--pose-backend yolo' on Python 3.13, or use Python 3.10/3.11."
        )

    model_path = Path(pose_task_model_path or DEFAULT_TASK_MODEL_PATH)
    if not model_path.exists():
        version = getattr(mp, "__version__", "unknown")
        raise RuntimeError(
            "The installed MediaPipe package "
            f"({version}) uses the newer Tasks API and needs a pose landmarker "
            f"model file. Download it to: {model_path}\n\n"
            "Run:\n"
            "  mkdir -p behavior_detection/models\n"
            "  curl -L -o behavior_detection/models/pose_landmarker_lite.task "
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task\n\n"
            "Then run inference again."
        )

    from mediapipe.tasks import python as mp_tasks_python
    from mediapipe.tasks.python import vision as mp_tasks_vision

    options = mp_tasks_vision.PoseLandmarkerOptions(
        base_options=mp_tasks_python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp_tasks_python.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp_tasks_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )
    landmarker = mp_tasks_vision.PoseLandmarker.create_from_options(options)
    mp_pose = SimpleNamespace(POSE_CONNECTIONS=POSE_CONNECTIONS)
    return mp_pose, None, None, _TaskPoseEstimator(mp, landmarker)


def process_bgr_frame(frame: np.ndarray, pose_estimator):
    """Convert BGR frame to RGB and run pose estimation."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = pose_estimator.process(rgb_frame)
    rgb_frame.flags.writeable = True
    return results


def _landmark_score(landmark, field_name: str, default: float = 1.0) -> float:
    """
    Return optional landmark confidence fields without treating unset protobuf
    fields as real zeros.
    """
    has_field = getattr(landmark, "HasField", None)
    if callable(has_field):
        try:
            if not has_field(field_name):
                return default
        except (TypeError, ValueError):
            pass
    return float(getattr(landmark, field_name, default))


def _landmark_is_visible(landmark) -> bool:
    visibility = _landmark_score(landmark, "visibility")
    presence = _landmark_score(landmark, "presence")
    return (
        visibility > 0.1
        and presence > 0.1
        and 0.0 <= landmark.x <= 1.0
        and 0.0 <= landmark.y <= 1.0
    )


def count_visible_landmarks(results) -> int:
    """Count detected pose landmarks that are usable for behavior inference."""
    if results.pose_landmarks is None:
        return 0

    return sum(
        1
        for landmark in results.pose_landmarks.landmark[:NUM_LANDMARKS]
        if _landmark_is_visible(landmark)
    )


def extract_landmarks(results) -> np.ndarray:
    """
    Extract 33 (x, y, z) pose landmarks into a flat (99,) vector.

    Unfilled YOLO-to-BlazePose slots are replaced with the mean of visible
    landmarks from that frame so the classifier does not learn raw zero
    placeholders. If no pose is detected, returns zeros to preserve shape.
    """
    landmarks = np.zeros((NUM_LANDMARKS, LANDMARK_DIMS), dtype=np.float32)

    if results.pose_landmarks is None:
        return landmarks.reshape(LANDMARK_VECTOR_SIZE)

    pose_landmarks = list(results.pose_landmarks.landmark[:NUM_LANDMARKS])
    unfilled_slots = np.zeros(NUM_LANDMARKS, dtype=bool)
    if len(pose_landmarks) < NUM_LANDMARKS:
        unfilled_slots[len(pose_landmarks) :] = True
    visible_landmarks = []
    for idx, landmark in enumerate(pose_landmarks):
        is_unfilled = (
            float(getattr(landmark, "visibility", 0.0)) == 0.0
            and float(landmark.x) == 0.0
            and float(landmark.y) == 0.0
        )
        if is_unfilled:
            unfilled_slots[idx] = True
            continue

        coords = np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)
        landmarks[idx] = coords
        if _landmark_is_visible(landmark):
            visible_landmarks.append(coords)

    if visible_landmarks:
        fill_value = np.mean(np.stack(visible_landmarks, axis=0), axis=0)
        landmarks[unfilled_slots] = fill_value

    return landmarks.reshape(LANDMARK_VECTOR_SIZE)


def draw_pose_landmarks(
    frame: np.ndarray,
    results,
    mp_pose,
    mp_drawing,
    mp_drawing_styles=None,
) -> np.ndarray:
    """Draw pose landmarks and skeleton onto the input frame."""
    if results.pose_landmarks is None:
        return frame

    if mp_drawing is None:
        landmarks = results.pose_landmarks.landmark
        height, width = frame.shape[:2]

        for start_idx, end_idx in mp_pose.POSE_CONNECTIONS:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            if not _landmark_is_visible(start) or not _landmark_is_visible(end):
                continue
            start_xy = (int(start.x * width), int(start.y * height))
            end_xy = (int(end.x * width), int(end.y * height))
            cv2.line(frame, start_xy, end_xy, (0, 255, 0), 2)

        for landmark in landmarks[:NUM_LANDMARKS]:
            if not _landmark_is_visible(landmark):
                continue
            center = (int(landmark.x * width), int(landmark.y * height))
            cv2.circle(frame, center, 3, (0, 255, 255), -1)

    elif mp_drawing_styles is not None:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
    else:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
        )
    return frame
