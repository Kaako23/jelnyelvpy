"""MediaPipe Tasks API: FaceLandmarker, HandLandmarker, PoseLandmarker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing

from jelnyelv.config import FACE_DIMS, HAND_DIMS, POSE_DIMS


@dataclass
class MpTasksModels:
    pose: Path
    face: Path
    hand: Path


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "mediapipe"


def _ensure_models_exist(models: MpTasksModels) -> None:
    missing = [str(p) for p in [models.pose, models.face, models.hand] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing MediaPipe Tasks model files:\n"
            + "\n".join(missing)
            + "\n\nRun: python scripts/download_mediapipe_models.py"
        )


class HolisticTasks:
    """Replacement for mp.solutions.holistic using MediaPipe Tasks."""

    def __init__(self, models: MpTasksModels | None = None) -> None:
        if models is None:
            d = _default_model_dir()
            models = MpTasksModels(
                pose=d / "pose_landmarker.task",
                face=d / "face_landmarker.task",
                hand=d / "hand_landmarker.task",
            )

        _ensure_models_exist(models)

        base = mp_python.BaseOptions

        self.pose = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=base(model_asset_path=str(models.pose)),
                running_mode=mp_vision.RunningMode.IMAGE,
                output_segmentation_masks=False,
            )
        )

        self.face = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=base(model_asset_path=str(models.face)),
                running_mode=mp_vision.RunningMode.IMAGE,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
            )
        )

        self.hand = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=base(model_asset_path=str(models.hand)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
            )
        )

    def close(self) -> None:
        self.pose.close()
        self.face.close()
        self.hand.close()

    def __enter__(self) -> HolisticTasks:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def mediapipe_detection(bgr_image: np.ndarray, tasks: HolisticTasks) -> tuple[np.ndarray, dict]:
    """Return (bgr_image, results_dict)."""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    pose_res = tasks.pose.detect(mp_image)
    face_res = tasks.face.detect(mp_image)
    hand_res = tasks.hand.detect(mp_image)

    return bgr_image, {"pose": pose_res, "face": face_res, "hands": hand_res}


def extract_keypoints(results: dict) -> np.ndarray:
    """Pack pose+face+hands into a (1662,) float32 vector."""
    pose_vec = np.zeros(POSE_DIMS, dtype=np.float32)
    if results["pose"].pose_landmarks:
        pts = results["pose"].pose_landmarks[0]
        arr = []
        for lm in pts:
            arr.extend([lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)])
        pose_vec[: min(len(arr), POSE_DIMS)] = np.array(arr[:POSE_DIMS], dtype=np.float32)

    face_vec = np.zeros(FACE_DIMS, dtype=np.float32)
    if results["face"].face_landmarks:
        pts = results["face"].face_landmarks[0]
        arr = []
        for lm in pts:
            arr.extend([lm.x, lm.y, lm.z])
        arr = np.array(arr, dtype=np.float32)
        face_vec[: min(len(arr), FACE_DIMS)] = arr[:FACE_DIMS]

    lh_vec = np.zeros(HAND_DIMS, dtype=np.float32)
    rh_vec = np.zeros(HAND_DIMS, dtype=np.float32)

    hands = results["hands"]
    if hands.hand_landmarks and hands.handedness:
        for landmarks, handed in zip(hands.hand_landmarks, hands.handedness):
            label = handed[0].category_name.lower()
            arr = []
            for lm in landmarks:
                arr.extend([lm.x, lm.y, lm.z])
            vec = np.array(arr, dtype=np.float32)
            if "left" in label:
                lh_vec[:] = vec
            elif "right" in label:
                rh_vec[:] = vec

    return np.concatenate([pose_vec, face_vec, lh_vec, rh_vec]).astype(np.float32)


def draw_landmarks_on_image(bgr_image: np.ndarray, results: dict) -> np.ndarray:
    """Draw pose, face, and hand landmarks on image. Returns annotated BGR image."""
    annotated = bgr_image.copy()
    pose_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
    face_spec = mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=1)
    left_hand_spec = mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2, circle_radius=2)
    right_hand_spec = mp_drawing.DrawingSpec(color=(0, 165, 255), thickness=2, circle_radius=2)

    if results["pose"].pose_landmarks:
        for pose_landmarks in results["pose"].pose_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated,
                landmark_list=pose_landmarks,
                connections=mp_vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=pose_spec,
                connection_drawing_spec=pose_spec,
            )

    if results["face"].face_landmarks:
        for face_landmarks in results["face"].face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks,
                connections=mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=face_spec,
                connection_drawing_spec=face_spec,
            )

    hands = results["hands"]
    if hands.hand_landmarks and hands.handedness:
        for landmarks, handed in zip(hands.hand_landmarks, hands.handedness):
            label = handed[0].category_name.lower()
            spec = left_hand_spec if "left" in label else right_hand_spec
            mp_drawing.draw_landmarks(
                image=annotated,
                landmark_list=landmarks,
                connections=mp_vision.HandLandmarksConnections.HAND_CONNECTIONS,
                landmark_drawing_spec=spec,
                connection_drawing_spec=spec,
            )

    return annotated


