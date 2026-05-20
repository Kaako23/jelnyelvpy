import os

import cv2
import numpy as np

from jelnyelv.config import DATA_PATH, RECOGNITION_INPUT_WIDTH, SEQUENCE_LENGTH
from jelnyelv.dataset import get_words_for_record
from jelnyelv.mp_features import (
    HolisticTasks,
    mediapipe_detection,
    pack_landmarks,
)


def _normalize_video_path(video_path) -> str | None:
    if video_path is None:
        return None
    if isinstance(video_path, list) and video_path:
        video_path = video_path[0]
    if isinstance(video_path, dict) and "path" in video_path:
        video_path = video_path["path"]
    path = str(video_path).strip()
    if not path:
        return None
    if path.startswith("file="):
        path = path[5:]
    return path if os.path.isfile(path) else None


def get_video_duration(video_path) -> tuple[float, float, str | None]:
    path = _normalize_video_path(video_path)
    if not path:
        return 0.0, 0.0, "Először válassz videófájlt."
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0, 0.0, "Nem sikerült megnyitni a videót."
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_sec = total_frames / fps if total_frames > 0 else 0.0
    return duration_sec, fps, None


def sec_to_mmss(sec: float) -> str:
    m = int(sec) // 60
    s = int(sec) % 60
    return f"{m:02d}:{s:02d}"


def import_from_video(
    video_path: str | None,
    word: str,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> tuple[str, list[str] | None]:
    path = _normalize_video_path(video_path)
    if not path:
        return "Először válassz videófájlt.", None

    if not word or not str(word).strip():
        return "Adj meg vagy válassz szócímkét.", None

    word = str(word).strip()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return f"Nem sikerült megnyitni a videót: {path}", None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if total_frames > 0 else 0

    start_frame = 0
    end_frame = total_frames
    if start_sec is not None and start_sec > 0:
        start_frame = int(start_sec * fps)
    if end_sec is not None and end_sec > 0 and end_sec < duration_sec:
        end_frame = int(end_sec * fps)
    if start_frame >= end_frame:
        cap.release()
        return "Érvénytelen kezdő vagy záró időpont.", None

    word_path = os.path.join(DATA_PATH, word)
    os.makedirs(word_path, exist_ok=True)
    existing = [int(d) for d in os.listdir(word_path) if d.isdigit()]
    next_seq = max(existing, default=-1) + 1

    keypoints_buffer: list[np.ndarray] = []
    seq_count = 0

    try:
        with HolisticTasks() as tasks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame

            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                small_w = RECOGNITION_INPUT_WIDTH
                small_h = int(h * small_w / w)
                small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                _, results = mediapipe_detection(small, tasks)
                kp = pack_landmarks(results)
                keypoints_buffer.append(kp)
                frame_idx += 1

                while len(keypoints_buffer) >= SEQUENCE_LENGTH:
                    seq_path = os.path.join(word_path, str(next_seq))
                    os.makedirs(seq_path, exist_ok=True)
                    to_save = keypoints_buffer[:SEQUENCE_LENGTH]
                    for out_idx, kp_arr in enumerate(to_save):
                        np.save(
                            os.path.join(seq_path, f"{out_idx}.npy"),
                            np.asarray(kp_arr, dtype=np.float32),
                        )
                    keypoints_buffer = keypoints_buffer[SEQUENCE_LENGTH:]
                    next_seq += 1
                    seq_count += 1

            if keypoints_buffer and len(keypoints_buffer) >= 5:
                seq_path = os.path.join(word_path, str(next_seq))
                os.makedirs(seq_path, exist_ok=True)
                last = keypoints_buffer[-1]
                pad = [last] * (SEQUENCE_LENGTH - len(keypoints_buffer))
                to_save = keypoints_buffer + pad
                for out_idx, kp_arr in enumerate(to_save):
                    np.save(
                        os.path.join(seq_path, f"{out_idx}.npy"),
                        np.asarray(kp_arr, dtype=np.float32),
                    )
                seq_count += 1
    finally:
        cap.release()

    msg = f"{seq_count} szekvencia importálva a(z) „{word}” szóhoz a videóból."
    return msg, get_words_for_record()
