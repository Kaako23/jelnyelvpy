#!/usr/bin/env python3
"""Download MediaPipe Tasks model files to assets/mediapipe/."""

import os
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE / "assets" / "mediapipe"

MODELS = {
    "pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        path = MODEL_DIR / name
        if path.exists():
            print(f"Skip (exists): {path}")
            continue
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, path)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
