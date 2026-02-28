"""Configuration constants for sign language recognition."""

import os

# Paths relative to project root (repo root, not package)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "jelek")
MODEL_PATH = os.path.join(_PROJECT_ROOT, "your_model.pth")
EVALUATION_REPORT_PATH = os.path.join(_PROJECT_ROOT, "ertekelesi_jelentes.txt")

# MediaPipe holistic: pose(33*4) + face(468*3) + left_hand(21*3) + right_hand(21*3)
POSE_DIMS = 33 * 4
FACE_DIMS = 468 * 3
HAND_DIMS = 21 * 3
INPUT_SIZE = POSE_DIMS + FACE_DIMS + HAND_DIMS + HAND_DIMS
HIDDEN_SIZE = 64
NO_SEQUENCES = 30
SEQUENCE_LENGTH = 31
PAUSE_BETWEEN_SEQUENCES_SEC = 1.0
BATCH_SIZE = 32
EPOCHS = 30
GRAD_CLIP_MAX_NORM = 1.0

# Recognition (inference)
MIN_RECOGNITION_FRAMES = 10  # Start inference after this many frames (~0.33 sec at 30fps)
CONFIDENCE_THRESHOLD = 0.5  # Only switch label when smoothed confidence >= this
PREDICTION_SMOOTHING_WINDOW = 5  # Rolling average of last N probability vectors
PREDICTION_HOLD_FRAMES = 5  # Keep previous label for N frames when conf drops below threshold

# Resize frame to this width for recognition (smaller = faster MediaPipe, ~2–3x speedup)
RECOGNITION_INPUT_WIDTH = 256

# Compact evaluation report (scalable to 2000+ classes)
TOP_N_CONFUSED_PAIRS = 15
TOP_N_BEST_CLASSES = 5
TOP_N_WORST_CLASSES = 5
