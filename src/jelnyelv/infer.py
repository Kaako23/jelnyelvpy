"""Realtime inference: buffer, probability smoothing, hold logic."""

from collections import deque

import numpy as np
import torch

from jelnyelv.config import (
    CONFIDENCE_THRESHOLD,
    HIDDEN_SIZE,
    INPUT_SIZE,
    MIN_RECOGNITION_FRAMES,
    MODEL_PATH,
    PREDICTION_HOLD_FRAMES,
    PREDICTION_SMOOTHING_WINDOW,
    SEQUENCE_LENGTH,
)
from jelnyelv.model import LSTMModel


class Recognizer:
    """Maintains buffer, runs inference with probability smoothing and hold logic."""

    def __init__(self) -> None:
        self._model: LSTMModel | None = None
        self._reverse_label_map: dict[int, str] = {}
        self._buffer: deque = deque(maxlen=SEQUENCE_LENGTH)
        self._prob_history: deque = deque(maxlen=PREDICTION_SMOOTHING_WINDOW)
        self._displayed_label: str = "?"
        self._hold_counter: int = 0

    def load(self) -> str | None:
        """Load model. Returns error message or None."""
        import os

        if not os.path.exists(MODEL_PATH):
            return f"Model not found: {MODEL_PATH}. Train first."

        try:
            ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        except Exception as e:
            return f"Failed to load model: {e}"

        state_dict = self._extract_state_dict(ckpt)
        if state_dict is None:
            return (
                f"Checkpoint has no 'model_state_dict' or 'state_dict'. "
                f"Keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'raw state_dict'}. "
                f"Retrain the model."
            )

        labels = ckpt.get("labels") or ckpt.get("actions") if isinstance(ckpt, dict) else None
        if labels is None:
            return "Checkpoint has no labels. Retrain the model."

        label_map = ckpt.get("label_map")
        if label_map is None:
            label_map = {label: i for i, label in enumerate(labels)}

        output_size = len(labels)
        if "fc3.weight" in state_dict:
            model_output_size = int(state_dict["fc3.weight"].shape[0])
            if output_size != model_output_size:
                return (
                    f"Checkpoint labels ({output_size}) do not match model output size ({model_output_size}). "
                    f"Retrain the model."
                )
        self._model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, output_size)
        self._model.load_state_dict(state_dict, strict=False)
        self._model.eval()
        self._reverse_label_map = {i: str(lbl) for lbl, i in label_map.items()}
        self._buffer.clear()
        self._prob_history.clear()
        self._displayed_label = "?"
        self._hold_counter = 0
        return None

    def _extract_state_dict(self, ckpt):
        """Extract state_dict from checkpoint dict, or raw state_dict/OrderedDict."""
        if ckpt is None:
            return None
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                return ckpt["model_state_dict"]
            if "state_dict" in ckpt:
                return ckpt["state_dict"]
            if "model" in ckpt:
                m = ckpt["model"]
                if hasattr(m, "state_dict"):
                    return m.state_dict()
                if isinstance(m, dict) and any(k.startswith("lstm") or k.startswith("fc") for k in m.keys()):
                    return m
            if any(k.startswith("lstm") or k.startswith("fc") for k in ckpt.keys()):
                return ckpt
        if hasattr(ckpt, "keys") and hasattr(ckpt, "values"):
            return ckpt
        return None

    def add_frame(self, keypoints: np.ndarray) -> tuple[str, float]:
        """Add one frame of keypoints. Returns (label, confidence)."""
        if self._model is None:
            return "?", 0.0

        kp = np.asarray(keypoints, dtype=np.float32)
        if kp.ndim != 1 or kp.shape[0] != INPUT_SIZE:
            return self._displayed_label, 0.0

        self._buffer.append(kp)
        if len(self._buffer) < MIN_RECOGNITION_FRAMES:
            return self._displayed_label, 0.0

        # Pad to SEQUENCE_LENGTH if we have fewer frames (repeat last frame for reactive early inference)
        buf_list = list(self._buffer)
        if len(buf_list) < SEQUENCE_LENGTH:
            last = buf_list[-1]
            buf_list = buf_list + [last] * (SEQUENCE_LENGTH - len(buf_list))
        seq = np.stack(buf_list, axis=0)
        x = torch.tensor(seq[np.newaxis, :, :])
        with torch.no_grad():
            out = self._model(x)
            probs = torch.softmax(out, dim=1)
        probs_np = probs.numpy().squeeze()

        # Rolling average of probability vectors
        self._prob_history.append(probs_np)
        avg_probs = np.mean(list(self._prob_history), axis=0)
        pred_idx = int(np.argmax(avg_probs))
        conf_val = float(avg_probs[pred_idx])
        label = self._reverse_label_map.get(pred_idx, "?")

        # Stable output: hold previous label briefly when confidence drops
        if conf_val >= CONFIDENCE_THRESHOLD:
            self._displayed_label = label
            self._hold_counter = 0
        else:
            if self._displayed_label != "?" and self._hold_counter < PREDICTION_HOLD_FRAMES:
                self._hold_counter += 1
                # Keep showing previous label
            else:
                self._displayed_label = "?"
                self._hold_counter = 0

        return self._displayed_label, conf_val
