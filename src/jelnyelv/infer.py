"""Realtime inference: buffer, thresholding, stability gate."""

from collections import deque

import numpy as np
import torch

from jelnyelv.config import (
    CONFIDENCE_THRESHOLD,
    HIDDEN_SIZE,
    INPUT_SIZE,
    MODEL_PATH,
    SEQUENCE_LENGTH,
    STABILITY_GATE_N,
)
from jelnyelv.model import LSTMModel


class Recognizer:
    """Maintains buffer and runs inference with stability gate."""

    def __init__(self) -> None:
        self._model: LSTMModel | None = None
        self._reverse_label_map: dict[int, str] = {}
        self._buffer: deque = deque(maxlen=SEQUENCE_LENGTH)
        self._stability_queue: deque = deque(maxlen=STABILITY_GATE_N)
        self._last_stable: str = "?"

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
        self._stability_queue.clear()
        self._last_stable = "?"
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
            return "?", 0.0  # Invalid shape; skip frame silently

        self._buffer.append(kp)
        if len(self._buffer) < SEQUENCE_LENGTH:
            return self._last_stable, 0.0

        seq = np.stack(list(self._buffer), axis=0)
        x = torch.tensor(seq[np.newaxis, :, :])
        with torch.no_grad():
            out = self._model(x)
            probs = torch.softmax(out, dim=1)
            conf, pred = torch.max(probs, 1)
        pred_idx = pred.item()
        conf_val = conf.item()
        label = self._reverse_label_map.get(pred_idx, "?")
        self._stability_queue.append((label, conf_val))

        if len(self._stability_queue) < STABILITY_GATE_N:
            return self._last_stable, conf_val

        labels = [p[0] for p in self._stability_queue]
        if all(l == labels[0] for l in labels) and conf_val >= CONFIDENCE_THRESHOLD:
            self._last_stable = label
        else:
            self._last_stable = "?"

        return self._last_stable, conf_val
