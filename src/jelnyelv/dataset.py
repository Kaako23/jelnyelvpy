"""Load/save sequences and word list. Streaming Dataset for scalable training."""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from jelnyelv.config import DATA_PATH, INPUT_SIZE, NO_SEQUENCES, SEQUENCE_LENGTH


def load_labels_from_data() -> tuple[list[str] | None, str | None]:
    """Derive labels from folder names under data/jelek/ that have at least one complete sequence.
    Returns (labels, error_message). Labels are sorted for deterministic order."""
    if not os.path.isdir(DATA_PATH):
        return None, f"No recorded data. Create {DATA_PATH} and record sequences first."

    labels = []
    for name in sorted(os.listdir(DATA_PATH)):
        action_path = os.path.join(DATA_PATH, name)
        if not os.path.isdir(action_path):
            continue
        for seq_dir in os.listdir(action_path):
            if not seq_dir.isdigit():
                continue
            seq_path = os.path.join(action_path, seq_dir)
            if not os.path.isdir(seq_path):
                continue
            frame_count = sum(1 for f in os.listdir(seq_path) if f.endswith(".npy"))
            if frame_count >= SEQUENCE_LENGTH:
                labels.append(name)
                break

    if not labels:
        return None, f"No recorded data. Record at least one complete sequence in {DATA_PATH}/<word>/<seq>/."
    return labels, None


def get_words_from_folders() -> list[str]:
    """Words that have recorded data (folder names under data/jelek/). For Record dropdown."""
    if not os.path.isdir(DATA_PATH):
        return []
    return sorted(d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d)))


def get_words_for_record() -> list[str]:
    """Words for Record tab: folders + optional szavak.txt, deduplicated and sorted.
    Labels come from recorded data; szavak.txt is optional for planning new signs."""
    from_folder = set(get_words_from_folders())
    szavak = _load_szavak_optional()
    return sorted(from_folder | szavak)


def _load_szavak_optional() -> set[str]:
    """Load words from szavak.txt if it exists. Returns empty set if missing."""
    path = Path(DATA_PATH).parent.parent / "szavak.txt"
    if not path.exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except OSError:
        return set()


def append_word_to_szavak(word: str) -> str | None:
    """Append word to szavak.txt (optional). Returns error message or None. Prevents duplicates."""
    word = word.strip()
    if not word:
        return "Word cannot be empty"
    path = Path(DATA_PATH).parent.parent / "szavak.txt"
    try:
        existing = _load_szavak_optional()
        if word in existing:
            return f"'{word}' already in list."
        with open(path, "a", encoding="utf-8") as f:
            f.write(word + "\n")
        return None
    except OSError as e:
        return str(e)


def ensure_data_directories(actions: list[str]) -> None:
    """Create data directories for each action and sequence."""
    for action in actions:
        for seq in range(NO_SEQUENCES):
            os.makedirs(os.path.join(DATA_PATH, action, str(seq)), exist_ok=True)


def _get_sequence_dirs(action_path: str) -> list[int]:
    if not os.path.isdir(action_path):
        return []
    return sorted(
        int(d) for d in os.listdir(action_path)
        if d.isdigit()
    )


def _validate_keypoints(arr: np.ndarray) -> bool:
    """Validate keypoint array has expected shape (INPUT_SIZE,)."""
    arr = np.asarray(arr)
    return arr.ndim == 1 and arr.shape[0] == INPUT_SIZE


def _load_sequence_from_disk(action_path: str, seq_idx: int) -> np.ndarray | None:
    """Load one sequence from disk. Returns (SEQUENCE_LENGTH, INPUT_SIZE) array or None if invalid."""
    window = []
    for frame_num in range(SEQUENCE_LENGTH):
        frame_path = os.path.join(action_path, str(seq_idx), f"{frame_num}.npy")
        if not os.path.exists(frame_path):
            return None
        try:
            res = np.load(frame_path, allow_pickle=True)
            if not _validate_keypoints(res):
                return None
            window.append(np.asarray(res, dtype=np.float32))
        except (OSError, ValueError):
            return None
    if len(window) != SEQUENCE_LENGTH:
        return None
    return np.stack(window, axis=0)


class StreamSequencesDataset(Dataset):
    """PyTorch Dataset that loads sequences from disk on demand. Scales to large vocabularies."""

    def __init__(
        self,
        actions: list[str],
        label_map: dict[str, int],
        indices: list[tuple[str, int]] | None = None,
    ) -> None:
        """
        Args:
            actions: Ordered list of label names.
            label_map: Mapping from label name to integer index.
            indices: Optional list of (action, seq_idx) to include. If None, scans disk for all valid sequences.
        """
        self.actions = actions
        self.label_map = label_map
        if indices is not None:
            self._indices = indices
        else:
            self._indices = self._build_indices()

    def _build_indices(self) -> list[tuple[str, int]]:
        """Scan disk and build list of (action, seq_idx) for complete sequences."""
        indices = []
        for action in self.actions:
            action_path = os.path.join(DATA_PATH, action)
            for seq_idx in _get_sequence_dirs(action_path):
                frame_path = os.path.join(action_path, str(seq_idx), f"{SEQUENCE_LENGTH - 1}.npy")
                if os.path.exists(frame_path):
                    indices.append((action, seq_idx))
        return indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        action, seq_idx = self._indices[idx]
        action_path = os.path.join(DATA_PATH, action)
        seq = _load_sequence_from_disk(action_path, seq_idx)
        if seq is None:
            raise ValueError(
                f"Invalid sequence shape or missing frames: {action}/{seq_idx}. "
                f"Expected {SEQUENCE_LENGTH} frames of {INPUT_SIZE} keypoints each."
            )
        label = self.label_map[action]
        return torch.from_numpy(seq), label
