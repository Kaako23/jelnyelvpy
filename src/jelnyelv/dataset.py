import os
import shutil

import numpy as np
import torch
from torch.utils.data import Dataset

from jelnyelv.config import DATA_PATH, INPUT_SIZE, NO_SEQUENCES, SEQUENCE_LENGTH
from jelnyelv.mp_features import scale_keypoint_vector


def load_labels_from_data() -> tuple[list[str] | None, str | None]:
    if not os.path.isdir(DATA_PATH):
        return None, f"Nincs felvett adat. Hozd létre: {DATA_PATH}, majd vegyél fel szekvenciákat."

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
        return None, f"Nincs felvett adat. Legalább egy teljes szekvenciát vegyél fel: {DATA_PATH}/<szó>/<szekv>/."
    return labels, None


def get_words_from_folders() -> list[str]:
    if not os.path.isdir(DATA_PATH):
        return []
    return sorted(d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d)))


def get_words_for_record() -> list[str]:
    return get_words_from_folders()


def _sanitize_label_name(word: str) -> str | None:
    if not word or not str(word).strip():
        return None
    name = os.path.basename(str(word).strip())
    if not name or name in (".", "..") or os.path.sep in name or "/" in name or "\\" in name:
        return None
    return name


def delete_word_data(word: str) -> tuple[bool, str]:
    name = _sanitize_label_name(word)
    if name is None:
        return False, "Adj meg vagy válassz érvényes szócímkét."

    path = os.path.join(DATA_PATH, name)
    if not os.path.isdir(path):
        return False, f"Nincs mappa a(z) „{name}” szóhoz a data/jelek/ alatt."

    shutil.rmtree(path)
    return True, f"A(z) „{name}” szó összes felvett adata törölve."


def ensure_data_directories(actions: list[str]) -> None:
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
    arr = np.asarray(arr)
    return arr.ndim == 1 and arr.shape[0] == INPUT_SIZE


def _load_sequence_from_disk(action_path: str, seq_idx: int) -> np.ndarray | None:
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
    seq = np.stack(window, axis=0)
    return scale_keypoint_vector(seq)


class StreamSequencesDataset(Dataset):
    def __init__(
        self,
        actions: list[str],
        label_map: dict[str, int],
        indices: list[tuple[str, int]] | None = None,
    ) -> None:
        self.actions = actions
        self.label_map = label_map
        if indices is not None:
            self._indices = indices
        else:
            self._indices = self._build_indices()

    def _build_indices(self) -> list[tuple[str, int]]:
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
                f"Érvénytelen szekvencia vagy hiányzó képkockák: {action}/{seq_idx}. "
                f"Elvárt: {SEQUENCE_LENGTH} képkocka, egyenként {INPUT_SIZE} kulcspont."
            )
        label = self.label_map[action]
        return torch.from_numpy(seq), label
