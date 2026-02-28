"""Training logic with metrics and compact evaluation report."""

import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from jelnyelv.config import (
    BATCH_SIZE,
    EPOCHS,
    EVALUATION_REPORT_PATH,
    GRAD_CLIP_MAX_NORM,
    HIDDEN_SIZE,
    INPUT_SIZE,
    MODEL_PATH,
    TOP_N_BEST_CLASSES,
    TOP_N_CONFUSED_PAIRS,
    TOP_N_WORST_CLASSES,
)
from jelnyelv.dataset import StreamSequencesDataset, load_labels_from_data
from jelnyelv.model import LSTMModel

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """CPU or MPS (Apple Silicon)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _write_compact_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    actions: np.ndarray,
    report_path: str,
) -> None:
    """Write a compact evaluation report scalable to 2000+ classes."""
    n_classes = len(actions)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    macro_p = precision.mean()
    macro_r = recall.mean()
    macro_f1 = f1.mean()
    acc = 100 * (y_true == y_pred).mean()

    per_class_acc = {}
    for i in range(n_classes):
        mask = y_true == i
        per_class_acc[i] = (y_pred[mask] == i).sum() / mask.sum() if mask.sum() > 0 else 0.0

    sorted_by_acc = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
    best = sorted_by_acc[: min(TOP_N_BEST_CLASSES, len(sorted_by_acc))]
    worst = sorted_by_acc[-min(TOP_N_WORST_CLASSES, len(sorted_by_acc)) :][::-1]

    confusion_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((actions[i], actions[j], int(cm[i, j])))
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    top_confused = confusion_pairs[:TOP_N_CONFUSED_PAIRS]

    lines = [
        "=" * 60,
        "KIÉRTÉKELÉSI JELENTÉS (Jelnyelv felismerő modell)",
        "=" * 60,
        "",
        "--- 1. ÖSSZESÍTŐ MÉRŐSZÁMOK ---",
        "",
        f"Osztályok száma: {n_classes}",
        f"Pontosság (accuracy): {acc:.1f}%",
        f"Macro Precision: {macro_p:.3f}",
        f"Macro Recall:    {macro_r:.3f}",
        f"Macro F1:        {macro_f1:.3f}",
        "",
        "--- 2. LEGJOBBAN FELISMERT SZAVAK ---",
        "",
    ]
    for idx, acc_val in best:
        lines.append(f"  - {actions[idx]}: {acc_val*100:.1f}%")

    lines.extend([
        "",
        "--- 3. LEGNEHEZEBBEN FELISMERT SZAVAK ---",
        "",
    ])
    for idx, acc_val in worst:
        lines.append(f"  - {actions[idx]}: {acc_val*100:.1f}%")

    lines.extend([
        "",
        "--- 4. LEGGYAKORIBB ÖSSZETÉVESZTÉSEK ---",
        "",
    ])
    for true_w, pred_w, cnt in top_confused:
        lines.append(f"  - '{true_w}' → '{pred_w}': {cnt}x")

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Report saved: %s", report_path)


def train_model() -> tuple[str, str | None]:
    """Train LSTM and save checkpoint. Returns (status_message, error).
    Uses streaming dataset loading for scalability."""
    labels, err = load_labels_from_data()
    if err:
        return "", err

    actions = np.array(labels)
    label_map = {label: i for i, label in enumerate(actions)}

    full_ds = StreamSequencesDataset(actions, label_map)
    if len(full_ds) == 0:
        return "", "No recorded data. Record sequences first."

    indices = list(range(len(full_ds)))
    random.seed(42)
    random.shuffle(indices)
    split = int(len(indices) * 0.8)
    train_indices = indices[:split]
    test_indices = indices[split:]

    train_ds = Subset(full_ds, train_indices)
    test_ds = Subset(full_ds, test_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    device = get_device()
    logger.info("Device: %s", device)
    logger.info("Training samples: %d, Test samples: %d", len(train_ds), len(test_ds))

    output_size = len(actions)
    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for inp, lbl in train_loader:
            inp, lbl = inp.to(device), lbl.to(device)
            optimizer.zero_grad()
            out = model(inp)
            loss = criterion(out, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inp, lbl in test_loader:
                inp, lbl = inp.to(device), lbl.to(device)
                out = model(inp)
                _, pred = torch.max(out, 1)
                total += lbl.size(0)
                correct += (pred == lbl).sum().item()

        avg_loss = train_loss / len(train_loader)
        test_acc = 100 * correct / total
        scheduler.step(avg_loss)

        if (epoch + 1) % 5 == 0:
            logger.info("Epoch %d/%d - Loss: %.4f - Test Acc: %.1f%%", epoch + 1, EPOCHS, avg_loss, test_acc)

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inp, lbl in test_loader:
            inp = inp.to(device)
            out = model(inp)
            _, pred = torch.max(out, 1)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_labels.extend(lbl.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    _write_compact_report(y_true, y_pred, actions, EVALUATION_REPORT_PATH)

    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "labels": [str(a) for a in actions],
        "actions": actions,
        "label_map": label_map,
        "config": {
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "output_size": output_size,
        },
    }
    torch.save(checkpoint, MODEL_PATH)

    acc_pct = 100 * (y_true == y_pred).mean()
    return (
        f"Model trained and saved: {MODEL_PATH}\n\n"
        f"Test accuracy: {acc_pct:.1f}%\n"
        f"Report: {EVALUATION_REPORT_PATH}"
    ), None
