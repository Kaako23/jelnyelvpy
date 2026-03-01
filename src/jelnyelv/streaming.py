"""Webcam streaming generators for Record and Recognize tabs."""

import os
import time

import cv2
import gradio as gr
import numpy as np

from jelnyelv.config import (
    DATA_PATH,
    NO_SEQUENCES,
    PAUSE_BETWEEN_SEQUENCES_SEC,
    RECOGNITION_INPUT_WIDTH,
    SEQUENCE_LENGTH,
)
from jelnyelv.dataset import ensure_data_directories, get_words_for_record
from jelnyelv.train import train_model
from jelnyelv.infer import Recognizer
from jelnyelv.mp_features import (
    HolisticTasks,
    draw_landmarks_on_image,
    extract_keypoints,
    mediapipe_detection,
)


def record_all_sequences_generator(word: str, seconds_per_sequence: float):
    """Generator: record 30 sequences with MediaPipe overlay, yield (frame, status, word_update, train_status).
    Captures for exactly `seconds_per_sequence` wall-clock seconds, then samples/pads to SEQUENCE_LENGTH.
    Trains automatically when done."""
    _no_change = gr.update()

    if not word or not str(word).strip():
        yield None, "Enter or select a word first.", _no_change, _no_change
        return

    word = str(word).strip()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        yield None, "Could not open camera. Check System Settings → Privacy → Camera.", _no_change, _no_change
        return

    ensure_data_directories([word])
    duration_sec = max(1.0, float(seconds_per_sequence))
    last_frame = None

    try:
        with HolisticTasks() as tasks:
            for seq in range(NO_SEQUENCES):
                seq_path = os.path.join(DATA_PATH, word, str(seq))
                os.makedirs(seq_path, exist_ok=True)
                keypoints_buffer = []
                start = time.perf_counter()

                while (time.perf_counter() - start) < duration_sec:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    _, results = mediapipe_detection(frame, tasks)
                    kp = extract_keypoints(results)
                    keypoints_buffer.append(kp)

                    elapsed = time.perf_counter() - start
                    annotated = draw_landmarks_on_image(frame, results)
                    status = (
                        f"Seq {seq + 1}/{NO_SEQUENCES}  "
                        f"{elapsed:.1f}s / {duration_sec:.0f}s"
                    )
                    cv2.putText(
                        annotated,
                        status,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2,
                    )
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    last_frame = rgb
                    yield rgb, status, _no_change, _no_change

                # Sample or pad to SEQUENCE_LENGTH and save
                if keypoints_buffer:
                    if len(keypoints_buffer) >= SEQUENCE_LENGTH:
                        indices = np.linspace(
                            0, len(keypoints_buffer) - 1, SEQUENCE_LENGTH, dtype=int
                        )
                        to_save = [keypoints_buffer[i] for i in indices]
                    else:
                        pad = [keypoints_buffer[-1]] * (
                            SEQUENCE_LENGTH - len(keypoints_buffer)
                        )
                        to_save = keypoints_buffer + pad
                    for out_idx, kp in enumerate(to_save):
                        np.save(
                            os.path.join(seq_path, f"{out_idx}.npy"),
                            np.asarray(kp, dtype=np.float32),
                        )

                if seq < NO_SEQUENCES - 1:
                    status = f"Pause 1 sec — next: Seq {seq + 2}/{NO_SEQUENCES}"
                    if last_frame is not None:
                        overlay_bgr = cv2.cvtColor(last_frame.copy(), cv2.COLOR_RGB2BGR)
                    else:
                        overlay_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
                        overlay_bgr[:] = (40, 40, 40)
                    cv2.putText(
                        overlay_bgr,
                        "1 sec pause...",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 255),
                        2,
                    )
                    overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
                    yield overlay, status, _no_change, _no_change
                    time.sleep(PAUSE_BETWEEN_SEQUENCES_SEC)

        yield last_frame, "Training model...", gr.update(choices=get_words_for_record()), "Training..."
        msg, err = train_model()
        train_status = f"Error: {err}" if err else msg
        yield (
            last_frame,
            f"Done. Saved {NO_SEQUENCES} sequences for '{word}'.",
            gr.update(choices=get_words_for_record()),
            train_status,
        )
    finally:
        cap.release()


def recognize_generator():
    """Generator: capture from webcam, run inference. Yields (frame, prediction, history, start_btn, stop_btn)."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        yield None, "Could not open camera.", "—", *_btn_idle()
        return

    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    recognizer = Recognizer()
    err = recognizer.load()
    if err:
        yield None, f"Model error: {err}", "—", *_btn_idle()
        return

    history = []
    last_added = None

    try:
        with HolisticTasks() as tasks:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                h, w = frame.shape[:2]
                small_w = RECOGNITION_INPUT_WIDTH
                small_h = int(h * small_w / w)
                small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                _, results = mediapipe_detection(small, tasks)
                kp = extract_keypoints(results)

                label, conf = recognizer.add_frame(kp)
                text = f"{label} ({conf*100:.0f}%)" if label != "?" else f"? ({conf*100:.0f}%)"

                if label != "?" and label != last_added:
                    history.append(label)
                    last_added = label
                elif label == "?":
                    last_added = None
                history_str = ", ".join(history) if history else "—"

                annotated = draw_landmarks_on_image(frame, results)
                cv2.putText(
                    annotated,
                    text,
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0) if label != "?" else (0, 165, 255),
                    2,
                )
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                yield rgb, text, history_str, *_btn_running()
    finally:
        cap.release()


def _btn_running():
    """Button states: Start hidden, Stop visible."""
    return gr.update(visible=False), gr.update(visible=True)


def _btn_idle():
    """Button states: Start visible, Stop hidden."""
    return gr.update(visible=True), gr.update(visible=False)
