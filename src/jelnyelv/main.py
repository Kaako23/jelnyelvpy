"""Sign language recognition app - Record, Train, Recognize tabs."""

import os

# macOS: ensure OpenCV can access camera (set before cv2 import when possible)
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
import socket
import sys
import threading
import time
import webbrowser

import gradio as gr

from jelnyelv.dataset import get_words_for_record
from jelnyelv.streaming import (
    _btn_idle,
    record_all_sequences_generator,
    recognize_generator,
)
from jelnyelv.train import train_model


def _check_imports() -> None:
    """Verify required imports. Exit with error message if any fail."""
    errors = []
    for name, mod in [("torch", "torch"), ("cv2", "cv2"), ("mediapipe", "mediapipe"), ("gradio", "gradio")]:
        try:
            __import__(mod)
        except ImportError as e:
            errors.append(f"{name}: {e}")

    if errors:
        print("Import check failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)


def create_ui():
    """Create Gradio UI with Record, Train, Recognize tabs."""
    word_choices = get_words_for_record()

    with gr.Blocks(title="Jelnyelv – Sign Language Recognition") as demo:
        gr.Markdown("# Jelnyelv – Sign Language Recognition")
        gr.Markdown("**1. Record** → **2. Train** → **3. Recognize**")

        with gr.Tabs():
            # Record tab
            with gr.Tab("1. Record"):
                gr.Markdown(
                    "Type a word or pick one. Set **seconds per sequence** — we capture for exactly that long (wall clock), "
                    "then sample/pad to 31 frames. Click **Record all 30**. "
                    "1 sec pause between sequences."
                )
                with gr.Row():
                    word_input = gr.Dropdown(
                        choices=word_choices,
                        label="Word",
                        value=word_choices[0] if word_choices else None,
                        allow_custom_value=True,
                        filterable=True,
                    )
                    sec_per_seq = gr.Slider(
                        minimum=2,
                        maximum=8,
                        value=4,
                        step=1,
                        label="Seconds per sequence (capture duration)",
                    )
                    record_btn = gr.Button("Record all 30 sequences", variant="primary")
                record_status = gr.Textbox(label="Status", interactive=False)
                record_preview = gr.Image(
                    label="Live preview (MediaPipe + Seq/Frame)",
                    type="numpy",
                )

                record_btn.click(
                    fn=record_all_sequences_generator,
                    inputs=[word_input, sec_per_seq],
                    outputs=[record_preview, record_status, word_input],
                )

            # Train tab
            with gr.Tab("2. Train"):
                gr.Markdown("Train the model on recorded sequences. Labels come from folder names.")
                train_btn = gr.Button("Train model", variant="primary")
                train_status = gr.Textbox(label="Status", interactive=False, lines=6)

                def do_train():
                    msg, err = train_model()
                    if err:
                        return f"Error: {err}"
                    return msg

                train_btn.click(fn=do_train, outputs=[train_status])

            # Recognize tab (same pattern as Record: button + generator, no Gradio webcam)
            with gr.Tab("3. Recognize"):
                gr.Markdown(
                    "Click **Start recognition** to begin. Show your sign to the camera. "
                    "The model needs ~31 frames (~1 sec) of the gesture. Click **Stop** when done."
                )
                with gr.Row():
                    start_btn = gr.Button("Start recognition", variant="primary", visible=True)
                    stop_btn = gr.Button("Stop", variant="stop", visible=False)
                rec_preview = gr.Image(
                    label="Live view (landmarks + prediction)",
                    type="numpy",
                )
                prediction_text = gr.Textbox(label="Prediction", interactive=False)
                history_text = gr.Textbox(
                    label="You showed",
                    value="—",
                    interactive=False,
                )

                rec_click = start_btn.click(
                    fn=recognize_generator,
                    inputs=[],
                    outputs=[rec_preview, prediction_text, history_text, start_btn, stop_btn],
                )
                stop_btn.click(
                    fn=_btn_idle,
                    inputs=[],
                    outputs=[start_btn, stop_btn],
                    cancels=[rec_click],
                )

    return demo


def _free_port() -> int:
    """Return an available port on 127.0.0.1."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch_ui(demo) -> None:
    """Launch Gradio UI, pick a port, and open the default browser after a short delay."""
    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    def _open_browser() -> None:
        time.sleep(0.8)
        webbrowser.open(url)

    threading.Thread(target=_open_browser, daemon=True).start()
    demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=False)


def main() -> None:
    """Run import check and launch Gradio app."""
    _check_imports()

    demo = create_ui()
    launch_ui(demo)


if __name__ == "__main__":
    main()
