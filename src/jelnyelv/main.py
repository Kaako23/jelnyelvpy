"""Sign language recognition app - Record & Train, Recognize tabs."""

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
from jelnyelv.video_import import get_video_duration, import_from_video, sec_to_mmss


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
    """Create Gradio UI with Record & Train, Recognize tabs."""
    word_choices = get_words_for_record()

    with gr.Blocks(title="Jelnyelv – Sign Language Recognition") as demo:
        gr.Markdown("# Jelnyelv – Sign Language Recognition")
        gr.Markdown("**1. Record & Train** → **2. Recognize**")

        with gr.Tabs():
            # Record & Train tab
            with gr.Tab("1. Record & Train"):
                gr.Markdown(
                    "Add training data by **recording with your webcam** or **importing a video file**. "
                    "Each sequence is 31 frames. Training runs automatically when data is added."
                )

                word_input = gr.Dropdown(
                    choices=word_choices,
                    label="Word (label for the sign)",
                    value=word_choices[0] if word_choices else None,
                    allow_custom_value=True,
                    filterable=True,
                )

                # --- Record from webcam ---
                with gr.Accordion("Record from webcam", open=True):
                    gr.Markdown(
                        "Capture 30 sequences live. Set duration per sequence; we sample to 31 frames. "
                        "1 sec pause between sequences."
                    )
                    with gr.Row():
                        record_preview = gr.Image(
                            label="Live preview",
                            type="numpy",
                        )
                        with gr.Column(scale=1):
                            sec_per_seq = gr.Slider(
                                minimum=2,
                                maximum=8,
                                value=4,
                                step=1,
                                label="Seconds per sequence",
                            )
                            record_btn = gr.Button("Record all 30", variant="primary")
                    record_status = gr.Textbox(label="Status", interactive=False)

                train_status = gr.Textbox(
                    label="Training status",
                    value="—",
                    interactive=False,
                    lines=6,
                )

                record_btn.click(
                    fn=record_all_sequences_generator,
                    inputs=[word_input, sec_per_seq],
                    outputs=[record_preview, record_status, word_input, train_status],
                )

                # --- Import from video ---
                with gr.Accordion("Import from video file", open=False):
                    gr.Markdown(
                        "Upload a local video. We extract 31-frame sequences and add them to the word above. "
                        "Use the range slider to trim (mm:ss format)."
                    )
                    with gr.Row():
                        video_input = gr.Video(
                            label="Video",
                            sources=["upload"],
                        )
                        with gr.Column(scale=1):
                            import_range_display = gr.Markdown(
                                value="*Select a video to set range*",
                            )
                            import_start = gr.Slider(
                                minimum=0,
                                maximum=60,
                                value=0,
                                step=1,
                                precision=0,
                                label="From",
                            )
                            import_end = gr.Slider(
                                minimum=0,
                                maximum=60,
                                value=60,
                                step=1,
                                precision=0,
                                label="To",
                            )
                            import_btn = gr.Button("Import", variant="secondary")
                    import_status = gr.Textbox(label="Status", interactive=False)

                def on_video_loaded(video):
                    duration, _, err = get_video_duration(video)
                    if err:
                        return (
                            gr.update(maximum=60, value=0),
                            gr.update(maximum=60, value=60),
                            gr.update(value=f"*{err}*"),
                        )
                    duration = max(1, int(duration))
                    return (
                        gr.update(maximum=duration, value=0),
                        gr.update(maximum=duration, minimum=0, value=duration),
                        gr.update(value=f"**Range:** 00:00 → {sec_to_mmss(duration)}"),
                    )

                def on_range_change(start, end):
                    start, end = min(start, end), max(start, end)
                    return gr.update(value=f"**Range:** {sec_to_mmss(start)} → {sec_to_mmss(end)}")

                video_input.change(
                    fn=on_video_loaded,
                    inputs=[video_input],
                    outputs=[import_start, import_end, import_range_display],
                )
                import_start.change(
                    fn=on_range_change,
                    inputs=[import_start, import_end],
                    outputs=[import_range_display],
                )
                import_end.change(
                    fn=on_range_change,
                    inputs=[import_start, import_end],
                    outputs=[import_range_display],
                )

                def do_import(video, w, start, end):
                    start_sec = min(start, end)
                    end_sec = max(start, end)
                    msg, choices = import_from_video(video, w, start_sec, end_sec)
                    out = gr.update(value=msg)
                    if choices is not None:
                        dd = gr.update(choices=choices)
                        train_msg, train_err = train_model()
                        train_out = f"Error: {train_err}" if train_err else train_msg
                        return out, dd, gr.update(value=train_out)
                    return out, gr.update(), gr.update()

                import_btn.click(
                    fn=do_import,
                    inputs=[video_input, word_input, import_start, import_end],
                    outputs=[import_status, word_input, train_status],
                )

            # Recognize tab
            with gr.Tab("2. Recognize"):
                gr.Markdown(
                    "Click **Start recognition** to begin. Show your sign to the camera. "
                    "Recognition starts after ~10 frames (~0.3 sec). Click **Stop** when done."
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
