import os

os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
import socket
import sys
import threading
import time
import webbrowser

import gradio as gr

from jelnyelv.dataset import delete_word_data, get_words_for_record, load_labels_from_data
from jelnyelv.streaming import (
    _btn_idle,
    record_all_sequences_generator,
    recognize_generator,
)
from jelnyelv.train import train_model
from jelnyelv.video_import import get_video_duration, import_from_video, sec_to_mmss


def _check_imports() -> None:
    errors = []
    for name, mod in [("torch", "torch"), ("cv2", "cv2"), ("mediapipe", "mediapipe"), ("gradio", "gradio")]:
        try:
            __import__(mod)
        except ImportError as e:
            errors.append(f"{name}: {e}")

    if errors:
        print("Import ellenőrzés sikertelen:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)


def _refresh_word_dropdown():
    choices = get_words_for_record()
    if not choices:
        return gr.update(choices=[], value=None)
    return gr.update(choices=choices, value=choices[0])


def create_ui():
    word_choices = get_words_for_record()

    with gr.Blocks(title="Jelnyelv – jelnyelv-felismerés") as demo:
        gr.Markdown("# Jelnyelv – jelnyelv-felismerés")
        gr.Markdown("**1. Felvétel és tanítás** → **2. Felismerés**")

        with gr.Tabs():
            with gr.Tab("1. Felvétel és tanítás"):
                gr.Markdown(
                    "Tanító adatot **webkamerás felvétellel** vagy **videófájl importálásával** adhatsz hozzá. "
                    "Egy szekvencia 31 képkocka. Új adat után a tanítás automatikusan elindul."
                )

                word_input = gr.Dropdown(
                    choices=word_choices,
                    label="Szó",
                    value=word_choices[0] if word_choices else None,
                    allow_custom_value=True,
                    filterable=True,
                )

                with gr.Accordion("Felvétel webkamerával", open=True):
                    gr.Markdown(
                        "30 szekvencia élő felvétele. Állítsd be a szekvencia hosszát másodpercben; "
                        "31 képkockára mintavételezünk. Szekvenciák között 1 mp szünet."
                    )
                    with gr.Row():
                        record_preview = gr.Image(
                            label="Élő előnézet",
                            type="numpy",
                        )
                        with gr.Column(scale=1):
                            sec_per_seq = gr.Slider(
                                minimum=2,
                                maximum=8,
                                value=4,
                                step=1,
                                label="Másodperc szekvenciónként",
                            )
                            record_btn = gr.Button("Összes 30 felvétele", variant="primary")
                    record_status = gr.Textbox(label="Állapot", interactive=False)

                with gr.Accordion("Importálás videófájlból", open=False):
                    gr.Markdown(
                        "Tölts fel egy helyi videót. 31 képkockás szekvenciákat készítünk, "
                        "és a fenti szóhoz adjuk hozzá. A csúszkával vágd meg a tartományt (mm:ss)."
                    )
                    with gr.Row():
                        video_input = gr.Video(
                            label="Videó",
                            sources=["upload"],
                        )
                        with gr.Column(scale=1):
                            import_range_display = gr.Markdown(
                                value="*Válassz videót a tartomány beállításához*",
                            )
                            import_start = gr.Slider(
                                minimum=0,
                                maximum=60,
                                value=0,
                                step=1,
                                precision=0,
                                label="Mettől",
                            )
                            import_end = gr.Slider(
                                minimum=0,
                                maximum=60,
                                value=60,
                                step=1,
                                precision=0,
                                label="Meddig",
                            )
                            import_btn = gr.Button("Importálás", variant="secondary")
                    import_status = gr.Textbox(label="Állapot", interactive=False)

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
                        gr.update(value=f"**Tartomány:** 00:00 → {sec_to_mmss(duration)}"),
                    )

                def on_range_change(start, end):
                    start, end = min(start, end), max(start, end)
                    return gr.update(value=f"**Tartomány:** {sec_to_mmss(start)} → {sec_to_mmss(end)}")

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
                        train_out = f"Hiba: {train_err}" if train_err else train_msg
                        return out, dd, gr.update(value=train_out)
                    return out, gr.update(), gr.update()

                with gr.Accordion("Szó törlése", open=False):
                    gr.Markdown(
                        "Véglegesen törli a kiválasztott szó mappáját a `data/jelek/` alatt "
                        "(minden szekvencia). Opcionálisan újratanítja a modellt a megmaradt szavakra."
                    )
                    delete_retrain = gr.Checkbox(
                        label="Modell újratanítása törlés után",
                        value=True,
                    )
                    delete_btn = gr.Button("Kiválasztott szó törlése", variant="stop")
                    delete_status = gr.Textbox(label="Törlés állapota", interactive=False)

                def do_delete_word(word: str, retrain_after: bool):
                    ok, msg = delete_word_data(word)
                    dd = _refresh_word_dropdown()
                    if not ok:
                        return msg, dd, gr.update()

                    train_out = gr.update(value="—")
                    if retrain_after:
                        _labels, err = load_labels_from_data()
                        if err:
                            train_out = gr.update(
                                value=f"{msg}\n\nNem tanult újra: {err}"
                            )
                        else:
                            train_msg, train_err = train_model()
                            if train_err:
                                train_out = gr.update(
                                    value=f"{msg}\n\nTanítási hiba: {train_err}"
                                )
                            else:
                                train_out = gr.update(
                                    value=f"{msg}\n\n{train_msg}"
                                )
                    else:
                        train_out = gr.update(
                            value=f"{msg}\n\nÚjratanítás kihagyva — a következő felvétel után tanul újra."
                        )
                    return msg, dd, train_out

                train_status = gr.Textbox(
                    label="Tanítás állapota",
                    value="—",
                    interactive=False,
                    lines=6,
                )

                def do_retrain():
                    _labels, err = load_labels_from_data()
                    if err:
                        return gr.update(value=f"Nem indult tanítás: {err}")
                    train_msg, train_err = train_model()
                    if train_err:
                        return gr.update(value=f"Hiba: {train_err}")
                    return gr.update(value=train_msg)

                retrain_btn = gr.Button("Tanítás újrafuttatása", variant="secondary")

                record_btn.click(
                    fn=record_all_sequences_generator,
                    inputs=[word_input, sec_per_seq],
                    outputs=[record_preview, record_status, word_input, train_status],
                )
                import_btn.click(
                    fn=do_import,
                    inputs=[video_input, word_input, import_start, import_end],
                    outputs=[import_status, word_input, train_status],
                )
                delete_btn.click(
                    fn=do_delete_word,
                    inputs=[word_input, delete_retrain],
                    outputs=[delete_status, word_input, train_status],
                )
                retrain_btn.click(fn=do_retrain, outputs=[train_status])

            with gr.Tab("2. Felismerés"):
                gr.Markdown(
                    "A **Felismerés indítása** gombbal kezdhetsz. Mutasd a jelet a kamerának. "
                    "A **Leállítás** gombbal állíthatod meg a kamerát."
                )
                with gr.Row():
                    start_btn = gr.Button("Felismerés indítása", variant="primary", visible=True)
                    stop_btn = gr.Button("Leállítás", variant="stop", visible=False)
                rec_preview = gr.Image(
                    label="Élő kép (jelzőpontok + előrejelzés)",
                    type="numpy",
                )
                history_text = gr.Textbox(
                    label="Eddig mutattad",
                    value="—",
                    interactive=False,
                )

                rec_click = start_btn.click(
                    fn=recognize_generator,
                    inputs=[],
                    outputs=[rec_preview, history_text, start_btn, stop_btn],
                )
                stop_btn.click(
                    fn=_btn_idle,
                    inputs=[],
                    outputs=[start_btn, stop_btn],
                    cancels=[rec_click],
                )

        demo.load(_refresh_word_dropdown, outputs=word_input)

    return demo


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch_ui(demo) -> None:
    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    def _open_browser() -> None:
        time.sleep(0.8)
        webbrowser.open(url)

    threading.Thread(target=_open_browser, daemon=True).start()
    demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=False)


def main() -> None:
    _check_imports()

    demo = create_ui()
    launch_ui(demo)


if __name__ == "__main__":
    main()
