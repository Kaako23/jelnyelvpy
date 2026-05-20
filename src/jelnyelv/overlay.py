from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _build_font_search() -> list[str]:
    paths: list[str] = []
    try:
        import matplotlib.font_manager as fm

        dejavu = fm.findfont("DejaVu Sans", fallback_to_default=False)
        if dejavu and Path(dejavu).is_file():
            paths.append(dejavu)
    except Exception:
        pass

    if sys.platform == "darwin":
        paths.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                "/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
            ]
        )
    elif sys.platform == "win32":
        windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
        paths.extend(
            [
                str(windir / "Fonts" / "arial.ttf"),
                str(windir / "Fonts" / "segoeui.ttf"),
            ]
        )
    else:
        paths.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            ]
        )
    return paths


_FONT_SEARCH = _build_font_search()


@lru_cache(maxsize=16)
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_SEARCH:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _font_size_for_image(height: int, scale: float) -> int:
    return max(14, int(height * scale * 0.045))


def draw_text_bottom_left_bgr(
    bgr: np.ndarray,
    text: str,
    *,
    color_bgr: tuple[int, int, int] = (0, 255, 255),
    scale: float = 1.0,
    margin: int = 20,
    text_padding: int | None = None,
    background_bgr: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    if not text:
        return bgr

    h, w = bgr.shape[:2]
    font_size = _font_size_for_image(h, scale)
    font = _load_font(font_size)
    pad = text_padding if text_padding is not None else max(12, int(font_size * 0.45))
    fill_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    bg_rgb = (background_bgr[2], background_bgr[1], background_bgr[0])

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    anchor = (margin, h - margin)
    bbox = draw.textbbox(anchor, text, font=font, anchor="ld")
    bg_box = (
        max(0, bbox[0] - pad),
        max(0, bbox[1] - pad),
        min(w, bbox[2] + pad),
        min(h, bbox[3] + pad),
    )
    draw.rectangle(bg_box, fill=bg_rgb)
    draw.text(anchor, text, font=font, fill=fill_rgb, anchor="ld")

    bgr[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return bgr
