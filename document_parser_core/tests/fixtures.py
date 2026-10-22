"""Synthetic fixture builders and fake providers for tests.

Fake providers exercise the *pipeline* (routing, evidence, validation). They say
nothing about real OCR/VLM accuracy — that is what benchmarks/evaluate.py is for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
from PIL import Image

from document_parser_core.core.base_handler import OCRProvider, VLMProvider, VLMResponse
from document_parser_core.core.paddle_handler import ocr_line
from document_parser_core.core.vlm_handler import parse_vlm_output
from document_parser_core.exceptions import MalformedModelOutputError, ProviderError

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def arabic_font() -> Path | None:
    for f in FONT_CANDIDATES:
        if Path(f).is_file():
            return Path(f)
    return None


def _html_page(doc: pymupdf.Document, rows: list[str], font: Path | None):
    page = doc.new_page(width=595, height=842)
    css, arch = "* {font-size: 11pt;}", None
    if font is not None:
        css = "@font-face {font-family: fx; src: url(%s);} * {font-family: fx; font-size: 11pt;}" % font.name
        arch = pymupdf.Archive(str(font.parent))
    y = 60
    for row in rows:
        rtl = any("\u0600" <= c <= "\u06ff" for c in row)
        html = f'<p dir="{"rtl" if rtl else "ltr"}">{row}</p>'
        page.insert_htmlbox(pymupdf.Rect(60, y, 535, y + 30), html, css=css, archive=arch)
        y += 26
    return page


def make_text_pdf(path: Path, pages: list[list[str]], font: Path | None = None) -> Path:
    doc = pymupdf.open()
    for rows in pages:
        _html_page(doc, rows, font)
    doc.subset_fonts()  # otherwise the whole font is embedded once per line
    doc.save(path, garbage=4, deflate=True)
    doc.close()
    return path


def make_table_pdf(path: Path, header: list[str], rows: list[list[str]], title: str = "Coordinates of plot corners") -> Path:
    """Coordinate table drawn with insert_text at fixed column x positions."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((60, 60), title, fontsize=12)
    xs = [60, 160, 300, 440]
    y = 100
    for i, h in enumerate(header):
        page.insert_text((xs[i], y), h, fontsize=11)
    for row in rows:
        y += 20
        for i, cell in enumerate(row):
            page.insert_text((xs[i], y), cell, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def make_scanned_pdf(source_pdf: Path, path: Path, dpi: int = 150) -> Path:
    """Rasterise every page of `source_pdf` into an image-only PDF (no text layer)."""
    src = pymupdf.open(source_pdf)
    out = pymupdf.open()
    for page in src:
        pix = page.get_pixmap(dpi=dpi)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, pixmap=pix)
    out.save(path)
    return path


def make_png(source_pdf: Path, path: Path, dpi: int = 100) -> Path:
    page = pymupdf.open(source_pdf)[0]
    pix = page.get_pixmap(dpi=dpi)
    Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(path)
    return path


def make_visual_order_arabic_pdf(path: Path) -> Path:
    """insert_text does no bidi/shaping: Arabic is stored in visual order and is
    extracted reversed — reproduces a real-world defect of legacy PDF producers."""
    font = arabic_font()
    doc = pymupdf.open()
    page = doc.new_page()
    if font:
        page.insert_font(fontname="ar", fontfile=str(font))
    lines = ["رقم الرخصة 2024/153", "اسم المالك محمد الأحمد", "المساحة الكلية 750 م2", "الحوض الجبيهة الشمالي", "تاريخ الإصدار 2024-03-01"]
    for i, line in enumerate(lines):
        page.insert_text((72, 80 + 22 * i), line, fontname="ar" if font else "helv", fontsize=12)
    doc.save(path)
    return path


# ----------------------------------------------------------------- fake providers
class TextLayerOCR(OCRProvider):
    """Fake OCR: returns the text layer of a reference PDF as 'recognised' lines.

    Lets tests drive scanned-document paths deterministically without OCR models.
    """

    name = "fake_ocr"

    def __init__(self, reference_pdf: Path, confidence: float = 0.95, overrides: dict[str, str] | None = None,
                 fail: bool = False, low_conf_tokens: set[str] | None = None):
        self.ref = pymupdf.open(reference_pdf)
        self.conf = confidence
        self.overrides = overrides or {}
        self.fail = fail
        self.low = low_conf_tokens or set()
        self.calls: list[int] = []

    def recognize(self, image, page_number, scale):
        self.calls.append(page_number)
        if self.fail:
            raise ProviderError("simulated OCR crash")
        page = self.ref[page_number - 1]
        lines = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(span["text"] for span in line["spans"]).strip()
                if not text:
                    continue
                for old, new in self.overrides.items():
                    text = text.replace(old, new)
                conf = 0.4 if any(t in text for t in self.low) else self.conf
                lines.append(ocr_line(text, list(line["bbox"]), page_number, conf))
        return lines


class ScriptedVLM(VLMProvider):
    """Fake VLM returning scripted raw outputs; parsing uses the real validator."""

    name = "scripted_vlm"

    def __init__(self, outputs: list[str | dict], max_retries: int = 1):
        self.outputs = [o if isinstance(o, str) else json.dumps(o, ensure_ascii=False) for o in outputs]
        self.max_retries = max_retries
        self.calls = 0

    def extract_fields(self, image, page_number, fields, document_hint=None):
        names = [f["name"] for f in fields]
        last = None
        for attempt in range(1, self.max_retries + 2):
            raw = self.outputs[min(self.calls, len(self.outputs) - 1)]
            self.calls += 1
            try:
                return VLMResponse(parse_vlm_output(raw, names), raw, attempt, "scripted")
            except MalformedModelOutputError as exc:
                last = exc
        raise MalformedModelOutputError(f"after retries: {last}")
