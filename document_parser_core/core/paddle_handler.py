"""PaddleOCR provider (paddleocr 3.7.0 / paddlepaddle 3.3.1, API source-verified).

* Always constructed with explicit local model directories; refuses to start (and
  therefore never downloads) when they are missing.
* Document orientation / unwarping / textline-orientation sub-models are disabled so
  no extra (unprovisioned) models are requested.
* PaddleOCR returns line polygons + recognition scores. Word boxes are interpolated
  along the line (flagged `approx_bbox`).
* The Arabic recogniser's output direction and its coverage of Latin text are
  UNVERIFIED; reversed-order output is detected by the quality assessment.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import ParserConfig, apply_offline_env
from ..exceptions import ModelNotFoundError, ProviderError, ProviderUnavailableError
from ..schemas import Line, Word
from .. import text_utils as tu
from .base_handler import OCRProvider

REQUIRED_MODEL_FILES = ("inference.yml", "inference.pdiparams")


def check_model_dir(path: Path | None, what: str) -> Path:
    if path is None:
        raise ModelNotFoundError(
            "paddle", "<unset>", what,
            "Set models_dir (or env DPC_MODELS_DIR) or paddle.det_model_dir / paddle.rec_model_dir.",
        )
    if not path.is_dir():
        raise ModelNotFoundError("paddle", str(path), what)
    missing = [f for f in REQUIRED_MODEL_FILES if not (path / f).is_file()]
    if missing:
        raise ModelNotFoundError(
            "paddle", str(path), f"{what} (missing files: {', '.join(missing)})"
        )
    return path


class PaddleOCRProvider(OCRProvider):
    name = "paddleocr"

    def __init__(self, config: ParserConfig):
        self.config = config
        self._ocr = None
        self._cross = None

    def check_ready(self) -> None:
        check_model_dir(self.config.paddle_det_dir(), f"detection model '{self.config.paddle.det_model_name}'")
        check_model_dir(self.config.paddle_rec_dir(), f"recognition model '{self.config.paddle.rec_model_name}'")
        if self.config.paddle.cross_check_rec_model_name:
            check_model_dir(self.config.paddle_cross_check_dir(),
                            f"cross-check recognition model '{self.config.paddle.cross_check_rec_model_name}' "
                            "(set paddle.cross_check_rec_model_name=None to disable)")
        try:
            import paddleocr  # noqa: F401
        except ImportError as exc:
            raise ProviderUnavailableError(
                "paddle", str(exc), "pip install -r document_parser_core/requirements-paddle.txt"
            ) from exc

    def _load(self):
        if self._ocr is not None:
            return self._ocr
        self.check_ready()
        if self.config.offline:
            apply_offline_env()
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        p = self.config.paddle
        try:
            self._ocr = PaddleOCR(
                text_detection_model_name=p.det_model_name,
                text_detection_model_dir=str(self.config.paddle_det_dir()),
                text_recognition_model_name=p.rec_model_name,
                text_recognition_model_dir=str(self.config.paddle_rec_dir()),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device=p.device,
                enable_mkldnn=p.enable_mkldnn,
                text_det_limit_type=p.det_limit_type,
                text_det_limit_side_len=p.det_limit_side_len,
            )
            if p.cross_check_rec_model_name:
                from paddleocr import TextRecognition

                self._cross = TextRecognition(
                    model_name=p.cross_check_rec_model_name,
                    model_dir=str(self.config.paddle_cross_check_dir()),
                    device=p.device,
                    enable_mkldnn=p.enable_mkldnn,
                )
        except Exception as exc:
            raise ProviderError(f"PaddleOCR failed to initialise: {exc}") from exc
        return self._ocr

    def recognize(self, image: Image.Image, page_number: int, scale: float) -> list[Line]:
        ocr = self._load()
        # paddlex treats ndarray input as BGR (image_reader.py) -> convert from RGB.
        arr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        try:
            results = ocr.predict(arr)
        except Exception as exc:
            raise ProviderError(f"PaddleOCR inference failed on page {page_number}: {exc}") from exc
        raw: list[tuple[str, float, np.ndarray]] = []
        for res in results:
            texts = list(res["rec_texts"])
            scores = [float(s) for s in res["rec_scores"]]
            polys = [np.asarray(p, dtype=float) for p in res["rec_polys"]]
            raw += [(t, sc, po) for t, sc, po in zip(texts, scores, polys, strict=True) if t.strip()]
        cross = self._cross_check(arr, raw, page_number)
        p = self.config.paddle
        lines: list[Line] = []
        for i, (text, score, poly) in enumerate(raw):
            xs, ys = poly[:, 0] * scale, poly[:, 1] * scale
            bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            alt = cross.get(i)
            r = merge_line_readings(text, score, *(alt or (None, None)), merge_min=p.latin_merge_min_score)
            line = ocr_line(r.text, bbox, page_number, r.score, latin_score=r.latin_score)
            line.reading = r.reading
            line.cross_check_text = r.cross_text
            if r.cross_text is not None:
                line.cross_check_model = p.rec_model_name if r.reading == "latin" else p.cross_check_rec_model_name
            lines.append(line)
        lines.sort(key=lambda ln: (round(ln.bbox[1] / 4), ln.bbox[0]))
        for i, line in enumerate(lines):
            for w in line.words:
                w.line_id = i
        return lines

    def _cross_check(self, arr: np.ndarray, raw: list, page_number: int) -> dict[int, tuple[str, float]]:
        """Re-read every line with the Latin/digit model. All lines, not only those where
        the Arabic model saw digits: across fonts it often drops Latin/digit runs entirely
        ('تاريخ الإصدار: 13/01/2017' -> 'تاريخ الإصدار')."""
        if self._cross is None:
            return {}
        idx, crops = [], []
        h, w = arr.shape[:2]
        for i, (_text, _score, poly) in enumerate(raw):
            x0, y0 = np.floor(poly.min(0)).astype(int) - 2
            x1, y1 = np.ceil(poly.max(0)).astype(int) + 2
            crop = arr[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
            if crop.shape[0] >= 4 and crop.shape[1] >= 4:
                idx.append(i)
                crops.append(np.ascontiguousarray(crop))
        if not crops:
            return {}
        try:
            out = self._cross.predict(crops)
        except Exception as exc:
            raise ProviderError(f"OCR cross-check failed on page {page_number}: {exc}") from exc
        return {i: (str(o["rec_text"]).strip(), float(o["rec_score"])) for i, o in zip(idx, out, strict=True)}


_AR = "ء-يٱ-ۓ"
_GLUED = re.compile(rf"(?<=[{_AR}])(?=[A-Za-z0-9])|(?<=[A-Za-z0-9])(?=[{_AR}])")
_COLON = re.compile(rf"(?<=[A-Za-z{_AR}]):(?=[A-Za-z0-9{_AR}])")


def split_script_boundaries(text: str) -> str:
    """Insert spaces the recogniser dropped where Arabic meets Latin/digits and after a
    colon glued to the next word. Observed with arabic_PP-OCRv5_mobile_rec:
    'BP-2024/0153رقم الرخصة', 'Owner:محمد الخطيب'. Arabic-Indic digits are untouched,
    so 'م2' becomes 'م 2' (the unit parser accepts both)."""
    return _GLUED.sub(" ", _COLON.sub(": ", text))


@dataclass
class MergedReading:
    text: str
    score: float | None
    latin_score: float | None  # score for Latin/digit tokens when they come from the Latin model
    cross_text: str | None  # independent reading kept for the cross-check
    reading: str  # "arabic" | "latin" | "arabic+latin"


# ':' is not a joiner: the Latin model glues junk from Arabic glyphs ('13/01/2017:j').
_LATIN_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[./\-,][A-Za-z0-9]+)*")
_ARABIC_INDIC = re.compile(r"[٠-٩۰-۹]")


def _latin_tokens(text: str) -> list[str]:
    toks = _LATIN_TOKEN.findall(text)
    return [t for t in toks if re.search(r"\d", t) or len(t) >= 2]


def merge_line_readings(ar_text: str, ar_score: float, en_text: str | None, en_score: float | None,
                        merge_min: float = 0.80) -> MergedReading:
    """Combine the Arabic and the Latin model's readings of one line.

    Measured on synthetic lines in 8 Arabic fonts (benchmarks/arabic_lines.py, 2026-10-24):
    the Arabic model dropped ASCII digits/Latin values in Arabic lines in most fonts,
    while the Latin model read 98% of them; on Arabic-Indic digits the Latin model
    produces plausible junk ('١٢٥٠٫٥' -> '2110.0') with scores <= 0.72, versus >= 0.71
    for real ASCII values. Rules:
      * no Arabic in either reading -> Latin model is primary;
      * Arabic line with Arabic-Indic digits -> Arabic reading only (Latin digits are junk);
      * Arabic line, Latin model >= merge_min -> Arabic words + Latin/digit tokens from
        the Latin model; the Arabic model's own Latin/digit runs become the cross-check;
      * otherwise -> Arabic reading, Latin reading kept as cross-check.
    """
    if en_text is None:
        return MergedReading(ar_text, ar_score, None, None, "arabic")
    if not tu.has_arabic(ar_text) and not tu.has_arabic(en_text):
        return MergedReading(en_text, en_score, None, ar_text, "latin")
    if not tu.has_arabic(ar_text) or _ARABIC_INDIC.search(ar_text):
        return MergedReading(ar_text, ar_score, None, en_text, "arabic")
    tokens = _latin_tokens(en_text)
    if tokens and en_score is not None and en_score >= merge_min:
        ar_core = re.sub(r"\s+", " ", _LATIN_TOKEN.sub(" ", ar_text)).strip(" :-")
        ar_own = _latin_tokens(ar_text)
        return MergedReading(f"{ar_core} {' '.join(tokens)}", ar_score, en_score,
                             ar_text if ar_own else None, "arabic+latin")
    return MergedReading(ar_text, ar_score, None, en_text, "arabic")


def ocr_line(text: str, bbox: list[float], page: int, score: float | None, latin_score: float | None = None) -> Line:
    """Build a Line from a recognised text line, interpolating word boxes by
    character count (right-to-left for Arabic lines). Tokens with Latin letters or
    ASCII digits get `latin_score` when given (they came from the Latin model)."""
    text = split_script_boundaries(tu.normalize_text(text))
    tokens = tu.tokens_of(text)
    total = sum(len(t) for t in tokens) + max(0, len(tokens) - 1)
    x0, y0, x1, y1 = bbox
    width = (x1 - x0) or 1.0
    rtl = tu.arabic_char_ratio(text) > 0.5  # same rule as extraction.build_rows
    words, offset = [], 0
    for tok in tokens:
        a, b = offset / total, (offset + len(tok)) / total
        if rtl:
            wx0, wx1 = x1 - b * width, x1 - a * width
        else:
            wx0, wx1 = x0 + a * width, x0 + b * width
        conf = latin_score if latin_score is not None and re.search(r"[A-Za-z0-9]", tok) else score
        words.append(Word(text=tok, bbox=[wx0, y0, wx1, y1], page=page, confidence=conf, approx_bbox=True))
        offset += len(tok) + 1
    confs = [w.confidence for w in words if w.confidence is not None]
    return Line(text=" ".join(tokens), bbox=bbox, page=page, words=words, confidence=min(confs) if confs else score)
