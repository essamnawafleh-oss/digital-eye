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

    def check_ready(self) -> None:
        check_model_dir(self.config.paddle_det_dir(), f"detection model '{self.config.paddle.det_model_name}'")
        check_model_dir(self.config.paddle_rec_dir(), f"recognition model '{self.config.paddle.rec_model_name}'")
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
        lines: list[Line] = []
        for res in results:
            texts = list(res["rec_texts"])
            scores = [float(s) for s in res["rec_scores"]]
            polys = [np.asarray(p, dtype=float) for p in res["rec_polys"]]
            for text, score, poly in zip(texts, scores, polys, strict=True):
                if not text.strip():
                    continue
                xs, ys = poly[:, 0] * scale, poly[:, 1] * scale
                bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                lines.append(ocr_line(text, bbox, page_number, score))
        lines.sort(key=lambda ln: (round(ln.bbox[1] / 4), ln.bbox[0]))
        for i, line in enumerate(lines):
            for w in line.words:
                w.line_id = i
        return lines


def ocr_line(text: str, bbox: list[float], page: int, score: float | None) -> Line:
    """Build a Line from a recognised text line, interpolating word boxes by
    character count (right-to-left for Arabic lines)."""
    text = tu.normalize_text(text)
    tokens = tu.tokens_of(text)
    total = sum(len(t) for t in tokens) + max(0, len(tokens) - 1)
    x0, y0, x1, y1 = bbox
    width = (x1 - x0) or 1.0
    rtl = tu.has_arabic(text)
    words, offset = [], 0
    for tok in tokens:
        a, b = offset / total, (offset + len(tok)) / total
        if rtl:
            wx0, wx1 = x1 - b * width, x1 - a * width
        else:
            wx0, wx1 = x0 + a * width, x0 + b * width
        words.append(Word(text=tok, bbox=[wx0, y0, wx1, y1], page=page, confidence=score, approx_bbox=True))
        offset += len(tok) + 1
    return Line(text=" ".join(tokens), bbox=bbox, page=page, words=words, confidence=score)
