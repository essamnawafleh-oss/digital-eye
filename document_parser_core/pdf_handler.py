"""Input loading for PDF, PNG and JPEG files.

PDF access goes through PyMuPDF (AGPL-3.0 / commercial, see docs/LICENSES.md) and is
confined to this module so the backend can be swapped. Images are loaded with Pillow.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .exceptions import InputFileError, ProviderUnavailableError
from .schemas import CoordSpace, DocumentMeta, Line, Word

_MAGIC = {
    b"%PDF-": "pdf",
    b"\x89PNG\r\n\x1a\n": "png",
    b"\xff\xd8\xff": "jpeg",
}
_EXTENSIONS = {".pdf": "pdf", ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}


def detect_format(path: Path) -> str:
    with path.open("rb") as fh:
        head = fh.read(1024)
    for magic, fmt in _MAGIC.items():
        # Some PDF producers put junk before %PDF; the spec allows it in the first 1 KB.
        if (fmt == "pdf" and magic in head) or head.startswith(magic):
            return fmt
    ext = _EXTENSIONS.get(path.suffix.lower())
    if ext:
        raise InputFileError(
            f"'{path.name}' has a {path.suffix} extension but its content is not a valid "
            f"{ext.upper()} file (corrupted or mislabelled)."
        )
    raise InputFileError(
        f"Unsupported file type for '{path.name}'. Supported: PDF, PNG, JPG/JPEG. "
        "(DWG/DXF are not supported by this version.)"
    )


def _import_pymupdf():
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ProviderUnavailableError(
            "pdf", str(exc), "pip install -r document_parser_core/requirements.txt"
        ) from exc
    return pymupdf


@dataclass
class PageGeometry:
    width: float
    height: float
    coord_space: CoordSpace


class LoadedDocument:
    """A PDF or image with page-level access. Use as a context manager."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise InputFileError(f"File not found: '{self.path}'")
        if not self.path.is_file():
            raise InputFileError(f"Not a file: '{self.path}'")
        if self.path.stat().st_size == 0:
            raise InputFileError(f"File is empty: '{self.path}'")
        self.format = detect_format(self.path)
        self._pdf = None
        self._image: Image.Image | None = None
        if self.format == "pdf":
            pymupdf = _import_pymupdf()
            try:
                self._pdf = pymupdf.open(self.path)
            except Exception as exc:
                raise InputFileError(f"Cannot open PDF '{self.path.name}': {exc}") from exc
            if self._pdf.needs_pass:
                raise InputFileError(f"PDF '{self.path.name}' is password-protected.")
            if self._pdf.page_count == 0:
                raise InputFileError(f"PDF '{self.path.name}' has no pages.")
        else:
            try:
                with Image.open(self.path) as img:
                    img.load()
                    self._image = img.convert("RGB")
            except (UnidentifiedImageError, OSError) as exc:
                raise InputFileError(f"Cannot decode image '{self.path.name}': {exc}") from exc

    # ---- context management ------------------------------------------------
    def close(self) -> None:
        if self._pdf is not None:
            self._pdf.close()
            self._pdf = None

    def __enter__(self) -> "LoadedDocument":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- metadata -------------------------------------------------------------
    @property
    def page_count(self) -> int:
        return self._pdf.page_count if self._pdf is not None else 1

    def metadata(self) -> DocumentMeta:
        digest = hashlib.sha256()
        with self.path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        pdf_meta = {}
        if self._pdf is not None:
            pdf_meta = {k: str(v) for k, v in (self._pdf.metadata or {}).items() if v}
        return DocumentMeta(
            path=str(self.path.resolve()),
            file_name=self.path.name,
            format=self.format,
            size_bytes=self.path.stat().st_size,
            sha256=digest.hexdigest(),
            page_count=self.page_count,
            pdf_metadata=pdf_meta,
        )

    def geometry(self, index: int) -> PageGeometry:
        if self._pdf is not None:
            rect = self._pdf[index].rect
            return PageGeometry(rect.width, rect.height, CoordSpace.PDF_POINTS)
        assert self._image is not None
        return PageGeometry(float(self._image.width), float(self._image.height), CoordSpace.IMAGE_PIXELS)

    # ---- native text -----------------------------------------------------------
    def native_lines(self, index: int) -> list[Line]:
        """Lines with word boxes from the PDF text layer (empty for images)."""
        if self._pdf is None:
            return []
        pymupdf = _import_pymupdf()
        page = self._pdf[index]
        page_no = index + 1
        # Words are built from characters and split ONLY on real whitespace glyphs.
        # MuPDF's default gap-based space insertion (also used by "words" mode)
        # splits Arabic words at non-joining letters ("ا لرخصة"); verified on 2.0.
        flags = pymupdf.TEXTFLAGS_RAWDICT | pymupdf.TEXT_INHIBIT_SPACES
        raw = page.get_text("rawdict", flags=flags)
        lines: list[Line] = []
        for block in raw.get("blocks", []):
            for ln in block.get("lines", []):
                words: list[Word] = []
                cur_text, cur_boxes = [], []
                chars = [c for span in ln.get("spans", []) for c in span.get("chars", [])]
                for ch in chars + [{"c": " ", "bbox": None}]:
                    if ch["c"].isspace():
                        if cur_text:
                            words.append(_make_word(cur_text, cur_boxes, page_no))
                        cur_text, cur_boxes = [], []
                    else:
                        cur_text.append(ch["c"])
                        cur_boxes.append(list(ch["bbox"]))
                if words:
                    lines.append(self._line(words, page_no))
        for line_id, line in enumerate(lines):
            for w in line.words:
                w.line_id = line_id
        lines.sort(key=lambda ln: (round(ln.bbox[1] / 3), ln.bbox[0]))
        return lines

    @staticmethod
    def _line(words: list[Word], page_no: int) -> Line:
        return Line(
            text=" ".join(w.text for w in words),
            bbox=union_bbox([w.bbox for w in words]),
            page=page_no,
            words=words,
        )

    def image_coverage(self, index: int) -> float:
        """Fraction of the page area covered by embedded images (scan indicator)."""
        if self._pdf is None:
            return 1.0
        page = self._pdf[index]
        area = page.rect.width * page.rect.height or 1.0
        covered = 0.0
        for info in page.get_image_info():
            x0, y0, x1, y1 = info["bbox"]
            covered += max(0.0, x1 - x0) * max(0.0, y1 - y0)
        return min(1.0, covered / area)

    def drawing_count(self, index: int, cap: int = 5000) -> int:
        """Number of vector path objects on the page (capped; 0 for images)."""
        if self._pdf is None:
            return 0
        try:
            return min(cap, len(self._pdf[index].get_cdrawings()))
        except Exception:
            return 0

    def find_tables(self, index: int) -> list[tuple[tuple, list[list[str | None]]]]:
        """PyMuPDF's vector/text table finder (native PDFs only): [(bbox, rows)]."""
        if self._pdf is None:
            return []
        try:
            tabs = self._pdf[index].find_tables()
        except Exception:
            return []
        return [(t.bbox, t.extract()) for t in tabs.tables]

    # ---- rendering -------------------------------------------------------------
    def render(self, index: int, dpi: int) -> tuple[Image.Image, float]:
        """Return (RGB image, scale) where page_coord = pixel_coord * scale."""
        if self._pdf is not None:
            pix = self._pdf[index].get_pixmap(dpi=dpi, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            return img, 72.0 / dpi
        assert self._image is not None
        return self._image.copy(), 1.0


_NUMERIC_CHARS = set("0123456789٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹.,٫٬/-:")


def _make_word(chars: list[str], boxes: list[list[float]], page_no: int) -> Word:
    # Numbers are always laid out left-to-right on the page, but MuPDF's logical
    # reordering of RTL lines can reverse Arabic-Indic digit runs ("١٢٥٠٫٥" came
    # back as "٥٫٠٥٢١" in testing). For purely numeric words, glyph geometry is
    # authoritative: order characters by x.
    if all(c in _NUMERIC_CHARS for c in chars) and any(c.isdigit() for c in chars):
        pairs = sorted(zip(chars, boxes), key=lambda cb: cb[1][0])
        chars, boxes = [c for c, _ in pairs], [b for _, b in pairs]
    return Word(text="".join(chars), bbox=union_bbox(boxes), page=page_no)


def union_bbox(boxes: list[list[float]]) -> list[float]:
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]
