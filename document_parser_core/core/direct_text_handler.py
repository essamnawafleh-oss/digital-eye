"""Level-1 provider: native PDF text layer with text-quality assessment."""

from __future__ import annotations

from ..config import RoutingConfig
from ..pdf_handler import LoadedDocument
from ..schemas import Line, PageContent, TextQuality
from .. import text_utils as tu

NAME = "direct_text"


def assess_quality(
    lines: list[Line],
    routing: RoutingConfig,
    image_coverage: float = 0.0,
    is_ocr: bool = False,
    drawing_count: int = 0,
) -> TextQuality:
    text = "\n".join(line.text for line in lines)
    tokens = tu.tokens_of(text)
    chars = sum(1 for c in text if not c.isspace())
    q = TextQuality(
        char_count=chars,
        word_count=len(tokens),
        arabic_char_ratio=round(tu.arabic_char_ratio(text), 4),
        garbage_ratio=round(tu.garbage_ratio(text), 4),
        suspicious_token_ratio=round(tu.suspicious_token_ratio(tokens), 4),
        presentation_form_ratio=round(tu.presentation_form_ratio(text), 4),
        reversed_arabic_score=tu.reversed_arabic_score(tokens),
        image_coverage=round(image_coverage, 4),
    )
    scores = [w.confidence for line in lines for w in line.words if w.confidence is not None]
    if is_ocr and scores:
        q.mean_confidence = round(sum(scores) / len(scores), 4)
        q.low_confidence_ratio = round(
            sum(s < routing.min_ocr_confidence for s in scores) / len(scores), 4
        )

    reasons = []
    if chars == 0:
        reasons.append("no_text")
    elif not is_ocr and chars < routing.min_chars_per_page and image_coverage >= 0.1:
        # Short text on a page with images: likely a scan with a partial text layer.
        # Short text on a clean vector page is accepted (e.g. a notes page).
        reasons.append(f"too_little_text({chars}<{routing.min_chars_per_page})")
    if q.garbage_ratio > routing.max_garbage_ratio:
        reasons.append(f"garbage_characters({q.garbage_ratio:.2f})")
    if q.suspicious_token_ratio > routing.max_suspicious_token_ratio:
        reasons.append(f"broken_glyph_mapping_suspected({q.suspicious_token_ratio:.2f})")
    if q.reversed_arabic_score is not None and q.reversed_arabic_score >= routing.reversed_arabic_threshold:
        reasons.append(f"arabic_visual_order_suspected({q.reversed_arabic_score:.2f})")
    if not is_ocr and image_coverage >= routing.min_image_coverage_for_scan and chars < 200:
        # A page mostly covered by an image with little text is probably a scan
        # whose text layer (if any) is partial, e.g. only a stamp or header.
        reasons.append(f"image_dominated_page(coverage={image_coverage:.2f})")
    if not is_ocr and drawing_count >= routing.min_drawings_for_vector_page and chars < 200:
        # CAD/plan exports often draw labels as vector paths: little real text.
        reasons.append(f"vector_dominated_page(paths={drawing_count})")
    if is_ocr and q.low_confidence_ratio is not None and q.low_confidence_ratio > 0.3:
        reasons.append(f"many_low_confidence_words({q.low_confidence_ratio:.2f})")
    if q.presentation_form_ratio > 0:
        # Not a failure (NFKC folds them), but worth surfacing.
        q.reasons.append(f"info:presentation_forms_normalised({q.presentation_form_ratio:.2f})")
    q.reasons = reasons + q.reasons
    q.is_reliable = not reasons
    return q


def extract_page(doc: LoadedDocument, index: int, routing: RoutingConfig) -> PageContent:
    geom = doc.geometry(index)
    lines = doc.native_lines(index)
    coverage = doc.image_coverage(index) if doc.format == "pdf" else 1.0
    # Assess before normalising so presentation-form usage is still visible.
    quality = assess_quality(lines, routing, image_coverage=coverage, drawing_count=doc.drawing_count(index))
    for line in lines:
        for w in line.words:
            w.text = tu.normalize_text(w.text)
        line.text = " ".join(w.text for w in line.words)
    return PageContent(
        page_number=index + 1,
        width=geom.width,
        height=geom.height,
        coord_space=geom.coord_space,
        text_source=NAME if lines else "none",
        lines=lines,
        quality=quality,
    )
