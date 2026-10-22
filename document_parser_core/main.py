"""Public entry point: `parse_document(file_path, document_type=None, options=None) -> dict`."""

from __future__ import annotations

import contextlib
from typing import Any

from . import __version__
from .config import ParserConfig, apply_offline_env
from .document_types import TEMPLATES
from .core import create_ocr_provider, create_vlm_provider
from .exceptions import DocumentParserError
from .offline import network_guard
from .pdf_handler import LoadedDocument
from .router import Router
from .schemas import (
    DocumentTypeInfo, PageSummary, ParseResult, ProcessingStatus, ValidationIssue,
)
from .telemetry import Telemetry, peak_rss_mb


def parse_document(file_path: str, document_type: str | None = None, options: dict | None = None) -> dict:
    """Parse a PDF/PNG/JPEG file and return a JSON-serialisable result dict.

    Never raises for document problems: errors are reported with status "failed".
    `options` may contain any `ParserConfig` field (nested dicts for `paddle`, `vlm`,
    `routing`), plus:
      * ``fields``: list of extra/override field specs (see document_types.FieldSpec)
      * ``providers``: {"ocr": OCRProvider, "vlm": VLMProvider} to inject providers
      * ``config``: a ready `ParserConfig`
    """
    return parse_document_model(file_path, document_type, options).model_dump(mode="json")


def parse_document_model(file_path: str, document_type: str | None = None, options: dict | None = None) -> ParseResult:
    options = dict(options or {})
    custom_fields = options.get("fields")
    injected: dict[str, Any] = options.get("providers") or {}
    result = ParseResult()
    try:
        cfg = ParserConfig.from_options(options)
    except DocumentParserError as exc:
        result.error = {"type": type(exc).__name__, "message": str(exc)}
        result.review_reasons.append("invalid configuration")
        return result

    tm = Telemetry(cfg.trace_python_memory)
    result.processing.offline_mode = cfg.offline
    result.processing.package_version = __version__
    result.processing.config = cfg.snapshot()
    if cfg.offline:
        apply_offline_env()

    ocr = injected.get("ocr", "auto")
    vlm = injected.get("vlm", "auto")
    created = []
    guard = network_guard() if cfg.strict_network_guard else contextlib.nullcontext()
    try:
        with guard:
            if ocr == "auto":
                ocr = _safe_create(create_ocr_provider, cfg, tm, "ocr")
                created.append(ocr)
            if vlm == "auto":
                vlm = _safe_create(create_vlm_provider, cfg, tm, "vlm")
                created.append(vlm)
            with LoadedDocument(file_path) as doc:
                result.document = doc.metadata()
                router = Router(cfg, ocr, vlm, tm)
                fields, st, type_info = router.run(doc, document_type, custom_fields)
                _assemble(result, cfg, fields, st, type_info)
                result.document.has_text_layer = any(
                    c.text_source == "direct_text" for contents in st.pages.values() for c in contents
                )
    except DocumentParserError as exc:
        result.status = ProcessingStatus.FAILED
        result.review_required = True
        result.error = {"type": type(exc).__name__, "code": exc.code, "message": str(exc)}
        result.review_reasons.append(f"processing failed: {exc}")
    except Exception as exc:  # unexpected bug: report, don't fabricate
        result.status = ProcessingStatus.FAILED
        result.review_required = True
        result.error = {"type": type(exc).__name__, "code": "internal_error", "message": str(exc)}
        result.review_reasons.append("internal error")
    finally:
        for p in created:
            if p is not None:
                with contextlib.suppress(Exception):
                    p.close()
    result.processing.stages = tm.stages
    result.processing.duration_ms = tm.elapsed_ms()
    result.processing.peak_python_memory_mb = tm.python_peak_mb()
    result.processing.peak_rss_mb = peak_rss_mb()
    return result


def _safe_create(factory, cfg, tm, kind):
    try:
        return factory(cfg)
    except DocumentParserError as exc:
        tm.record(1 if kind == "ocr" else 3, f"{kind}_create", "unavailable", detail=str(exc))
        return None


def _assemble(result: ParseResult, cfg: ParserConfig, fields, st, type_info) -> None:
    result.document_type = DocumentTypeInfo(
        requested=type_info["requested"], detected=type_info["detected"],
        detection_score=type_info["detection_score"], used=type_info["used"],
    )
    result.fields = fields
    result.tables = st.tables
    for page_no in sorted(st.pages):
        primary = st.pages[page_no][0]
        result.pages.append(PageSummary(
            page_number=page_no, width=primary.width, height=primary.height,
            coord_space=primary.coord_space, text_source=primary.text_source,
            max_level=st.page_level.get(page_no, 1), quality=primary.quality,
            text=primary.text if cfg.include_page_text else None,
        ))
    result.processing.providers_used = sorted(st.providers_used)
    result.processing.max_level_reached = st.max_level

    issues = list(st.issues)
    reasons = list(st.review_reasons)
    for f in fields.values():
        for msg in f.validation_messages:
            sev = "error" if f.validation_status == "invalid" else "warning"
            issues.append(ValidationIssue(code=f"field_{f.validation_status}", severity=sev, message=msg, field=f.name, page=f.page))
        if f.requires_review:
            reasons.append(f"field '{f.name}': " + "; ".join(f.review_reasons or [f.status]))
    for t in st.tables:
        for msg in t.issues:
            issues.append(ValidationIssue(code="table_issue", severity="warning", message=msg, page=t.page))
        for i, row in enumerate(t.rows):
            for msg in row.issues:
                issues.append(ValidationIssue(code="table_row_issue", severity="warning", message=f"{t.table_id} row {i + 1}: {msg}", page=t.page))
        if t.requires_review:
            reasons.append(f"table {t.table_id}: needs review" + ("" if t.header_raw else " (no recognised header; column meaning unknown)"))
    if result.document_type.used != "generic" and _expects_table(result.document_type.used) and not st.tables:
        reasons.append("expected a coordinate table but none was found")

    result.validation.errors = [i for i in issues if i.severity == "error"]
    result.validation.warnings = [i for i in issues if i.severity != "error"]
    result.review_reasons = list(dict.fromkeys(reasons))
    result.review_required = bool(result.review_reasons)
    optional_gaps = any(f.status != "extracted" for f in fields.values())
    if result.review_required:
        result.status = ProcessingStatus.REVIEW_REQUIRED
    elif optional_gaps or result.validation.warnings:
        result.status = ProcessingStatus.PARTIAL
    else:
        result.status = ProcessingStatus.SUCCESS


def _expects_table(doc_type: str) -> bool:
    return TEMPLATES[doc_type].expects_coordinate_table
