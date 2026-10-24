"""Public result schema (version 1.0).

Serialization rules
-------------------
* The public API returns `ParseResult.model_dump(mode="json")`: plain JSON types only.
* `null` means "no value" — the *reason* is always given by `FieldResult.status`:
    - ``extracted``          a value was found (see `validation_status` for whether it is valid)
    - ``missing``            no evidence for the field was found anywhere
    - ``unreadable``         the field's location/label was found but the value is illegible
                             or below the confidence threshold (`raw_text` keeps what was read)
    - ``explicitly_absent``  the document states the value is absent (e.g. "N/A", "لا يوجد")
    - ``conflict``           providers disagree; candidates are listed, no value is chosen
* Bounding boxes are ``[x0, y0, x1, y1]`` with a top-left origin, in the page's
  `coord_space`: ``pdf_points`` (1/72 inch) for PDF pages, ``image_pixels`` for images.
* Page numbers are 1-based.
* `confidence` is a provider score whose meaning is given by `confidence_kind`; it is
  never a calibrated probability. VLM self-reports are labelled
  ``vlm_self_report_uncalibrated`` and are not used for acceptance decisions.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

BBox = list[float]  # [x0, y0, x1, y1]


class CoordSpace(str, Enum):
    PDF_POINTS = "pdf_points"
    IMAGE_PIXELS = "image_pixels"


class FieldStatus(str, Enum):
    EXTRACTED = "extracted"
    MISSING = "missing"
    UNREADABLE = "unreadable"
    EXPLICITLY_ABSENT = "explicitly_absent"
    CONFLICT = "conflict"


class ValidationStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    NOT_VALIDATED = "not_validated"


class ConfidenceKind(str, Enum):
    OCR_SCORE = "ocr_recognition_score"
    NATIVE_TEXT = "native_text_layer"
    HEURISTIC = "heuristic"
    VLM_SELF_REPORT = "vlm_self_report_uncalibrated"
    NONE = "none"


class ProcessingStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    REVIEW_REQUIRED = "review_required"
    FAILED = "failed"


class _Model(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")


class Word(_Model):
    text: str
    bbox: BBox
    page: int
    confidence: float | None = None
    line_id: int | None = None
    # True when the box was interpolated from a line box (OCR without word boxes).
    approx_bbox: bool = False


class Line(_Model):
    text: str
    bbox: BBox
    page: int
    words: list[Word] = Field(default_factory=list)
    confidence: float | None = None  # min word / line score where available
    # Independent second reading of the same line image (OCR cross-check recogniser).
    cross_check_text: str | None = None
    cross_check_model: str | None = None
    # Which recogniser(s) produced `text`: "arabic", "latin" or "arabic+latin".
    reading: str | None = None


class TextQuality(_Model):
    char_count: int = 0
    word_count: int = 0
    arabic_char_ratio: float = 0.0
    garbage_ratio: float = 0.0
    suspicious_token_ratio: float = 0.0
    presentation_form_ratio: float = 0.0
    reversed_arabic_score: float | None = None
    image_coverage: float = 0.0
    mean_confidence: float | None = None
    low_confidence_ratio: float | None = None
    is_reliable: bool = False
    reasons: list[str] = Field(default_factory=list)


class PageContent(_Model):
    """Internal and public representation of one page's text."""

    page_number: int
    width: float
    height: float
    coord_space: CoordSpace
    text_source: str  # "native_text" | "ocr:<provider>" | "none"
    lines: list[Line] = Field(default_factory=list)
    quality: TextQuality = Field(default_factory=TextQuality)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def words(self) -> list[Word]:
        return [w for line in self.lines for w in line.words]


class Evidence(_Model):
    provider: str
    stage: str
    level: int
    method: str
    page: int | None = None
    bbox: BBox | None = None
    raw_text: str | None = None
    value: Any = None
    confidence: float | None = None
    confidence_kind: ConfidenceKind = ConfidenceKind.NONE
    note: str | None = None


class FieldResult(_Model):
    name: str
    value: Any = None
    raw_text: str | None = None
    status: FieldStatus = FieldStatus.MISSING
    validation_status: ValidationStatus = ValidationStatus.NOT_VALIDATED
    validation_messages: list[str] = Field(default_factory=list)
    page: int | None = None
    bbox: BBox | None = None
    provider: str | None = None
    confidence: float | None = None
    confidence_kind: ConfidenceKind = ConfidenceKind.NONE
    requires_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class TableRow(_Model):
    cells: list[str]
    values: dict[str, Any] = Field(default_factory=dict)
    bbox: BBox | None = None
    confidence: float | None = None
    issues: list[str] = Field(default_factory=list)


class TableResult(_Model):
    table_id: str
    kind: str  # "coordinate_table" | "generic"
    page: int
    bbox: BBox | None = None
    provider: str
    method: str
    columns: list[str]
    header_raw: list[str] | None = None
    rows: list[TableRow] = Field(default_factory=list)
    validation_status: ValidationStatus = ValidationStatus.NOT_VALIDATED
    issues: list[str] = Field(default_factory=list)
    requires_review: bool = False


class ValidationIssue(_Model):
    code: str
    severity: str  # "error" | "warning"
    message: str
    field: str | None = None
    page: int | None = None


class StageRecord(_Model):
    level: int
    stage: str
    provider: str | None = None
    pages: list[int] = Field(default_factory=list)
    duration_ms: float = 0.0
    outcome: str  # "ok" | "skipped" | "failed" | "unavailable"
    detail: str | None = None


class DocumentMeta(_Model):
    path: str
    file_name: str
    format: str  # "pdf" | "png" | "jpeg"
    size_bytes: int
    sha256: str
    page_count: int
    has_text_layer: bool | None = None
    pdf_metadata: dict[str, str] = Field(default_factory=dict)


class DocumentTypeInfo(_Model):
    requested: str | None = None
    detected: str | None = None
    detection_method: str = "keyword_heuristic"
    detection_score: float | None = None  # heuristic keyword score, not a probability
    used: str = "generic"


class PageSummary(_Model):
    page_number: int
    width: float
    height: float
    coord_space: CoordSpace
    text_source: str
    max_level: int
    quality: TextQuality
    text: str | None = None


class ProcessingInfo(_Model):
    stages: list[StageRecord] = Field(default_factory=list)
    providers_used: list[str] = Field(default_factory=list)
    max_level_reached: int = 0
    duration_ms: float = 0.0
    peak_python_memory_mb: float | None = None
    peak_rss_mb: float | None = None
    offline_mode: bool = True
    package_version: str = ""
    config: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(_Model):
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)


class ParseResult(_Model):
    schema_version: str = SCHEMA_VERSION
    status: ProcessingStatus = ProcessingStatus.FAILED
    review_required: bool = True
    review_reasons: list[str] = Field(default_factory=list)
    document: DocumentMeta | None = None
    document_type: DocumentTypeInfo = Field(default_factory=DocumentTypeInfo)
    fields: dict[str, FieldResult] = Field(default_factory=dict)
    tables: list[TableResult] = Field(default_factory=list)
    pages: list[PageSummary] = Field(default_factory=list)
    validation: ValidationReport = Field(default_factory=ValidationReport)
    processing: ProcessingInfo = Field(default_factory=ProcessingInfo)
    error: dict[str, str] | None = None


def result_json_schema() -> dict:
    """JSON Schema of the public result, for consumers and contract tests."""
    return ParseResult.model_json_schema()
