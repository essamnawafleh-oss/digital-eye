"""Generic field templates per document type, and cheap keyword-based type detection.

Templates contain only *labels* that commonly precede values. They encode no
regulation, zoning or legal rule. Callers can override or extend them with
`options={"fields": [...]}`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .exceptions import ConfigError
from . import text_utils as tu

FIELD_TYPES = ("text", "identifier", "number", "integer", "currency_amount", "date", "area", "length")
DEFAULT_ABSENT_MARKERS = ["n/a", "na", "none", "nil", "-", "—", "لا يوجد", "لايوجد", "غير متوفر", "لا ينطبق"]


@dataclass
class FieldSpec:
    name: str
    type: str = "text"
    required: bool = False
    labels: list[str] = field(default_factory=list)
    description: str = ""
    absent_markers: list[str] = field(default_factory=lambda: list(DEFAULT_ABSENT_MARKERS))

    def __post_init__(self):
        if self.type not in FIELD_TYPES:
            raise ConfigError(f"Field '{self.name}': unknown type '{self.type}'. Valid: {FIELD_TYPES}")
        if not self.name.isidentifier():
            raise ConfigError(f"Field name '{self.name}' must be a valid identifier")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DocumentTemplate:
    name: str
    fields: list[FieldSpec]
    keywords: list[str]
    expects_coordinate_table: bool = False
    notes: str = ""


TEMPLATES: dict[str, DocumentTemplate] = {
    "generic": DocumentTemplate("generic", [], []),
    "building_permit": DocumentTemplate(
        "building_permit",
        [
            FieldSpec("permit_number", "identifier", True, ["permit no", "permit number", "licence no", "license no", "رقم الرخصة", "رقم الترخيص", "رقم رخصة البناء"], "Building permit / licence number"),
            FieldSpec("issue_date", "date", False, ["issue date", "date of issue", "تاريخ الإصدار", "تاريخ الاصدار", "تاريخ الرخصة"], "Date the permit was issued"),
            FieldSpec("owner_name", "text", False, ["owner name", "owner", "applicant", "اسم المالك", "المالك", "مقدم الطلب"], "Owner or applicant name"),
            FieldSpec("plot_number", "identifier", False, ["plot no", "plot number", "parcel no", "parcel number", "رقم القطعة"], "Plot / parcel number"),
            FieldSpec("basin", "text", False, ["basin", "الحوض", "اسم الحوض", "رقم الحوض"], "Basin name or number as written"),
            FieldSpec("area", "area", False, ["total area", "plot area", "area", "المساحة", "مساحة القطعة", "المساحة الكلية"], "Area with its stated unit"),
        ],
        ["permit", "building licence", "building license", "رخصة", "ترخيص", "رخصة بناء", "إذن إنشاء"],
    ),
    "invoice": DocumentTemplate(
        "invoice",
        [
            FieldSpec("invoice_number", "identifier", True, ["invoice no", "invoice number", "invoice #", "رقم الفاتورة"], "Invoice number"),
            FieldSpec("invoice_date", "date", False, ["invoice date", "date", "تاريخ الفاتورة", "التاريخ"], "Invoice date"),
            FieldSpec("vendor_name", "text", False, ["vendor", "supplier", "seller", "المورد", "البائع"], "Seller / supplier"),
            FieldSpec("tax_amount", "currency_amount", False, ["vat", "tax", "ضريبة المبيعات", "الضريبة"], "Tax amount as printed"),
            FieldSpec("total_amount", "currency_amount", True, ["grand total", "total amount", "total", "المجموع الكلي", "الإجمالي", "المجموع"], "Total amount as printed"),
            FieldSpec("currency", "text", False, ["currency", "العملة"], "Currency only if explicitly written"),
        ],
        ["invoice", "tax invoice", "فاتورة", "فاتورة ضريبية"],
    ),
    "coordinate_table": DocumentTemplate(
        "coordinate_table",
        [
            # Only reported when the document states it; never inferred from values.
            FieldSpec("coordinate_system", "text", False, ["coordinate system", "datum", "projection", "grid", "نظام الإحداثيات", "نظام الاحداثيات", "المسقط"], "Coordinate system exactly as stated on the document"),
        ],
        ["coordinates", "easting", "northing", "إحداثيات", "احداثيات", "شرقي", "شمالي"],
        expects_coordinate_table=True,
    ),
    "architectural_plan": DocumentTemplate(
        "architectural_plan",
        [
            FieldSpec("drawing_number", "identifier", False, ["drawing no", "dwg no", "sheet no", "رقم المخطط", "رقم اللوحة"], "Drawing / sheet number"),
            FieldSpec("scale", "text", False, ["scale", "مقياس الرسم", "المقياس"], "Scale as written, e.g. 1:100"),
            FieldSpec("project_name", "text", False, ["project", "project name", "اسم المشروع", "المشروع"], "Project name"),
        ],
        ["drawing", "plan", "elevation", "section", "مخطط", "واجهة", "مسقط أفقي"],
        notes="Drawings are a difficult case: title-block text only; geometry is not interpreted.",
    ),
}


def resolve_fields(document_type: str, custom: list[dict] | None) -> list[FieldSpec]:
    if document_type not in TEMPLATES:
        raise ConfigError(f"Unknown document_type '{document_type}'. Known: {sorted(TEMPLATES)}")
    specs = {f.name: f for f in TEMPLATES[document_type].fields}
    for item in custom or []:
        spec = FieldSpec(**item)
        specs[spec.name] = spec
    return list(specs.values())


def detect_document_type(text: str) -> tuple[str | None, float]:
    """Keyword-count heuristic. Returns (type or None, score). Score is a share of
    keyword hits, NOT a probability."""
    norm = tu.normalize_for_match(text)
    scores = {}
    for name, tpl in TEMPLATES.items():
        if not tpl.keywords:
            continue
        scores[name] = sum(norm.count(tu.normalize_for_match(k)) for k in tpl.keywords)
    total = sum(scores.values())
    if total == 0:
        return None, 0.0
    best = max(scores, key=scores.get)
    return best, round(scores[best] / total, 3)
