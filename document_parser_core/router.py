"""Three-level routing with bounded escalation.

Level 1 (fast):    native PDF text; OCR only for pages whose text layer is unreliable.
Level 2 (layout):  table finder on native PDFs; OCR cross-check of native pages that
                   miss required fields (detects incomplete/incorrect text layers).
                   Surya layout is deferred (recorded as unavailable).
Level 3 (vision):  local VLM, only for problem fields, only on a bounded set of pages.

Each level runs at most once and VLM retries are capped, so escalation cannot loop.
Anything still uncertain ends in `review_required` instead of a guessed value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import ParserConfig
from .core.base_handler import OCRProvider, VLMProvider
from .core.direct_text_handler import assess_quality, extract_page
from .document_types import TEMPLATES, FieldSpec, detect_document_type
from .exceptions import DocumentParserError
from .extraction import Candidate, extract_coordinate_tables, extract_fields
from .pdf_handler import LoadedDocument
from .schemas import (
    ConfidenceKind, Evidence, FieldResult, FieldStatus, PageContent, TableResult,
    TableRow, ValidationIssue, ValidationStatus,
)
from .telemetry import Telemetry
from . import text_utils as tu
from . import validators as v


@dataclass
class _Cand:
    """A candidate plus the provider/level that produced it."""

    c: Candidate
    provider: str
    level: int
    is_ocr: bool


@dataclass
class RouteState:
    pages: dict[int, list[PageContent]] = field(default_factory=dict)  # page -> contents by provider
    page_level: dict[int, int] = field(default_factory=dict)
    candidates: dict[str, list[_Cand]] = field(default_factory=dict)
    vlm_evidence: dict[str, list[Evidence]] = field(default_factory=dict)
    vlm_results: dict[str, FieldResult] = field(default_factory=dict)
    tables: list[TableResult] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)
    providers_used: set[str] = field(default_factory=set)
    max_level: int = 1


class Router:
    def __init__(self, config: ParserConfig, ocr: OCRProvider | None, vlm: VLMProvider | None, telemetry: Telemetry):
        self.cfg = config
        self.ocr = ocr
        self.vlm = vlm
        self.tm = telemetry
        self._ocr_ready: bool | None = None
        self._vlm_ready: bool | None = None
        self._images: dict[int, tuple] = {}

    # ------------------------------------------------------------ readiness
    def _provider_ready(self, provider, level: int, kind: str) -> bool:
        if provider is None:
            self.tm.record(level, f"{kind}_check", "unavailable", detail=f"no {kind} provider configured")
            return False
        try:
            provider.check_ready()
        except DocumentParserError as exc:
            self.tm.record(level, f"{kind}_check", "unavailable", provider.name, detail=str(exc))
            return False
        return True

    def ocr_ready(self) -> bool:
        if self._ocr_ready is None:
            self._ocr_ready = self._provider_ready(self.ocr, 1, "ocr")
        return self._ocr_ready

    def vlm_ready(self) -> bool:
        if self._vlm_ready is None:
            self._vlm_ready = self._provider_ready(self.vlm, 3, "vlm")
        return self._vlm_ready

    def _render(self, doc: LoadedDocument, index: int):
        if index not in self._images:
            self._images[index] = doc.render(index, self.cfg.render_dpi)
        return self._images[index]

    # ------------------------------------------------------------ OCR
    def _ocr_page(self, doc: LoadedDocument, index: int, level: int, reason: str) -> PageContent | None:
        page_no = index + 1
        try:
            with self.tm.stage(level, "ocr", self.ocr.name, [page_no]) as rec:
                rec.detail = reason
                image, scale = self._render(doc, index)
                lines = self.ocr.recognize(image, page_no, scale)
        except DocumentParserError:
            return None
        geom = doc.geometry(index)
        content = PageContent(
            page_number=page_no, width=geom.width, height=geom.height, coord_space=geom.coord_space,
            text_source=f"ocr:{self.ocr.name}", lines=lines,
            quality=assess_quality(lines, self.cfg.routing, is_ocr=True),
        )
        self.state.providers_used.add(self.ocr.name)
        return content

    # ------------------------------------------------------------ main entry
    def run(self, doc: LoadedDocument, requested_type: str | None, custom_fields: list[dict] | None):
        from .document_types import resolve_fields

        self.state = st = RouteState()
        r = self.cfg.routing

        # ---- Level 1: native text, OCR for unreliable pages -----------------
        for index in range(doc.page_count):
            page_no = index + 1
            st.page_level[page_no] = 1
            native = None
            if doc.format == "pdf":
                with self.tm.stage(1, "native_text", "direct_text", [page_no]):
                    native = extract_page(doc, index, r)
                st.providers_used.add("direct_text")
            if native is not None and native.quality.is_reliable:
                st.pages[page_no] = [native]
                continue
            reason = "image input" if native is None else "unreliable text layer: " + ", ".join(native.quality.reasons)
            if self.ocr_ready():
                ocr_content = self._ocr_page(doc, index, 1, reason)
                if ocr_content is not None:
                    st.pages[page_no] = [ocr_content] + ([native] if native and native.lines else [])
                    if not ocr_content.quality.is_reliable:
                        st.review_reasons.append(
                            f"page {page_no}: OCR result is low quality ({', '.join(ocr_content.quality.reasons)})")
                    continue
                st.review_reasons.append(f"page {page_no}: OCR failed ({reason})")
            else:
                st.review_reasons.append(f"page {page_no}: {reason}; no OCR provider available")
            if native is None:
                geom = doc.geometry(index)
                native = PageContent(page_number=page_no, width=geom.width, height=geom.height,
                                     coord_space=geom.coord_space, text_source="none")
                native.quality.reasons.append("no_text_source")
            st.pages[page_no] = [native]

        # ---- Document type ----------------------------------------------------
        all_text = "\n".join(p.text for contents in st.pages.values() for p in contents[:1])
        detected, score = detect_document_type(all_text)
        used = requested_type or detected or "generic"
        specs = resolve_fields(used, custom_fields)
        template = TEMPLATES[used]
        want_tables = self.cfg.extract_tables if self.cfg.extract_tables is not None else template.expects_coordinate_table

        # ---- Level 1 extraction ---------------------------------------------
        for contents in st.pages.values():
            for content in contents:
                self._extract_from(content, specs, 1, want_tables)

        # ---- Level 2 ----------------------------------------------------------
        if r.max_level >= 2:
            problems = self._problem_fields(specs)
            required_problems = [s for s in specs if s.name in problems and s.required]
            table_problem = want_tables and (not st.tables or any(t.issues or t.requires_review for t in st.tables))
            if required_problems or table_problem:
                st.max_level = 2
                self.tm.record(2, "surya_layout", "unavailable", "surya_layout",
                               detail="deferred in this version (see KNOWN_LIMITATIONS)")
                if table_problem and doc.format == "pdf":
                    self._level2_tables(doc)
                if required_problems and self.ocr_ready():
                    for page_no, contents in st.pages.items():
                        if contents[0].text_source == "direct_text" and len(contents) == 1:
                            ocr_content = self._ocr_page(doc, page_no - 1, 2, "cross-check: required fields missing")
                            if ocr_content is not None:
                                contents.append(ocr_content)
                                st.page_level[page_no] = 2
                                self._extract_from(ocr_content, specs, 2, want_tables)

        # ---- Level 3 ----------------------------------------------------------
        if r.max_level >= 3:
            problems = self._problem_fields(specs)
            targets = [s for s in specs if s.name in problems and (s.required or self.cfg.vlm_for_optional_fields)]
            if targets and self.vlm_ready():
                st.max_level = 3
                self._level3_vlm(doc, targets, problems, used)
            elif targets:
                st.review_reasons.append("vision-assisted level needed but no VLM available: " + ", ".join(s.name for s in targets))

        fields = {s.name: self._resolve_field(s) for s in specs}
        return fields, st, {"requested": requested_type, "detected": detected, "detection_score": score, "used": used}

    # ------------------------------------------------------------ helpers
    def _extract_from(self, content: PageContent, specs: list[FieldSpec], level: int, want_tables: bool):
        st = self.state
        provider = content.text_source
        is_ocr = provider.startswith("ocr:")
        if not content.lines:
            return
        if specs:
            for cand in extract_fields(content, specs, min_conf=self.cfg.routing.min_ocr_confidence,
                                       provider=provider, date_order=self.cfg.date_order,
                                       decimal_separator=self.cfg.decimal_separator, dunum_m2=self.cfg.dunum_m2,
                                       review_conf=self.cfg.routing.review_ocr_confidence):
                st.candidates.setdefault(cand.name, []).append(_Cand(cand, provider, level, is_ocr))
        if want_tables:
            for t in extract_coordinate_tables(content, provider, self.cfg.decimal_separator):
                t.table_id = f"{t.table_id}_{'ocr' if is_ocr else 'native'}"
                if is_ocr and any((row.confidence or 1) < self.cfg.routing.min_ocr_confidence for row in t.rows):
                    t.requires_review = True
                    t.issues.append("rows with OCR scores below threshold")
                st.tables.append(t)

    def _problem_fields(self, specs: list[FieldSpec]) -> set[str]:
        problems = set()
        for s in specs:
            res = self._resolve_field(s)
            if res.status in (FieldStatus.MISSING, FieldStatus.UNREADABLE, FieldStatus.CONFLICT) or res.requires_review:
                problems.add(s.name)
        return problems

    def _level2_tables(self, doc: LoadedDocument):
        st = self.state
        for page_no in st.pages:
            with self.tm.stage(2, "pymupdf_find_tables", "pymupdf", [page_no]) as rec:
                found = doc.find_tables(page_no - 1)
                rec.detail = f"{len(found)} table(s)"
            for k, (bbox, rows) in enumerate(found):
                clean = [[(c or "").strip() for c in row] for row in rows if any(c for c in row)]
                if len(clean) < 2:
                    continue
                st.tables.append(TableResult(
                    table_id=f"p{page_no}_find{k + 1}", kind="generic", page=page_no, bbox=list(bbox),
                    provider="pymupdf", method="find_tables", columns=clean[0],
                    header_raw=clean[0], rows=[TableRow(cells=r) for r in clean[1:]],
                ))
        if any(t.provider == "pymupdf" for t in st.tables):
            st.providers_used.add("pymupdf_tables")
        self._compare_tables()

    def _compare_tables(self):
        """Flag disagreement between geometric coordinate tables and find_tables output."""
        st = self.state
        geo = [t for t in st.tables if t.method == "row_geometry"]
        fnd = [t for t in st.tables if t.method == "find_tables"]
        for g in geo:
            for f in (t for t in fnd if t.page == g.page):
                f_nums = sorted(_numbers_in(" ".join(c for r in f.rows for c in r.cells)))
                g_nums = sorted(_numbers_in(" ".join(c for r in g.rows for c in r.cells[1:])))
                if g_nums and f_nums and not set(g_nums) <= set(f_nums):
                    msg = f"table {g.table_id}: values disagree with {f.table_id}"
                    g.issues.append(msg)
                    g.requires_review = True
                    st.issues.append(ValidationIssue(code="table_provider_conflict", severity="warning", message=msg, page=g.page))

    def _level3_vlm(self, doc: LoadedDocument, targets: list[FieldSpec], problems: set[str], doc_type: str):
        st = self.state
        r = self.cfg.routing
        # Pages: where candidates were seen first, then remaining pages in order.
        page_order: list[int] = []
        for s in targets:
            for c in st.candidates.get(s.name, []):
                if c.c.page not in page_order:
                    page_order.append(c.c.page)
        page_order += [p for p in st.pages if p not in page_order]
        remaining = {s.name: s for s in targets}
        for page_no in page_order[: r.max_vlm_pages]:
            if not remaining:
                break
            batch = list(remaining.values())[: r.max_vlm_fields_per_call]
            image, _ = self._render(doc, page_no - 1)
            try:
                with self.tm.stage(3, "vlm_extract", self.vlm.name, [page_no]) as rec:
                    resp = self.vlm.extract_fields(image, page_no, [s.to_dict() for s in batch], doc_type)
                    rec.detail = f"{len(batch)} field(s), attempts={resp.attempts}"
            except DocumentParserError as exc:
                st.review_reasons.append(f"page {page_no}: VLM failed ({exc})")
                continue
            st.providers_used.add(self.vlm.name)
            st.page_level[page_no] = 3
            page_text = _compact(" ".join(p.text for p in st.pages[page_no]))
            for s in batch:
                ans = resp.answers.get(s.name)
                if ans is None:
                    continue
                corroborated = bool(ans.evidence_text) and _compact(ans.evidence_text) in page_text if page_text else False
                if ans.status == "found" and ans.value is not None:
                    corroborated = corroborated or (_compact(ans.value) in page_text if page_text else False)
                ev = Evidence(
                    provider=self.vlm.name, stage="vlm_extract", level=3, method="vlm_json_schema",
                    page=page_no, raw_text=ans.evidence_text, value=ans.value,
                    confidence_kind=ConfidenceKind.VLM_SELF_REPORT if ans.self_reported_confidence is not None else ConfidenceKind.NONE,
                    confidence=ans.self_reported_confidence,
                    note=f"vlm_status={ans.status}; corroborated_by_page_text={corroborated}",
                )
                st.vlm_evidence.setdefault(s.name, []).append(ev)
                if ans.status in ("found", "explicitly_absent", "illegible"):
                    st.vlm_results[s.name] = self._vlm_field(s, ans, ev, corroborated, page_no)
                    remaining.pop(s.name, None)

    def _vlm_field(self, spec: FieldSpec, ans, ev: Evidence, corroborated: bool, page_no: int) -> FieldResult:
        res = FieldResult(name=spec.name, provider=self.vlm.name, page=page_no, raw_text=ans.evidence_text or ans.value)
        if ans.status == "illegible":
            res.status = FieldStatus.UNREADABLE
        elif ans.status == "explicitly_absent":
            res.status = FieldStatus.EXPLICITLY_ABSENT
        else:
            parsed = v.parse_typed(ans.value, spec.type, date_order=self.cfg.date_order,
                                   decimal_separator=self.cfg.decimal_separator, dunum_m2=self.cfg.dunum_m2)
            res.status = FieldStatus.EXTRACTED
            res.value = parsed.value if parsed.ok else None
            _apply_validation(res, parsed)
        if not corroborated:
            res.requires_review = True
            res.review_reasons.append("vlm_answer_not_corroborated_by_page_text")
        res.evidence.append(ev)
        return res

    # ------------------------------------------------------------ field resolution
    def _resolve_field(self, spec: FieldSpec) -> FieldResult:
        st = self.state
        cands = st.candidates.get(spec.name, [])
        evidence = [_to_evidence(c) for c in cands] + st.vlm_evidence.get(spec.name, [])
        res = FieldResult(name=spec.name, evidence=evidence)

        extracted = [c for c in cands if c.c.status == "extracted" and c.c.parsed and c.c.parsed.ok]
        if extracted:
            # Group by normalised value; disagreement => conflict.
            groups: list[list[_Cand]] = []
            for c in extracted:
                for g in groups:
                    if not v.values_conflict(g[0].c.parsed.value, c.c.parsed.value):
                        g.append(c)
                        break
                else:
                    groups.append([c])
            if len(groups) > 1:
                vlm = st.vlm_results.get(spec.name)
                agreeing = [g for g in groups if vlm and vlm.value is not None and not v.values_conflict(g[0].c.parsed.value, vlm.value)]
                if len(agreeing) == 1 and not vlm.requires_review:
                    best = _best(agreeing[0])
                    _fill(res, best)
                    res.review_reasons.append("conflict resolved by corroborated VLM answer")
                    return res
                res.status = FieldStatus.CONFLICT
                res.requires_review = True
                res.review_reasons.append("providers or occurrences disagree: " + " | ".join(repr(g[0].c.raw_text) for g in groups))
                return res
            best = _best(groups[0])
            _fill(res, best)
            if any(c.c.status == "unreadable" for c in cands):
                res.validation_messages.append("another occurrence was below the OCR score threshold")
            return res

        if spec.name in st.vlm_results:
            vlm = st.vlm_results[spec.name]
            vlm.evidence = evidence
            return vlm

        invalid = [c for c in cands if c.c.status == "extracted"]
        if invalid:
            best = _best(invalid)
            _fill(res, best)
            return res
        absent = [c for c in cands if c.c.status == "explicitly_absent"]
        if absent:
            c = absent[0].c
            res.status, res.raw_text, res.page, res.bbox, res.provider = FieldStatus.EXPLICITLY_ABSENT, c.raw_text, c.page, c.bbox, absent[0].provider
            return res
        unreadable = [c for c in cands if c.c.status == "unreadable"]
        if unreadable:
            c = unreadable[0]
            res.status, res.raw_text, res.page, res.bbox, res.provider = FieldStatus.UNREADABLE, c.c.raw_text, c.c.page, c.c.bbox, c.provider
            res.confidence, res.confidence_kind = c.c.confidence, ConfidenceKind.OCR_SCORE
            res.requires_review = True
            res.review_reasons.append("value below OCR score threshold")
            return res
        res.status = FieldStatus.MISSING
        if spec.required:
            res.requires_review = True
            res.review_reasons.append("required field not found")
        return res


def _best(group: list[_Cand]) -> _Cand:
    # Prefer native text over OCR, then higher score, then earlier page.
    return sorted(group, key=lambda c: (c.is_ocr, -(c.c.confidence or 1.0), c.c.page))[0]


def _fill(res: FieldResult, c: _Cand) -> None:
    cand = c.c
    res.status = FieldStatus.EXTRACTED
    res.raw_text = cand.raw_text
    res.page, res.bbox, res.provider = cand.page, cand.bbox, c.provider
    res.confidence = cand.confidence
    res.confidence_kind = ConfidenceKind.OCR_SCORE if c.is_ocr else ConfidenceKind.NATIVE_TEXT
    parsed = cand.parsed
    res.value = parsed.value if parsed and parsed.ok else None
    if parsed is not None:
        _apply_validation(res, parsed)
    if cand.review_flags:
        res.requires_review = True
        res.review_reasons.extend(cand.review_flags)


def _apply_validation(res: FieldResult, parsed: v.Parsed) -> None:
    if not parsed.ok:
        res.validation_status = ValidationStatus.INVALID
        res.requires_review = True
        res.review_reasons.append("value failed validation")
    elif parsed.messages:
        res.validation_status = ValidationStatus.WARNING
    else:
        res.validation_status = ValidationStatus.VALID
    res.validation_messages.extend(parsed.messages)


def _to_evidence(c: _Cand) -> Evidence:
    cand = c.c
    return Evidence(
        provider=c.provider, stage="ocr" if c.is_ocr else "native_text", level=c.level,
        method="label_value", page=cand.page, bbox=cand.bbox, raw_text=cand.raw_text,
        value=cand.parsed.value if cand.parsed and cand.parsed.ok else None,
        confidence=cand.confidence,
        confidence_kind=ConfidenceKind.OCR_SCORE if c.is_ocr else ConfidenceKind.NATIVE_TEXT,
        note=f"label={cand.label!r}; status={cand.status}" + ("; approximate word box" if cand.approx_bbox else "")
        + ("; value printed before label (mixed-direction line)" if cand.value_before_label else ""),
    )


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", tu.to_ascii_digits(tu.normalize_for_match(text or "")))


def _numbers_in(text: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)?", tu.to_ascii_digits(text))
