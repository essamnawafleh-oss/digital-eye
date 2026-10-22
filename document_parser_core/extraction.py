"""Deterministic (non-ML) extraction from positioned text.

* Words are grouped into visual rows by vertical overlap, ordered right-to-left for
  Arabic-dominant rows, and split into segments at large horizontal gaps.
* Fields: a label must start a segment, or be followed by an explicit ':' separator.
  The value is the rest of the segment, else the next segment in reading order,
  else the first segment of the row below.
* Coordinate tables: runs of >= 2 consecutive rows with the same number of numeric
  cells. Value columns are reported left-to-right as printed; their meaning is taken
  only from an explicit header, never inferred from magnitudes.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from .document_types import FieldSpec
from .pdf_handler import union_bbox
from .schemas import PageContent, TableResult, TableRow, ValidationStatus, Word
from . import text_utils as tu
from . import validators as v

_SEPARATORS = {":", "-", "–", "—", "=", "：", "؛"}
_PUNCT_STRIP = " :：-–—=.,؛()[]#"


@dataclass
class Row:
    words: list[Word]
    rtl: bool
    segments: list[list[Word]] = field(default_factory=list)

    @property
    def bbox(self):
        return union_bbox([w.bbox for w in self.words])


def build_rows(page: PageContent) -> list[Row]:
    words = sorted(page.words, key=lambda w: (w.bbox[1] + w.bbox[3]) / 2)
    if not words:
        return []
    heights = [w.bbox[3] - w.bbox[1] for w in words if w.bbox[3] > w.bbox[1]]
    med_h = statistics.median(heights) if heights else 10.0
    rows: list[list[Word]] = []
    for w in words:
        cy = (w.bbox[1] + w.bbox[3]) / 2
        if rows:
            last = rows[-1]
            top = min(x.bbox[1] for x in last)
            bottom = max(x.bbox[3] for x in last)
            if top - 0.1 * med_h <= cy <= bottom + 0.1 * med_h:
                last.append(w)
                continue
        rows.append([w])
    out = []
    for ws in rows:
        text = " ".join(w.text for w in ws)
        rtl = tu.arabic_char_ratio(text) > 0.5
        ws.sort(key=lambda w: w.bbox[0], reverse=rtl)
        row = Row(ws, rtl)
        row.segments = _segments(ws, rtl, med_h)
        out.append(row)
    return out


def _segments(words: list[Word], rtl: bool, med_h: float) -> list[list[Word]]:
    segs = [[words[0]]]
    for prev, cur in zip(words, words[1:]):
        gap = (prev.bbox[0] - cur.bbox[2]) if rtl else (cur.bbox[0] - prev.bbox[2])
        if gap > 1.8 * med_h:
            segs.append([cur])
        else:
            segs[-1].append(cur)
    return segs


def _norm_word(text: str) -> str:
    return tu.normalize_for_match(text).strip(_PUNCT_STRIP)


@dataclass
class Candidate:
    name: str
    status: str  # extracted | unreadable | explicitly_absent
    raw_text: str
    parsed: v.Parsed | None
    page: int
    bbox: list[float]
    confidence: float | None
    approx_bbox: bool
    label: str


def _match_label(seg: list[Word], start: int, label_tokens: list[str]) -> int | None:
    """Return the index after the label if it matches at `start`."""
    j = start
    for tok in label_tokens:
        while j < len(seg) and _norm_word(seg[j].text) == "":
            j += 1
        if j >= len(seg) or _norm_word(seg[j].text) != tok:
            return None
        j += 1
    return j


def _has_colon_after(seg: list[Word], end: int) -> bool:
    if seg[end - 1].text.rstrip().endswith((":", "：")):
        return True
    return end < len(seg) and seg[end].text.strip() in _SEPARATORS


def _find_labels(seg: list[Word], specs: list[FieldSpec]) -> list[tuple[int, int, FieldSpec, str]]:
    """All (start, end, spec, label) label hits in a segment, longest label first."""
    hits = []
    labels = sorted(((l, s) for s in specs for l in s.labels), key=lambda x: -len(x[0]))
    taken: set[int] = set()
    for label, spec in labels:
        toks = [t for t in (_norm_word(x) for x in label.split()) if t]
        for start in range(len(seg)):
            if start in taken:
                continue
            end = _match_label(seg, start, toks)
            if end is None:
                continue
            leading_only_punct = all(_norm_word(w.text) == "" or re.fullmatch(r"\d+[.)]?", w.text) for w in seg[:start])
            if leading_only_punct or _has_colon_after(seg, end):
                hits.append((start, end, spec, label))
                taken.update(range(start, end))
    return sorted(hits, key=lambda h: h[0])


def extract_fields(page: PageContent, specs: list[FieldSpec], *, min_conf: float, provider: str,
                   date_order="DMY", decimal_separator="auto", dunum_m2=None) -> list[Candidate]:
    rows = build_rows(page)
    out: list[Candidate] = []
    for r_idx, row in enumerate(rows):
        for s_idx, seg in enumerate(row.segments):
            hits = _find_labels(seg, specs)
            for h_idx, (_start, end, spec, label) in enumerate(hits):
                stop = hits[h_idx + 1][0] if h_idx + 1 < len(hits) else len(seg)
                value_words = _strip_separators(seg[end:stop])
                if not value_words and h_idx + 1 == len(hits):
                    if s_idx + 1 < len(row.segments) and not _find_labels(row.segments[s_idx + 1], specs):
                        value_words = row.segments[s_idx + 1]
                    elif r_idx + 1 < len(rows):
                        below = rows[r_idx + 1].segments[0]
                        if not _find_labels(below, specs) and _x_overlap(seg, below):
                            value_words = below
                if not value_words:
                    continue
                cand = _make_candidate(spec, label, value_words, page.page_number, min_conf,
                                       date_order, decimal_separator, dunum_m2)
                out.append(cand)
    return out


def _strip_separators(words: list[Word]) -> list[Word]:
    """Drop separator-only words at the edges; keep internal ones (e.g. '2024 - 03 - 05')."""
    i, j = 0, len(words)
    while i < j and words[i].text.strip() in _SEPARATORS:
        i += 1
    while j > i and words[j - 1].text.strip() in _SEPARATORS:
        j -= 1
    return words[i:j]


def _x_overlap(a: list[Word], b: list[Word]) -> bool:
    ba, bb = union_bbox([w.bbox for w in a]), union_bbox([w.bbox for w in b])
    return min(ba[2], bb[2]) - max(ba[0], bb[0]) > -0.5 * (ba[2] - ba[0])


def _make_candidate(spec, label, words, page, min_conf, date_order, dec, dunum) -> Candidate:
    raw = " ".join(w.text for w in words).strip()
    confs = [w.confidence for w in words if w.confidence is not None]
    conf = min(confs) if confs else None
    bbox = union_bbox([w.bbox for w in words])
    approx = any(w.approx_bbox for w in words)
    if _norm_word(raw) in {_norm_word(m) for m in spec.absent_markers} or raw.strip() in spec.absent_markers:
        return Candidate(spec.name, "explicitly_absent", raw, None, page, bbox, conf, approx, label)
    if conf is not None and conf < min_conf:
        return Candidate(spec.name, "unreadable", raw, None, page, bbox, conf, approx, label)
    parsed = v.parse_typed(raw, spec.type, date_order=date_order, decimal_separator=dec, dunum_m2=dunum)
    return Candidate(spec.name, "extracted", raw, parsed, page, bbox, conf, approx, label)


# ------------------------------------------------------------- coordinate tables
_HEADER_HINTS = ["point", "pt", "no", "id", "station", "e", "n", "x", "y", "z", "easting", "northing",
                 "elevation", "level", "نقطة", "النقطة", "رقم", "شرقي", "شمالي", "الارتفاع", "منسوب"]


def _is_numeric_token(text: str) -> bool:
    t = tu.to_ascii_digits(text).strip()
    return bool(re.fullmatch(r"[-+−]?\d[\d.,٫٬]*", t))


def _is_id_token(text: str) -> bool:
    t = tu.to_ascii_digits(text).strip()
    return bool(re.fullmatch(r"[A-Za-zء-ي]{0,3}[-_]?\d{1,4}[A-Za-z]?", t)) or (not _is_numeric_token(t) and len(t) <= 6)


def _split_row(row: Row) -> tuple[str | None, list[Word]] | None:
    """Return (point_id, value words left-to-right) if the row looks like a coordinate row."""
    ltr = sorted(row.words, key=lambda w: w.bbox[0])
    if len(ltr) < 2:
        return None
    numeric = [w for w in ltr if _is_numeric_token(w.text)]
    others = [w for w in ltr if not _is_numeric_token(w.text)]
    if len(others) > 1:
        return None
    edge = ltr[-1] if row.rtl else ltr[0]
    if others:
        if others[0] is not edge:
            return None
        return others[0].text, numeric
    # All numeric: an id is a short integer at the reading-start edge.
    if len(numeric) >= 3 and re.fullmatch(r"\d{1,4}", tu.to_ascii_digits(edge.text)):
        return edge.text, [w for w in numeric if w is not edge]
    return None, numeric


def _loose_split(row: Row, ncols: int, has_id: bool) -> tuple[str | None, list[Word]] | None:
    ltr = sorted(row.words, key=lambda w: w.bbox[0])
    if len(ltr) != ncols + int(has_id):
        return None
    if sum(_is_numeric_token(w.text) for w in ltr) < max(1, ncols - 1):
        return None
    if not has_id:
        return None, ltr
    edge = ltr[-1] if row.rtl else ltr[0]
    return edge.text, [w for w in ltr if w is not edge]


def extract_coordinate_tables(page: PageContent, provider: str, decimal_separator="auto") -> list[TableResult]:
    rows = build_rows(page)
    tables: list[TableResult] = []
    i = 0
    while i < len(rows):
        split = _split_row(rows[i])
        if not split or len(split[1]) < 2:
            i += 1
            continue
        ncols = len(split[1])
        has_id = split[0] is not None
        block = [(rows[i], split)]
        j = i + 1
        while j < len(rows):
            s = _split_row(rows[j])
            if not s or len(s[1]) != ncols or (s[0] is not None) != has_id:
                # Keep malformed rows of the same shape (e.g. OCR 'O' for '0') so they
                # are flagged by validation instead of silently ending the table.
                s = _loose_split(rows[j], ncols, has_id)
                if s is None:
                    break
            block.append((rows[j], s))
            j += 1
        if len(block) >= 2:
            header_row = rows[i - 1] if i > 0 else None
            tables.append(_build_table(page, block, header_row, provider, len(tables), decimal_separator))
        i = j
    return tables


def _build_table(page, block, header_row, provider, n, dec) -> TableResult:
    first_vals = block[0][1][1]
    col_centers = [(w.bbox[0] + w.bbox[2]) / 2 for w in first_vals]
    has_id = block[0][1][0] is not None
    header_raw = None
    columns = [f"col_{k + 1}" for k in range(len(col_centers))]
    if header_row is not None:
        hdr_tokens = [_norm_word(w.text) for w in header_row.words]
        if any(t in _HEADER_HINTS for t in hdr_tokens):
            header_raw = [w.text for w in sorted(header_row.words, key=lambda w: w.bbox[0])]
            assigned: dict[int, list[str]] = {}
            for w in header_row.words:
                cx = (w.bbox[0] + w.bbox[2]) / 2
                k = min(range(len(col_centers)), key=lambda c: abs(col_centers[c] - cx))
                if abs(col_centers[k] - cx) < 0.6 * (page.width / max(2, len(col_centers) + 1)):
                    assigned.setdefault(k, []).append(w.text)
            for k, names in assigned.items():
                columns[k] = " ".join(names)
    cells = [([s[0]] if has_id else [f"#{r + 1}"]) + [w.text for w in s[1]] for r, (_, s) in enumerate(block)]
    parsed, row_issues, table_issues = v.validate_coordinate_rows(cells, len(col_centers), dec)
    rows_out = []
    for (row, s), vals, issues, raw_cells in zip(block, parsed, row_issues, cells):
        confs = [w.confidence for w in row.words if w.confidence is not None]
        rows_out.append(TableRow(
            cells=raw_cells,
            values={"point_id": s[0], **{columns[k]: vals[k] for k in range(len(vals))}},
            bbox=row.bbox, confidence=min(confs) if confs else None, issues=issues,
        ))
    any_issue = bool(table_issues) or any(row_issues)
    return TableResult(
        table_id=f"p{page.page_number}_t{n + 1}", kind="coordinate_table", page=page.page_number,
        bbox=union_bbox([r.bbox for r, _ in block]), provider=provider, method="row_geometry",
        columns=(["point_id"] if has_id else []) + columns, header_raw=header_raw, rows=rows_out,
        validation_status=ValidationStatus.WARNING if any_issue else ValidationStatus.VALID,
        issues=table_issues,
        requires_review=header_raw is None or any_issue,
    )
