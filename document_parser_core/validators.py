"""Provider-independent value parsing and validation.

Deliberately NOT done here: inferring a coordinate reference system, legal setbacks,
zoning or any regulatory requirement. Domain rules belong in an integration layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from . import text_utils as tu


@dataclass
class Parsed:
    value: Any = None
    ok: bool = False
    messages: list[str] = field(default_factory=list)  # warnings when ok, errors when not

    @property
    def has_warnings(self) -> bool:
        return self.ok and bool(self.messages)


# A number must not be glued to a preceding letter: the '2' of 'm2' / 'م2' is not a value,
# and in 'الكلية289366' the match must not start in the middle of the digits either.
_NUM_RE = re.compile(r"(?<![A-Za-z\u0621-\u064A0-9])[-+−]?\d[\d.,'\s]*")


def parse_number(raw: str, decimal_separator: str = "auto", strict: bool = False) -> Parsed:
    """Parse a number, handling Arabic-Indic digits and both separator conventions.

    strict=True requires the whole string to be a number (used for table cells, where
    '11O2360.45' must fail rather than yield 11)."""
    s = tu.to_ascii_digits(tu.normalize_text(raw)).strip()
    s = s.replace(tu.ARABIC_DECIMAL_SEP, ".").replace(tu.ARABIC_THOUSANDS_SEP, ",")
    s = s.replace("−", "-").replace(" ", " ")
    m = _NUM_RE.fullmatch(s) if strict else _NUM_RE.search(s)
    if not m:
        return Parsed(messages=[f"not a number: {raw!r}" if strict else f"no number in {raw!r}"])
    num = re.sub(r"[\s']", "", m.group(0)).rstrip(".,")
    sign = -1 if num.startswith("-") else 1
    num = num.lstrip("+-")
    warnings: list[str] = []
    has_dot, has_comma = "." in num, "," in num
    if decimal_separator == ".":
        num = num.replace(",", "")
    elif decimal_separator == ",":
        num = num.replace(".", "").replace(",", ".")
    elif has_dot and has_comma:
        # The right-most separator is the decimal separator.
        if num.rfind(",") > num.rfind("."):
            num = num.replace(".", "").replace(",", ".")
        else:
            num = num.replace(",", "")
    elif has_comma:
        parts = num.split(",")
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
            if len(parts) == 2:
                warnings.append(f"ambiguous separator in {raw!r}: ',' read as thousands separator")
            num = num.replace(",", "")
        else:
            num = num.replace(",", ".")
    elif has_dot and num.count(".") > 1:
        num = num.replace(".", "")  # 1.234.567 thousands groups
        warnings.append(f"multiple '.' in {raw!r}: read as thousands separators")
    try:
        value = float(num) * sign
    except ValueError:
        return Parsed(messages=[f"malformed number {raw!r}"])
    if value.is_integer() and "." not in num:
        value = int(value)
    return Parsed(value=value, ok=True, messages=warnings)


_DATE_PATTERNS = [
    (re.compile(r"(\d{4})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})"), "YMD"),
    (re.compile(r"(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{4})"), "XXY"),
]


def parse_date(raw: str, order: str = "DMY") -> Parsed:
    s = tu.to_ascii_digits(tu.normalize_text(raw))
    for rx, kind in _DATE_PATTERNS:
        m = rx.search(s)
        if not m:
            continue
        a, b, c = (int(g) for g in m.groups())
        warnings = []
        if kind == "YMD":
            y, mo, d = a, b, c
        else:
            y = c
            if order == "DMY":
                d, mo = a, b
            else:
                mo, d = a, b
            if a <= 12 and b <= 12 and a != b:
                warnings.append(f"ambiguous day/month order in {raw!r}; interpreted as {order}")
        try:
            return Parsed(value=date(y, mo, d).isoformat(), ok=True, messages=warnings)
        except ValueError:
            return Parsed(messages=[f"invalid calendar date {raw!r}"])
    return Parsed(messages=[f"no recognisable date in {raw!r} (Hijri and month-name dates are not parsed)"])


# unit token -> (dimension, canonical unit, factor to canonical)
_UNITS: list[tuple[str, str, str, float]] = [
    (r"(?:m2|m²|sq\.?\s*m|sqm|م2|م²|متر\s*مربع|م\s*2)", "area", "m2", 1.0),
    (r"(?:km2|km²|كم2|كم²)", "area", "m2", 1_000_000.0),
    (r"(?:ha|hectares?|هكتار)", "area", "m2", 10_000.0),
    (r"(?:dunums?|donums?|دونم|دنم)", "area", "dunum", 1.0),
    (r"(?:km|كم)", "length", "m", 1000.0),
    (r"(?:cm|سم)", "length", "m", 0.01),
    (r"(?:mm|ملم|مم)", "length", "m", 0.001),
    (r"(?:m|متر|م)", "length", "m", 1.0),
]


def parse_measure(raw: str, dimension: str, decimal_separator: str = "auto", dunum_m2: float | None = None) -> Parsed:
    num = parse_number(raw, decimal_separator)
    if not num.ok:
        return num
    text = tu.to_ascii_digits(tu.normalize_text(raw)).casefold()
    unit = None
    for pattern, dim, canonical, factor in _UNITS:
        if dim != dimension:
            continue
        if re.search(rf"(?<![a-zء-ي]){pattern}(?![a-zء-ي])", text):
            unit = (canonical, factor)
            break
    msgs = list(num.messages)
    if unit is None:
        msgs.append("unit not stated; value kept without unit")
        return Parsed(value={"value": num.value, "unit": None}, ok=True, messages=msgs)
    canonical, factor = unit
    if canonical == "dunum":
        if dunum_m2 is None:
            msgs.append("dunum not converted (set dunum_m2 to convert; the dunum varies by jurisdiction)")
            return Parsed(value={"value": num.value, "unit": "dunum"}, ok=True, messages=msgs)
        return Parsed(value={"value": round(num.value * dunum_m2, 6), "unit": "m2", "original_unit": "dunum"}, ok=True, messages=msgs)
    return Parsed(value={"value": round(num.value * factor, 6), "unit": canonical}, ok=True, messages=msgs)


def parse_identifier(raw: str) -> Parsed:
    s = tu.normalize_text(raw).strip(" :-–—.,؛")
    if not s:
        return Parsed(messages=["empty identifier"])
    if not re.search(r"[\w؀-ۿ]", s):
        return Parsed(messages=[f"identifier has no alphanumerics: {raw!r}"])
    return Parsed(value=tu.to_ascii_digits(s), ok=True)


def parse_text(raw: str) -> Parsed:
    s = re.sub(r"\s+", " ", tu.normalize_text(raw)).strip(" :-–—")
    return Parsed(value=s, ok=bool(s), messages=[] if s else ["empty text"])


def parse_typed(raw: str, ftype: str, *, date_order="DMY", decimal_separator="auto", dunum_m2=None) -> Parsed:
    if ftype in ("number", "integer", "currency_amount"):
        p = parse_number(raw, decimal_separator)
        if p.ok and ftype == "integer" and not isinstance(p.value, int):
            return Parsed(messages=[f"expected integer, got {p.value}"])
        return p
    if ftype == "date":
        return parse_date(raw, date_order)
    if ftype in ("area", "length"):
        return parse_measure(raw, ftype, decimal_separator, dunum_m2)
    if ftype == "identifier":
        return parse_identifier(raw)
    return parse_text(raw)


def values_conflict(a: Any, b: Any, rel_tol: float = 1e-6) -> bool:
    """True if two normalised values disagree."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.get("unit") and b.get("unit") and a.get("unit") != b.get("unit"):
            return True
        return values_conflict(a.get("value"), b.get("value"), rel_tol)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) > rel_tol * max(1.0, abs(a), abs(b))
    if isinstance(a, str) and isinstance(b, str):
        return tu.normalize_for_match(a).replace(" ", "") != tu.normalize_for_match(b).replace(" ", "")
    return a != b


def validate_coordinate_rows(rows: list[list[str]], value_columns: int, decimal_separator="auto") -> tuple[list[list[float | None]], list[list[str]], list[str]]:
    """Check column counts and numeric validity of coordinate rows.

    Returns (parsed values per row, per-row issues, table-level issues). No CRS,
    datum or plausibility-by-region checks are made.
    """
    parsed_rows, row_issues, table_issues = [], [], []
    seen_ids: dict[str, int] = {}
    for i, cells in enumerate(rows):
        issues = []
        if len(cells) != value_columns + 1:
            issues.append(f"expected {value_columns + 1} cells, found {len(cells)}")
        point_id = cells[0] if cells else ""
        if point_id in seen_ids:
            issues.append(f"duplicate point id {point_id!r} (also row {seen_ids[point_id] + 1})")
        seen_ids.setdefault(point_id, i)
        values: list[float | None] = []
        for cell in cells[1:]:
            p = parse_number(cell, decimal_separator, strict=True)
            if not p.ok:
                issues.append(f"non-numeric value {cell!r}")
                values.append(None)
            else:
                if p.messages:
                    issues.extend(p.messages)
                values.append(float(p.value))
        parsed_rows.append(values)
        row_issues.append(issues)
    # Decimal-precision consistency per column (a cheap OCR-error signal).
    for col in range(value_columns):
        decimals = set()
        for cells in rows:
            if col + 1 < len(cells):
                d = tu.to_ascii_digits(cells[col + 1])
                decimals.add(len(re.split(r"[.,٫]", d)[1]) if re.search(r"[.,٫]\d+$", d) else 0)
        if len(decimals) > 1:
            table_issues.append(f"column {col + 1}: inconsistent decimal places {sorted(decimals)}")
    return parsed_rows, row_issues, table_issues
