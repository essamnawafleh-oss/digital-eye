import json

import pytest
from pydantic import ValidationError

from document_parser_core import SCHEMA_VERSION, result_json_schema
from document_parser_core import validators as v
from document_parser_core.core.vlm_handler import build_output_schema, parse_vlm_output
from document_parser_core.document_types import FieldSpec
from document_parser_core.exceptions import ConfigError, MalformedModelOutputError
from document_parser_core.schemas import FieldResult, FieldStatus, ParseResult


# ---------------------------------------------------------------- schema
def test_result_schema_is_json_and_versioned():
    schema = result_json_schema()
    json.dumps(schema)
    assert "fields" in schema["properties"]
    assert ParseResult().schema_version == SCHEMA_VERSION == "1.0"


def test_field_states_are_distinct_and_serialise():
    states = {s.value for s in FieldStatus}
    assert states == {"extracted", "missing", "unreadable", "explicitly_absent", "conflict"}
    dumped = FieldResult(name="x", status=FieldStatus.EXPLICITLY_ABSENT).model_dump(mode="json")
    assert dumped["status"] == "explicitly_absent" and dumped["value"] is None


def test_schema_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        FieldResult(name="x", surprise=1)


def test_field_spec_validation():
    with pytest.raises(ConfigError):
        FieldSpec("bad name", "text")
    with pytest.raises(ConfigError):
        FieldSpec("ok", "colour")


# ---------------------------------------------------------------- numbers
@pytest.mark.parametrize("raw,expected", [
    ("1,250.50", 1250.5),
    ("1.250,50", 1250.5),
    ("١٢٥٠٫٥", 1250.5),
    ("١٬٢٥٠٫٥", 1250.5),
    ("۱۲۵۰", 1250),
    ("245100.25", 245100.25),
    ("-12.5", -12.5),
    ("12,5", 12.5),
    ("1 250 000", 1250000),
    ("٣٦٠٨،٠٠", 3608.0),
    ("٦٤٦٧،٥", 6467.5),
])
def test_parse_number(raw, expected):
    p = v.parse_number(raw)
    assert p.ok and p.value == pytest.approx(expected)


def test_parse_number_ambiguous_comma_warns():
    p = v.parse_number("1,250")
    assert p.ok and p.value == 1250 and p.messages


def test_parse_number_explicit_separator():
    assert v.parse_number("1,250", decimal_separator=",").value == pytest.approx(1.25)


def test_parse_number_rejects_text():
    assert not v.parse_number("abc").ok


# ---------------------------------------------------------------- dates & units
def test_parse_date_formats():
    assert v.parse_date("2024-03-05").value == "2024-03-05"
    assert v.parse_date("٠٥/٠٣/٢٠٢٤").value == "2024-03-05"
    p = v.parse_date("05/03/2024")
    assert p.value == "2024-03-05" and p.messages  # ambiguous day/month warned
    assert v.parse_date("05/03/2024", order="MDY").value == "2024-05-03"
    assert v.parse_date("25/03/2024").messages == []
    assert not v.parse_date("31/02/2024").ok
    assert not v.parse_date("March 5th").ok


def test_parse_measure_units():
    assert v.parse_measure("1,250.5 m2", "area").value == {"value": 1250.5, "unit": "m2"}
    assert v.parse_measure("١٢٥٠٫٥ م2", "area").value == {"value": 1250.5, "unit": "m2"}
    assert v.parse_measure("2 ha", "area").value == {"value": 20000.0, "unit": "m2"}
    assert v.parse_measure("12.5 cm", "length").value == {"value": 0.125, "unit": "m"}


def test_dunum_not_converted_without_explicit_factor():
    p = v.parse_measure("3 دونم", "area")
    assert p.value == {"value": 3, "unit": "dunum"} and any("dunum" in m for m in p.messages)
    p = v.parse_measure("3 dunum", "area", dunum_m2=1000.0)
    assert p.value["value"] == 3000.0 and p.value["unit"] == "m2"


def test_missing_unit_is_reported_not_assumed():
    p = v.parse_measure("1250", "area")
    assert p.value == {"value": 1250, "unit": None}
    assert any("unit not stated" in m for m in p.messages)


def test_values_conflict():
    assert not v.values_conflict({"value": 1250.5, "unit": "m2"}, {"value": 1250.5, "unit": "m2"})
    assert v.values_conflict({"value": 1250.5, "unit": "m2"}, {"value": 1205.5, "unit": "m2"})
    assert not v.values_conflict("BP-2024/0153", "bp-2024/0153")
    assert v.values_conflict("2024/153", "2024/0153")


# ---------------------------------------------------------------- coordinate rows
def test_coordinate_row_validation():
    rows = [["1", "245100.25", "1102345.10"], ["2", "245130.6", "11O2360.45"], ["2", "245150.05"]]
    parsed, row_issues, table_issues = v.validate_coordinate_rows(rows, 2)
    assert parsed[0] == [245100.25, 1102345.10]
    assert any("non-numeric" in i for i in row_issues[1])  # letter O instead of zero
    assert any("expected 3 cells" in i for i in row_issues[2])
    assert any("duplicate point id" in i for i in row_issues[2])
    assert any("inconsistent decimal places" in i for i in table_issues)


# ---------------------------------------------------------------- VLM output contract
NAMES = ["permit_number", "area"]


def _ok(**over):
    data = {"fields": {
        "permit_number": {"status": "found", "value": "2024/153", "evidence_text": "رقم الرخصة: 2024/153"},
        "area": {"status": "not_found", "value": None, "evidence_text": None},
    }}
    data["fields"].update(over)
    return json.dumps(data, ensure_ascii=False)


def test_vlm_output_valid():
    ans = parse_vlm_output(_ok(), NAMES)
    assert ans["permit_number"].value == "2024/153"
    assert ans["area"].status == "not_found" and ans["area"].value is None


@pytest.mark.parametrize("raw", [
    "not json",
    '{"fields": []}',
    _ok(area={"status": "maybe", "value": None, "evidence_text": None}),
    _ok(area={"status": "found", "value": None, "evidence_text": None}),
    _ok(area={"status": "found", "value": {"x": 1}, "evidence_text": None}),
    json.dumps({"fields": {"permit_number": {"status": "found", "value": "1", "evidence_text": None}}}),
    _ok(extra={"status": "found", "value": "1", "evidence_text": None}),
])
def test_vlm_output_malformed(raw):
    with pytest.raises(MalformedModelOutputError):
        parse_vlm_output(raw, NAMES)


def test_vlm_value_dropped_when_not_found():
    raw = _ok(area={"status": "not_found", "value": "1250", "evidence_text": None})
    assert parse_vlm_output(raw, NAMES)["area"].value is None


def test_vlm_schema_converts_to_llama_grammar():
    llama_grammar = pytest.importorskip("llama_cpp.llama_grammar")
    gbnf = llama_grammar.json_schema_to_gbnf(json.dumps(build_output_schema(NAMES)))
    assert "permit_number" in gbnf and "not_found" in gbnf
