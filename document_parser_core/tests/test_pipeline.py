"""End-to-end tests through the public API on synthetic documents."""

import json
import os
from pathlib import Path

import pymupdf
import pytest

from document_parser_core import parse_document
from document_parser_core.schemas import ParseResult
from document_parser_core.tests.conftest import COORD_ROWS
from document_parser_core.tests.fixtures import TextLayerOCR, make_png, make_table_pdf, make_text_pdf

NO_PROVIDERS = {"providers": {"ocr": None, "vlm": None}}


def _roundtrip(result: dict) -> dict:
    """Result must be plain JSON and re-validate against the schema."""
    text = json.dumps(result, ensure_ascii=False)
    ParseResult.model_validate(json.loads(text))
    return json.loads(text)


# ---------------------------------------------------------------- text PDFs
def test_text_pdf_english_fields_with_evidence(permit_en_pdf):
    r = _roundtrip(parse_document(str(permit_en_pdf), "building_permit", NO_PROVIDERS))
    f = r["fields"]
    assert f["permit_number"]["value"] == "BP-2024/0153"
    assert f["plot_number"]["value"] == "412"
    assert f["area"]["value"] == {"value": 1250.5, "unit": "m2"}
    assert f["issue_date"]["value"] == "2024-03-05"
    assert f["issue_date"]["validation_status"] == "warning"  # 05/03 is ambiguous
    pn = f["permit_number"]
    assert pn["page"] == 1 and len(pn["bbox"]) == 4 and pn["provider"] == "direct_text"
    assert pn["confidence_kind"] == "native_text_layer" and pn["evidence"][0]["raw_text"] == "BP-2024/0153"
    assert r["document"]["has_text_layer"] is True
    assert r["document"]["format"] == "pdf" and len(r["document"]["sha256"]) == 64


def test_arabic_only_pdf(permit_ar_pdf):
    r = _roundtrip(parse_document(str(permit_ar_pdf), None, NO_PROVIDERS))
    assert r["document_type"]["detected"] == "building_permit"
    f = r["fields"]
    assert f["permit_number"]["value"] == "2024/153"
    assert f["owner_name"]["value"] == "محمد أحمد الخطيب"
    assert f["basin"]["value"] == "الجبيهة الشمالي"
    assert f["issue_date"]["value"] == "2024-03-05"
    assert f["area"]["value"] == {"value": 1250.5, "unit": "m2"}  # from Arabic-Indic digits
    assert r["status"] == "success"


def test_mixed_language_multipage(permit_mixed_pdf):
    r = _roundtrip(parse_document(str(permit_mixed_pdf), None, NO_PROVIDERS))
    assert r["document"]["page_count"] == 2 and [p["page_number"] for p in r["pages"]] == [1, 2]
    f = r["fields"]
    assert f["permit_number"]["value"] == "BP-2024/0153"
    assert f["owner_name"]["value"] == "محمد أحمد الخطيب"
    # plot number appears on both pages with the same value -> not a conflict
    assert f["plot_number"]["status"] == "extracted"
    assert {e["page"] for e in f["plot_number"]["evidence"]} == {1, 2}


def test_conflicting_occurrences_are_not_resolved_by_guessing(tmp_path, font):
    pdf = make_text_pdf(tmp_path / "c.pdf", [["Permit No: 2024/0153", "رقم الرخصة: 2024/0135"]], font)
    r = parse_document(str(pdf), "building_permit", NO_PROVIDERS)
    f = r["fields"]["permit_number"]
    assert f["status"] == "conflict" and f["value"] is None and len(f["evidence"]) == 2
    assert r["status"] == "review_required"


def test_explicitly_absent_value(tmp_path):
    pdf = make_text_pdf(tmp_path / "a.pdf", [["Building Permit", "Permit No: 12/2024", "Basin: N/A"]])
    r = parse_document(str(pdf), "building_permit", NO_PROVIDERS)
    assert r["fields"]["basin"]["status"] == "explicitly_absent"
    assert r["fields"]["basin"]["raw_text"] == "N/A"
    assert r["fields"]["owner_name"]["status"] == "missing"


def test_custom_field_spec(tmp_path):
    pdf = make_text_pdf(tmp_path / "x.pdf", [["Survey Report", "Surveyor Licence: S-778"]])
    r = parse_document(str(pdf), "generic", {
        **NO_PROVIDERS,
        "fields": [{"name": "surveyor_licence", "type": "identifier", "required": True, "labels": ["surveyor licence"]}],
    })
    assert r["fields"]["surveyor_licence"]["value"] == "S-778"
    assert r["status"] == "success"


# ---------------------------------------------------------------- tables
def test_coordinate_table_extraction(coord_pdf):
    r = _roundtrip(parse_document(str(coord_pdf), "coordinate_table", NO_PROVIDERS))
    tables = [t for t in r["tables"] if t["kind"] == "coordinate_table"]
    assert len(tables) == 1
    t = tables[0]
    assert t["columns"] == ["point_id", "E", "N", "Z"] and t["validation_status"] == "valid"
    assert [row["values"]["point_id"] for row in t["rows"]] == ["1", "2", "3", "4"]
    assert t["rows"][1]["values"]["N"] == pytest.approx(1102360.45)
    # CRS is never inferred from coordinate magnitudes
    assert r["fields"]["coordinate_system"]["status"] == "missing"
    assert r["fields"]["coordinate_system"]["value"] is None


def test_coordinate_table_with_bad_cell_is_flagged(tmp_path):
    rows = [list(r) for r in COORD_ROWS]
    rows[2][2] = "11O2330.90"  # letter O: typical OCR confusion
    pdf = make_table_pdf(tmp_path / "bad.pdf", ["Point", "E", "N", "Z"], rows)
    r = parse_document(str(pdf), "coordinate_table", NO_PROVIDERS)
    assert r["status"] == "review_required"
    assert any("table" in x for x in r["review_reasons"])


def test_table_without_header_needs_review(tmp_path):
    pdf = make_table_pdf(tmp_path / "nh.pdf", ["", "", "", ""], COORD_ROWS, title="")
    r = parse_document(str(pdf), "coordinate_table", NO_PROVIDERS)
    t = [t for t in r["tables"] if t["kind"] == "coordinate_table"][0]
    assert t["header_raw"] is None and t["requires_review"]


# ---------------------------------------------------------------- scans & images
def test_scanned_pdf_without_ocr_requires_review(scanned_permit_pdf):
    r = _roundtrip(parse_document(str(scanned_permit_pdf), "building_permit", NO_PROVIDERS))
    assert r["document"]["has_text_layer"] is False
    assert r["pages"][0]["text_source"] == "none"
    assert r["status"] == "review_required"
    assert r["fields"]["permit_number"]["status"] == "missing"


def test_png_input_with_ocr(tmp_path, permit_en_pdf):
    png = make_png(permit_en_pdf, tmp_path / "permit.png", dpi=72)  # 1 px == 1 pt
    r = _roundtrip(parse_document(str(png), "building_permit", {"providers": {"ocr": TextLayerOCR(permit_en_pdf), "vlm": None}}))
    assert r["document"]["format"] == "png"
    assert r["pages"][0]["coord_space"] == "image_pixels"
    assert r["fields"]["permit_number"]["value"] == "BP-2024/0153"


def test_jpeg_input_detected(tmp_path, permit_en_pdf):
    png = make_png(permit_en_pdf, tmp_path / "p.png")
    from PIL import Image
    jpg = tmp_path / "p.jpeg"
    Image.open(png).save(jpg, quality=85)
    r = parse_document(str(jpg), None, NO_PROVIDERS)
    assert r["document"]["format"] == "jpeg" and r["status"] == "review_required"


# ---------------------------------------------------------------- bad inputs
def _failed(r, code):
    assert r["status"] == "failed" and r["review_required"] is True
    assert r["error"]["code"] == code, r["error"]
    return r


def test_missing_file(tmp_path):
    _failed(parse_document(str(tmp_path / "nope.pdf")), "input_file_error")


def test_empty_file(tmp_path):
    p = tmp_path / "empty.pdf"
    p.write_bytes(b"")
    _failed(parse_document(str(p)), "input_file_error")


def test_corrupted_pdf(tmp_path, permit_en_pdf):
    p = tmp_path / "corrupt.pdf"
    p.write_bytes(b"%PDF-1.7\n" + os.urandom(2000))
    r = parse_document(str(p), None, NO_PROVIDERS)
    assert r["status"] in ("failed", "review_required")  # never "success"
    assert r["status"] != "success"


def test_mislabelled_file(tmp_path):
    p = tmp_path / "fake.pdf"
    p.write_text("hello, I am not a PDF")
    r = _failed(parse_document(str(p)), "input_file_error")
    assert "not a valid PDF" in r["error"]["message"]


def test_truncated_png(tmp_path, permit_en_pdf):
    png = make_png(permit_en_pdf, tmp_path / "p.png")
    data = Path(png).read_bytes()
    bad = tmp_path / "trunc.png"
    bad.write_bytes(data[: len(data) // 3])
    _failed(parse_document(str(bad)), "input_file_error")


def test_unsupported_type(tmp_path):
    p = tmp_path / "plan.dwg"
    p.write_bytes(b"AC1032" + b"\0" * 100)
    r = _failed(parse_document(str(p)), "input_file_error")
    assert "DWG" in r["error"]["message"]


def test_encrypted_pdf(tmp_path):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    p = tmp_path / "enc.pdf"
    doc.save(p, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    _failed(parse_document(str(p)), "input_file_error")


def test_invalid_option_is_reported(permit_en_pdf):
    r = parse_document(str(permit_en_pdf), None, {"no_such_option": 1})
    assert r["status"] == "failed" and "Unknown option" in r["error"]["message"]


def test_unknown_document_type(permit_en_pdf):
    r = parse_document(str(permit_en_pdf), "spaceship_manual", NO_PROVIDERS)
    assert r["status"] == "failed" and "Unknown document_type" in r["error"]["message"]


# ---------------------------------------------------------------- diagnostics
def test_processing_metadata(permit_en_pdf):
    r = parse_document(str(permit_en_pdf), None, NO_PROVIDERS)
    p = r["processing"]
    assert p["duration_ms"] > 0 and p["stages"] and p["providers_used"] == ["direct_text"]
    assert p["config"]["render_dpi"] == 200 and p["offline_mode"] is True
