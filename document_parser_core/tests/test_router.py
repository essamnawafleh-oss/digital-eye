"""Routing / escalation behaviour, driven by fake providers (no models needed)."""

import json

from document_parser_core import parse_document
from document_parser_core.config import RoutingConfig
from document_parser_core.core.direct_text_handler import assess_quality
from document_parser_core.core.paddle_handler import ocr_line
from document_parser_core.tests.fixtures import (
    ScriptedVLM, TextLayerOCR, make_text_pdf, make_visual_order_arabic_pdf,
)

NO_PROVIDERS = {"ocr": None, "vlm": None}


def stages(result, name):
    return [s for s in result["processing"]["stages"] if s["stage"] == name]


# ---------------------------------------------------------------- quality signals
def _q(text, **kw):
    return assess_quality([ocr_line(t, [0, 0, 100, 10], 1, None) for t in text.split("\n")], RoutingConfig(), **kw)


def test_quality_flags_empty_page():
    q = _q("")
    assert not q.is_reliable and "no_text" in q.reasons


def test_short_clean_vector_page_is_reliable():
    q = _q("Notes: none")
    assert q.is_reliable, q.reasons


def test_short_text_on_image_page_is_not_reliable():
    q = _q("Stamp 12", image_coverage=0.9)
    assert not q.is_reliable


def test_vector_dominated_page_is_not_reliable():
    q = _q("Title block A-101 scale 1:100 project tower", drawing_count=3000)
    assert any("vector_dominated" in r for r in q.reasons)


def test_quality_flags_reversed_arabic():
    q = _q("ةصخرلا مقر ةحاسملا ةيلكلا ضوحلا يلامشلا ةهيبجلا")
    assert q.reversed_arabic_score == 1.0 and not q.is_reliable


def test_quality_accepts_logical_arabic():
    q = _q("رقم الرخصة 2024/153 المساحة الكلية 1250 م2 الحوض الجبيهة الشمالي")
    assert q.is_reliable, q.reasons


def test_quality_flags_broken_glyph_map():
    q = _q("رقم اǄرخصة 2024 اǄمساحة اǄكلية 1250 اǄحوض اǄشمالي")
    assert any("broken_glyph_mapping" in r for r in q.reasons)


def test_quality_flags_garbage():
    q = _q(" �� abc def ghi jkl mno pqr")
    assert any("garbage" in r for r in q.reasons)


def test_quality_flags_low_ocr_scores():
    lines = [ocr_line(f"word{i} text more words here", [0, 0, 100, 10], 1, 0.3) for i in range(5)]
    q = assess_quality(lines, RoutingConfig(), is_ocr=True)
    assert q.mean_confidence == 0.3 and any("low_confidence" in r for r in q.reasons)


# ---------------------------------------------------------------- level 1
def test_reliable_native_text_does_not_call_ocr(permit_en_pdf):
    ocr = TextLayerOCR(permit_en_pdf)
    r = parse_document(str(permit_en_pdf), "building_permit", {"providers": {"ocr": ocr, "vlm": None}})
    assert ocr.calls == []
    assert r["fields"]["permit_number"]["status"] == "extracted"
    assert r["processing"]["max_level_reached"] == 1


def test_scanned_page_escalates_to_ocr(scanned_permit_pdf, permit_en_pdf):
    ocr = TextLayerOCR(permit_en_pdf)
    r = parse_document(str(scanned_permit_pdf), "building_permit", {"providers": {"ocr": ocr, "vlm": None}})
    assert ocr.calls == [1]
    assert r["pages"][0]["text_source"] == "ocr:fake_ocr"
    f = r["fields"]["permit_number"]
    assert f["value"] == "BP-2024/0153" and f["confidence_kind"] == "ocr_recognition_score"
    assert f["confidence"] == 0.95


def test_visual_order_arabic_escalates_and_never_fakes_success(tmp_path):
    pdf = make_visual_order_arabic_pdf(tmp_path / "visual.pdf")
    r = parse_document(str(pdf), "building_permit", {"providers": NO_PROVIDERS})
    assert r["pages"][0]["quality"]["is_reliable"] is False
    assert r["status"] == "review_required"
    assert any("no OCR provider" in x for x in r["review_reasons"])


def test_ocr_failure_falls_back_to_review(scanned_permit_pdf, permit_en_pdf):
    ocr = TextLayerOCR(permit_en_pdf, fail=True)
    r = parse_document(str(scanned_permit_pdf), "building_permit", {"providers": {"ocr": ocr, "vlm": None}})
    assert r["status"] == "review_required"
    assert stages(r, "ocr")[0]["outcome"] == "failed"
    assert any("OCR failed" in x for x in r["review_reasons"])
    assert r["fields"]["permit_number"]["status"] == "missing"


def test_low_ocr_score_value_is_unreadable_not_guessed(scanned_permit_pdf, permit_en_pdf):
    ocr = TextLayerOCR(permit_en_pdf, low_conf_tokens={"BP-2024/0153"})
    r = parse_document(str(scanned_permit_pdf), "building_permit", {"providers": {"ocr": ocr, "vlm": None}})
    f = r["fields"]["permit_number"]
    assert f["status"] == "unreadable" and f["value"] is None
    assert f["raw_text"] == "BP-2024/0153" and f["requires_review"]


# ---------------------------------------------------------------- level 2
def test_level2_ocr_cross_check_detects_provider_conflict(tmp_path):
    # Native text layer lacks the permit number; OCR of the rendered page sees one.
    native = make_text_pdf(tmp_path / "native.pdf", [["Building Permit", "Owner: A. Khatib", "Plot No: 412"]])
    ref = make_text_pdf(tmp_path / "ref.pdf", [["Building Permit", "Owner: A. Khatib", "Plot No: 413", "Permit No: 77/2024"]])
    ocr = TextLayerOCR(ref)
    r = parse_document(str(native), "building_permit", {"providers": {"ocr": ocr, "vlm": None}})
    assert ocr.calls == [1]
    assert r["processing"]["max_level_reached"] == 2
    assert r["fields"]["permit_number"]["value"] == "77/2024"
    assert r["fields"]["plot_number"]["status"] == "conflict"
    assert r["status"] == "review_required"


def test_max_level_1_prevents_escalation(tmp_path):
    pdf = make_text_pdf(tmp_path / "p.pdf", [["Building Permit", "Owner: A. Khatib"]])
    ocr = TextLayerOCR(pdf)
    r = parse_document(str(pdf), "building_permit",
                       {"routing": {"max_level": 1}, "providers": {"ocr": ocr, "vlm": ScriptedVLM(["{}"])}})
    assert ocr.calls == []
    assert r["processing"]["max_level_reached"] == 1
    assert r["fields"]["permit_number"]["status"] == "missing"
    assert r["status"] == "review_required"


def test_surya_recorded_as_deferred(tmp_path):
    pdf = make_text_pdf(tmp_path / "p.pdf", [["Building Permit", "Owner: A. Khatib"]])
    r = parse_document(str(pdf), "building_permit", {"providers": NO_PROVIDERS})
    assert stages(r, "surya_layout")[0]["outcome"] == "unavailable"


# ---------------------------------------------------------------- level 3
def _vlm_answer(value, evidence, status="found"):
    return {"fields": {"permit_number": {"status": status, "value": value, "evidence_text": evidence}}}


def test_vlm_answer_corroborated_by_page_text(tmp_path):
    # Label is unusual so rules miss it, but the value is on the page.
    pdf = make_text_pdf(tmp_path / "p.pdf", [["Building Permit", "Ref. of licence 2024/0153", "Owner: A. Khatib"]])
    vlm = ScriptedVLM([_vlm_answer("2024/0153", "Ref. of licence 2024/0153")])
    r = parse_document(str(pdf), "building_permit", {"providers": {"ocr": None, "vlm": vlm}})
    f = r["fields"]["permit_number"]
    assert f["status"] == "extracted" and f["value"] == "2024/0153"
    assert f["provider"] == "scripted_vlm" and not f["requires_review"]
    assert r["processing"]["max_level_reached"] == 3
    assert any(e["provider"] == "scripted_vlm" and e["raw_text"] for e in f["evidence"])


def test_vlm_hallucination_requires_review(tmp_path):
    pdf = make_text_pdf(tmp_path / "p.pdf", [["Building Permit", "Owner: A. Khatib"]])
    vlm = ScriptedVLM([_vlm_answer("9999/0001", "Permit 9999/0001")])
    r = parse_document(str(pdf), "building_permit", {"providers": {"ocr": None, "vlm": vlm}})
    f = r["fields"]["permit_number"]
    assert f["requires_review"] and "vlm_answer_not_corroborated_by_page_text" in f["review_reasons"]
    assert r["status"] == "review_required"


def test_vlm_not_found_keeps_field_missing(tmp_path):
    pdf = make_text_pdf(tmp_path / "p.pdf", [["Building Permit", "Owner: A. Khatib"]])
    vlm = ScriptedVLM([_vlm_answer(None, None, status="not_found")])
    r = parse_document(str(pdf), "building_permit", {"providers": {"ocr": None, "vlm": vlm}})
    assert r["fields"]["permit_number"]["status"] == "missing"
    assert r["fields"]["permit_number"]["value"] is None


def test_malformed_vlm_output_is_retried_then_reported(tmp_path):
    pdf = make_text_pdf(tmp_path / "p.pdf", [["Building Permit", "Owner: A. Khatib"]])
    vlm = ScriptedVLM(["not json", '{"fields": {}}'], max_retries=1)
    r = parse_document(str(pdf), "building_permit", {"providers": {"ocr": None, "vlm": vlm}})
    assert vlm.calls == 2  # one retry, then give up (no loop)
    assert stages(r, "vlm_extract")[0]["outcome"] == "failed"
    assert r["fields"]["permit_number"]["status"] == "missing"
    assert r["status"] == "review_required"


def test_malformed_then_valid_vlm_output_succeeds(tmp_path):
    pdf = make_text_pdf(tmp_path / "p.pdf", [["Building Permit", "Ref. of licence 2024/0153"]])
    vlm = ScriptedVLM(["{broken", json.dumps(_vlm_answer("2024/0153", "2024/0153"))], max_retries=1)
    r = parse_document(str(pdf), "building_permit", {"providers": {"ocr": None, "vlm": vlm}})
    assert r["fields"]["permit_number"]["value"] == "2024/0153"


def test_vlm_page_budget_is_bounded(tmp_path):
    pdf = make_text_pdf(tmp_path / "p.pdf", [["Building Permit"]] * 6)
    vlm = ScriptedVLM([_vlm_answer(None, None, status="not_found")])
    parse_document(str(pdf), "building_permit", {"routing": {"max_vlm_pages": 2}, "providers": {"ocr": None, "vlm": vlm}})
    assert vlm.calls == 2


def test_missing_vlm_reports_need_for_review(tmp_path):
    pdf = make_text_pdf(tmp_path / "p.pdf", [["Building Permit", "Owner: A. Khatib"]])
    r = parse_document(str(pdf), "building_permit", {"providers": NO_PROVIDERS})
    assert any("no VLM available" in x for x in r["review_reasons"])
