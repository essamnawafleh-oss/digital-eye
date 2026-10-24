"""OCR post-processing rules. Each input below is a real arabic_PP-OCRv5_mobile_rec
output observed on 2026-10-24 (see docs/BENCHMARK_RESULTS.md)."""

import pytest

from document_parser_core import parse_document
from document_parser_core import validators as v
from document_parser_core.core.base_handler import OCRProvider
from document_parser_core.core.paddle_handler import ocr_line, split_script_boundaries
from document_parser_core.tests.fixtures import make_text_pdf


class LinesOCR(OCRProvider):
    """Fake OCR returning fixed lines: (text, score, cross_check_text)."""

    name = "lines_ocr"

    def __init__(self, lines):
        self.lines = lines

    def recognize(self, image, page_number, scale):
        out = []
        for i, (text, score, cross) in enumerate(self.lines):
            ln = ocr_line(text, [60, 60 + 30 * i, 500, 80 + 30 * i], page_number, score)
            ln.cross_check_text = cross
            for w in ln.words:
                w.line_id = i
            out.append(ln)
        return out


def _parse(tmp_path, lines, doc_type="building_permit"):
    blank = make_text_pdf(tmp_path / "blank.pdf", [[""]])  # no text layer -> OCR path
    return parse_document(str(blank), doc_type, {"providers": {"ocr": LinesOCR(lines), "vlm": None}})


@pytest.mark.parametrize("raw,expected", [
    ("BP-2024/0153رقم الرخصة", "BP-2024/0153 رقم الرخصة"),
    ("Owner:محمد الخطيب", "Owner: محمد الخطيب"),
    ("رقم القطعة7", "رقم القطعة 7"),
    ("المساحة ١٢٥٠", "المساحة ١٢٥٠"),
])
def test_split_script_boundaries(raw, expected):
    assert split_script_boundaries(raw) == expected


def test_number_not_taken_from_unit_or_middle_of_digits():
    assert v.parse_number("m2 1250.5").value == 1250.5
    assert v.parse_number("م2 ١٢٥٠٫٥").value == 1250.5
    assert not v.parse_number("الكلية289366x").ok or v.parse_number("الكلية289366x").value != 89366


def test_value_printed_before_label(tmp_path):
    r = _parse(tmp_path, [("BP-2024/0153 :رقم الرخصة", 0.99, None)])
    f = r["fields"]["permit_number"]
    assert f["value"] == "BP-2024/0153"
    assert "value printed before label" in f["evidence"][0]["note"]


def test_value_before_label_without_colon_in_rtl_line(tmp_path):
    r = _parse(tmp_path, [("BP-2024/0153 رقم الرخصة", 0.99, None)])
    assert r["fields"]["permit_number"]["value"] == "BP-2024/0153"


def test_glued_arabic_label_and_value(tmp_path):
    r = _parse(tmp_path, [("اسم المالكمحمد أحمد الخطيب", 0.99, None), ("الحوضالجبيهة الشمالي", 0.99, None)])
    assert r["fields"]["owner_name"]["value"] == "محمد أحمد الخطيب"
    assert r["fields"]["basin"]["value"] == "الجبيهة الشمالي"


def test_word_starting_with_label_is_not_split(tmp_path):
    r = _parse(tmp_path, [("المالكين: شركة النور", 0.99, None)])
    assert r["fields"]["owner_name"]["status"] == "missing"


def test_label_read_as_value_is_rejected(tmp_path):
    r = _parse(tmp_path, [("رقم الرخصة: تاريخالإصدا", 0.95, None)])
    assert r["fields"]["permit_number"]["status"] == "missing"


def test_arabic_indic_digits_from_ocr_need_review(tmp_path):
    r = _parse(tmp_path, [("المساحة الكلية: ٢٨٩٣٦٦ م2", 0.97, None)])
    f = r["fields"]["area"]
    assert f["status"] == "extracted" and f["requires_review"]
    assert "ocr_of_arabic_indic_digits_is_unreliable" in f["review_reasons"]


def test_area_without_unit_needs_review(tmp_path):
    r = _parse(tmp_path, [("Total Area: 1250.5", 0.99, "Total Area: 1250.5")])
    assert "unit_not_stated_or_not_read" in r["fields"]["area"]["review_reasons"]


def test_mid_confidence_value_returned_but_flagged(tmp_path):
    r = _parse(tmp_path, [("Plot No: 412", 0.85, "Plot No: 412")])
    f = r["fields"]["plot_number"]
    assert f["value"] == "412" and f["requires_review"]
    assert any("below_review_threshold" in x for x in f["review_reasons"])


def test_cross_check_disagreement_flags_value(tmp_path):
    # Arabic model dropped a digit; the Latin model read the line correctly.
    r = _parse(tmp_path, [("رقم القطعة 18", 0.94, "128 ")])
    f = r["fields"]["plot_number"]
    assert f["value"] == "18" and f["requires_review"]
    assert any("cross_check_disagrees" in x and "128" in x for x in f["review_reasons"])
    assert r["status"] == "review_required"


def test_cross_check_agreement_accepts_value(tmp_path):
    r = _parse(tmp_path, [("رقم القطعة 412", 0.98, "412 :")])
    f = r["fields"]["plot_number"]
    assert f["value"] == "412" and not f["requires_review"]


def test_cross_check_catches_truncated_identifier(tmp_path):
    r = _parse(tmp_path, [("BP-2025/907 رقم الرخصة", 0.95, "BP-2025/9070 ")])
    assert r["fields"]["permit_number"]["requires_review"]


def test_persian_letter_forms_are_normalised(tmp_path):
    # OCR returned Farsi yeh (U+06CC) in an Arabic name: 'لیلى' for 'ليلى'.
    r = _parse(tmp_path, [("اسم المالك: لیلى نور القاسم", 0.98, None), ("الحوض: کفر", 0.98, None)])
    assert r["fields"]["owner_name"]["value"] == "ليلى نور القاسم"
    assert r["fields"]["basin"]["value"] == "كفر"


# ---------------------------------------------------------------- merging the two readers
from document_parser_core.core.paddle_handler import merge_line_readings  # noqa: E402


def test_merge_takes_latin_value_from_latin_model():
    # Arabic model dropped the date entirely (observed in 7 of 8 fonts).
    r = merge_line_readings("تاريخ الإصدار", 0.97, "13/01/2017:j", 0.93)
    assert r.reading == "arabic+latin" and r.text == "تاريخ الإصدار 13/01/2017"
    assert r.latin_score == 0.93 and r.cross_text is None


def test_merge_keeps_arabic_models_own_digits_as_cross_check():
    r = merge_line_readings("P-20رقم الرخصة", 0.9, "BP-2023/0351", 0.99)
    assert "BP-2023/0351" in r.text and r.cross_text == "P-20رقم الرخصة"


def test_merge_never_uses_latin_digits_for_arabic_indic_numbers():
    # Latin model reads '١٢٥٠٫٥' as plausible junk.
    r = merge_line_readings("المساحة الكلية ١٢٥٠٫٥", 0.9, "2110.0:", 0.95)
    assert r.reading == "arabic" and "2110" not in r.text


def test_merge_ignores_low_score_latin_reading():
    r = merge_line_readings("المساحة الكلية", 0.95, "21r0.0", 0.64)
    assert r.reading == "arabic" and r.text == "المساحة الكلية"


def test_latin_only_line_uses_latin_model():
    r = merge_line_readings("Permit No: LC-2019/9764", 0.98, "Permit No: LIC-2019/9764", 0.91)
    assert r.reading == "latin" and r.text.endswith("LIC-2019/9764") and r.cross_text.endswith("LC-2019/9764")


def test_merged_line_gives_latin_tokens_the_latin_score():
    r = merge_line_readings("رقم القطعة", 0.97, "128 ", 0.84)
    ln = ocr_line(r.text, [0, 0, 300, 20], 1, r.score, latin_score=r.latin_score)
    assert {w.text: w.confidence for w in ln.words}["128"] == 0.84
    assert ln.confidence == 0.84
