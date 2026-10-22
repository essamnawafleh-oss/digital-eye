from __future__ import annotations

import pytest

from document_parser_core.tests.fixtures import (
    arabic_font, make_scanned_pdf, make_table_pdf, make_text_pdf,
)

PERMIT_PAGE_1 = [
    "Building Permit / رخصة بناء",
    "Permit No: BP-2024/0153",
    "رقم الرخصة: BP-2024/0153",
    "Issue Date: 05/03/2024",
    "اسم المالك: محمد أحمد الخطيب",
    "Plot No: 412",
    "الحوض: الجبيهة الشمالي",
    "Total Area: 1,250.50 m2",
]
PERMIT_PAGE_2 = [
    "Notes / ملاحظات",
    "Plot No: 412",
]

COORD_HEADER = ["Point", "E", "N", "Z"]
COORD_ROWS = [
    ["1", "245100.25", "1102345.10", "880.15"],
    ["2", "245130.60", "1102360.45", "881.02"],
    ["3", "245150.05", "1102330.90", "879.77"],
    ["4", "245120.75", "1102310.35", "879.40"],
]


@pytest.fixture(scope="session")
def font():
    f = arabic_font()
    if f is None:
        pytest.skip("no Arabic-capable system font found for fixtures")
    return f


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("fixtures")


@pytest.fixture(scope="session")
def permit_en_pdf(fixture_dir):
    rows = [r for r in PERMIT_PAGE_1 if not any("؀" <= c <= "ۿ" for c in r)]
    return make_text_pdf(fixture_dir / "permit_en.pdf", [rows])


@pytest.fixture(scope="session")
def permit_mixed_pdf(fixture_dir, font):
    return make_text_pdf(fixture_dir / "permit_mixed.pdf", [PERMIT_PAGE_1, PERMIT_PAGE_2], font)


@pytest.fixture(scope="session")
def permit_ar_pdf(fixture_dir, font):
    rows = ["رخصة بناء", "رقم الرخصة: 2024/153", "تاريخ الإصدار: 2024-03-05", "اسم المالك: محمد أحمد الخطيب",
            "رقم القطعة: 412", "الحوض: الجبيهة الشمالي", "المساحة الكلية: ١٢٥٠٫٥ م2"]
    return make_text_pdf(fixture_dir / "permit_ar.pdf", [rows], font)


@pytest.fixture(scope="session")
def coord_pdf(fixture_dir):
    return make_table_pdf(fixture_dir / "coords.pdf", COORD_HEADER, COORD_ROWS)


@pytest.fixture(scope="session")
def scanned_permit_pdf(fixture_dir, permit_en_pdf):
    return make_scanned_pdf(permit_en_pdf, fixture_dir / "permit_scanned.pdf")
