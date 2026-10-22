"""The evaluation script runs and computes metrics correctly (not an accuracy claim)."""

import importlib.util
import json
from pathlib import Path

import pytest

from document_parser_core.tests.conftest import COORD_HEADER, COORD_ROWS
from document_parser_core.tests.fixtures import make_table_pdf, make_text_pdf

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks" / "evaluate.py"


def _load():
    spec = importlib.util.spec_from_file_location("evaluate", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_levenshtein_and_error_rates():
    ev = _load()
    assert ev.levenshtein(list("kitten"), list("sitting")) == 3
    assert ev.error_rate("a b c", "a b c", "word") == 0
    assert ev.error_rate("abcd", "abed", "char") == pytest.approx(0.25)


def test_evaluate_synthetic_dataset(tmp_path):
    ev = _load()
    make_text_pdf(tmp_path / "p1.pdf", [["Building Permit", "Permit No: 12/2024", "Plot No: 7"]])
    make_table_pdf(tmp_path / "c1.pdf", COORD_HEADER, COORD_ROWS)
    (tmp_path / "p1.txt").write_text("Building Permit\nPermit No: 12/2024\nPlot No: 7", encoding="utf-8")
    manifest = [
        {"file": "p1.pdf", "split": "benchmark", "document_type": "building_permit",
         "critical_fields": ["permit_number"], "expected_text": "p1.txt",
         # plot_number deliberately expected wrong -> must count as error and false acceptance
         "expected_fields": {"permit_number": "12/2024", "plot_number": "8", "owner_name": None}},
        {"file": "c1.pdf", "split": "benchmark", "document_type": "coordinate_table",
         "expected_tables": [{"rows": [[r[0], *map(float, r[1:])] for r in COORD_ROWS]}]},
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    opts = tmp_path / "opts.json"
    opts.write_text(json.dumps({"ocr_provider": None, "vlm": {"enabled": False}}))
    out, report = tmp_path / "res.json", tmp_path / "report.md"
    assert ev.main([str(tmp_path), "--options", str(opts), "--out", str(out), "--report", str(report)]) == 0
    s = json.loads(out.read_text())["summary"]
    assert s["documents"] == 2 and s["failed_documents"] == 0
    assert s["critical_field_exact_match"] == 1.0
    assert s["field_exact_match"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["false_acceptance_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["mean_cer"] == 0.0
    assert s["coordinate_numeric_accuracy"] == 1.0
    assert "{{" not in report.read_text()


def test_refuses_non_benchmark_split(tmp_path):
    ev = _load()
    (tmp_path / "manifest.json").write_text(json.dumps([{"file": "x.pdf", "split": "dev"}]))
    assert ev.main([str(tmp_path), "--out", str(tmp_path / "o.json")]) == 2
