"""Evaluate document_parser_core against a manually verified dataset.

Usage:
    python benchmarks/evaluate.py DATASET_DIR [--options options.json] [--out results.json]
                                  [--report report.md] [--tolerance 0.005]

DATASET_DIR/manifest.json is a list of entries:
{
  "file": "docs/permit_001.pdf",          # relative to DATASET_DIR
  "split": "benchmark",                   # entries with any other split are refused
  "document_type": "building_permit",     # optional
  "critical_fields": ["permit_number"],   # optional subset reported separately
  "expected_fields": {                    # normalised expected values; null = must be missing
      "permit_number": "2024/153",
      "area": {"value": 1250.5, "unit": "m2"}
  },
  "expected_text": "gt/permit_001.txt",   # optional: full ground-truth text for CER/WER
  "expected_tables": [                    # optional: coordinate rows [id, v1, v2, ...]
      {"rows": [["1", 245100.25, 1102345.10]]}
  ]
}

Benchmark documents must not be used for prompt or rule development. Keep them in a
separate directory and mark them "split": "benchmark".
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document_parser_core import __version__, parse_document  # noqa: E402
from document_parser_core.telemetry import peak_rss_mb  # noqa: E402
from document_parser_core.validators import values_conflict  # noqa: E402


def levenshtein(a: list, b: list) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def error_rate(ref: str, hyp: str, unit: str) -> float | None:
    r = list("".join(ref.split())) if unit == "char" else ref.split()
    h = list("".join(hyp.split())) if unit == "char" else hyp.split()
    if not r:
        return None
    return levenshtein(r, h) / len(r)


def evaluate_entry(entry: dict, root: Path, options: dict, tol: float) -> dict:
    path = root / entry["file"]
    tracemalloc.start()
    t0 = time.perf_counter()
    result = parse_document(str(path), entry.get("document_type"), options)
    seconds = time.perf_counter() - t0
    py_peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()

    critical = set(entry.get("critical_fields", []))
    fields = []
    for name, expected in entry.get("expected_fields", {}).items():
        got = result["fields"].get(name, {})
        status = got.get("status", "missing")
        flagged = bool(got.get("requires_review")) or status != "extracted"
        if expected is None:
            correct = status in ("missing", "explicitly_absent")
        else:
            correct = status == "extracted" and got.get("value") is not None and not values_conflict(got["value"], expected)
        fields.append({
            "name": name, "critical": name in critical, "expected": expected, "got": got.get("value"),
            "raw_text": got.get("raw_text"), "confidence": got.get("confidence"),
            "status": status, "correct": correct, "flagged_for_review": flagged,
            # accepted = system presented a value as usable without asking for review
            "false_acceptance": (status == "extracted" and not got.get("requires_review") and not correct),
        })

    text_metrics = {}
    if entry.get("expected_text"):
        ref = (root / entry["expected_text"]).read_text(encoding="utf-8")
        hyp = "\n".join(p.get("text") or "" for p in result["pages"])
        text_metrics = {"cer": error_rate(ref, hyp, "char"), "wer": error_rate(ref, hyp, "word")}

    table_metrics = {}
    if entry.get("expected_tables"):
        exp_rows = {str(r[0]): r[1:] for t in entry["expected_tables"] for r in t["rows"]}
        got_rows = {}
        for t in result["tables"]:
            if t["kind"] != "coordinate_table":
                continue
            for row in t["rows"]:
                vals = [v for k, v in row["values"].items() if k != "point_id"]
                got_rows.setdefault(str(row["values"].get("point_id")), vals)
        n_vals = n_ok = 0
        for pid, exp in exp_rows.items():
            got = got_rows.get(pid)
            for k, e in enumerate(exp):
                n_vals += 1
                if got and k < len(got) and got[k] is not None and abs(float(got[k]) - float(e)) <= tol:
                    n_ok += 1
        table_metrics = {"expected_values": n_vals, "correct_values": n_ok,
                         "numeric_accuracy": n_ok / n_vals if n_vals else None,
                         "rows_found": len(set(exp_rows) & set(got_rows)), "rows_expected": len(exp_rows)}

    return {
        "file": entry["file"], "status": result["status"], "seconds": round(seconds, 3),
        "python_peak_mb": round(py_peak, 1), "max_level": result["processing"]["max_level_reached"],
        "providers": result["processing"]["providers_used"], "fields": fields,
        "text": text_metrics, "tables": table_metrics, "error": result.get("error"),
    }


def summarise(rows: list[dict]) -> dict:
    fields = [f for r in rows for f in r["fields"]]
    crit = [f for f in fields if f["critical"]]
    wrong = [f for f in fields if not f["correct"]]
    right = [f for f in fields if f["correct"]]
    present = [f for f in fields if f["expected"] is not None]

    def rate(xs, pred):
        return round(sum(pred(x) for x in xs) / len(xs), 4) if xs else None

    cers = [r["text"]["cer"] for r in rows if r["text"].get("cer") is not None]
    wers = [r["text"]["wer"] for r in rows if r["text"].get("wer") is not None]
    tabs = [r["tables"] for r in rows if r["tables"]]
    times = [r["seconds"] for r in rows]
    return {
        "documents": len(rows),
        "failed_documents": sum(r["status"] == "failed" for r in rows),
        "field_exact_match": rate(fields, lambda f: f["correct"]),
        "critical_field_exact_match": rate(crit, lambda f: f["correct"]),
        "field_coverage": rate(present, lambda f: f["status"] == "extracted"),
        "false_acceptance_rate": rate(fields, lambda f: f["false_acceptance"]),
        "uncertainty_detection_rate": rate(wrong, lambda f: f["flagged_for_review"]),
        "unnecessary_review_rate": rate(right, lambda f: f["flagged_for_review"]),
        "mean_cer": round(statistics.mean(cers), 4) if cers else None,
        "mean_wer": round(statistics.mean(wers), 4) if wers else None,
        "coordinate_numeric_accuracy": (round(sum(t["correct_values"] for t in tabs) / sum(t["expected_values"] for t in tabs), 4)
                                        if tabs and sum(t["expected_values"] for t in tabs) else None),
        "seconds_mean": round(statistics.mean(times), 3) if times else None,
        "seconds_p95": round(sorted(times)[max(0, int(0.95 * len(times)) - 1)], 3) if times else None,
        "process_peak_rss_mb": peak_rss_mb(),
        "max_python_peak_mb": max((r["python_peak_mb"] for r in rows), default=None),
    }


def render_report(template: str, summary: dict, meta: dict) -> str:
    values = {**{k: ("n/a" if v is None else v) for k, v in summary.items()}, **meta}
    for key, val in values.items():
        template = template.replace("{{" + key + "}}", str(val))
    return template


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--options", type=Path, help="JSON file with parse_document options")
    ap.add_argument("--out", type=Path, default=Path("benchmark_results.json"))
    ap.add_argument("--report", type=Path)
    ap.add_argument("--tolerance", type=float, default=0.005, help="absolute tolerance for coordinate values")
    args = ap.parse_args(argv)

    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    bad = [e["file"] for e in manifest if e.get("split") != "benchmark"]
    if bad:
        print(f"Refusing entries not marked split='benchmark': {bad}", file=sys.stderr)
        return 2
    options = json.loads(args.options.read_text()) if args.options else {}
    rows = []
    for e in manifest:
        rows.append(evaluate_entry(e, args.dataset, options, args.tolerance))
        print(f"{e['file']}: {rows[-1]['status']} ({rows[-1]['seconds']}s)", file=sys.stderr)
    summary = summarise(rows)
    meta = {"package_version": __version__, "dataset": str(args.dataset), "date": time.strftime("%Y-%m-%d"),
            "options": json.dumps(options)}
    args.out.write_text(json.dumps({"meta": meta, "summary": summary, "documents": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.report:
        tpl = (Path(__file__).parent / "REPORT_TEMPLATE.md").read_text(encoding="utf-8")
        args.report.write_text(render_report(tpl, summary, meta), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
