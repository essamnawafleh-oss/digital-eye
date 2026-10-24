# digital-eye

Home of **`document_parser_core`**: a standalone, offline-first Python package that
extracts structured, evidence-tracked data from engineering documents, building permits,
plans, coordinate tables, invoices and scans (PDF / PNG / JPEG, Arabic and English).

- Package docs, installation, offline setup and usage: [`document_parser_core/README.md`](document_parser_core/README.md)
- Dependency feasibility & compatibility report: [`docs/COMPATIBILITY_REPORT.md`](docs/COMPATIBILITY_REPORT.md)
- Licensing report: [`docs/LICENSES.md`](docs/LICENSES.md)
- Known limitations: [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md)
- Sample output: [`docs/sample_output.json`](docs/sample_output.json) (input: `docs/samples/permit_mixed.pdf`)
- Model provisioning: [`document_parser_core/models/README.md`](document_parser_core/models/README.md)
- Benchmarks: [`benchmarks/`](benchmarks/)

Status: first vertical slice. Native PDF extraction and PaddleOCR (Arabic/English scans)
work offline and are tested, OCR only on **synthetic** scans so far
([benchmark](docs/BENCHMARK_RESULTS.md)). The optional vision model (Qwen2.5-VL via
llama.cpp) is implemented but **not yet run with real weights**.
