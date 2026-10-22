# document_parser_core

Offline-first, modular parsing of engineering documents, building permits, plans,
coordinate tables, invoices and scans (PDF, PNG, JPG/JPEG) into structured JSON with
field-level evidence. Standalone; not yet integrated into Survey-Licensing-Studio.

**Status: first vertical slice.** Native-PDF extraction, routing, validation, schema,
offline enforcement and the OCR/VLM adapters are implemented and tested with synthetic
documents and fake providers. **PaddleOCR and the Qwen2.5-VL provider have not been run
with real model weights** (model hosts were unreachable from the development
environment), and no accuracy has been measured. See
[`docs/COMPATIBILITY_REPORT.md`](../docs/COMPATIBILITY_REPORT.md) and
[`docs/KNOWN_LIMITATIONS.md`](../docs/KNOWN_LIMITATIONS.md).

## Quick start

```python
from document_parser_core import parse_document

result = parse_document("permit.pdf")                          # type auto-detected
result = parse_document("scan.jpg", "building_permit",
                        {"models_dir": "/opt/dpc-models", "render_dpi": 300})

result["status"]            # success | partial | review_required | failed
result["review_required"]   # True whenever anything is uncertain
result["fields"]["permit_number"]["value"]
result["fields"]["permit_number"]["evidence"]   # every occurrence, provider, page, bbox
```

CLI: `python -m document_parser_core.examples.parse_file permit.pdf --type building_permit`

A complete real output (from `docs/samples/permit_mixed.pdf`) is in
[`docs/sample_output.json`](../docs/sample_output.json).

## Installation

Python 3.11–3.13 (verified: 3.13.16 on Ubuntu 24.04 x86_64, CPU only).

| Purpose | Command | Notes |
|---|---|---|
| Core (native PDFs, validation) | `pip install -r document_parser_core/requirements.txt` then `pip install -e . --no-deps` | Always required |
| + PaddleOCR | `pip install -r document_parser_core/requirements-paddle.txt` | CPU wheels for Linux x86_64, Windows x64, macOS arm64 |
| + VLM | `CMAKE_ARGS="-DGGML_NATIVE=OFF" pip install -r document_parser_core/requirements-vlm.txt` | Compiles llama.cpp (no wheels on PyPI) |
| Surya (deferred) | separate venv, `requirements-surya.txt` | Not used by routing yet |

Install only what your pipeline needs. Text-only PDFs need the core alone.

### Separate environments (recommended)

PaddleOCR and Surya **must not** share an environment (they install two OpenCV
distributions into the same `cv2` folder). Torch, Paddle and llama.cpp also each grab
their own GPU memory. Every provider can run in its own venv through a local worker
process, behind the same `parse_document` call:

```bash
python -m venv .venv-paddle && .venv-paddle/bin/pip install -r document_parser_core/requirements-paddle.txt && .venv-paddle/bin/pip install -e . --no-deps
python -m venv .venv-vlm    && .venv-vlm/bin/pip install -r document_parser_core/requirements-vlm.txt    && .venv-vlm/bin/pip install -e . --no-deps
```
```python
parse_document("scan.pdf", options={
    "models_dir": "/opt/dpc-models",
    "paddle": {"python_executable": ".venv-paddle/bin/python"},
    "vlm": {"python_executable": ".venv-vlm/bin/python"},
})
```
(Verified: core venv → Paddle-venv worker and VLM-venv worker; errors such as missing
models propagate with their type.)

### Windows

- Use 64-bit Python 3.11–3.13. PyMuPDF, PaddlePaddle (CPU) and Torch publish `win_amd64` wheels.
- `llama-cpp-python` builds from source: install *Visual Studio 2022 Build Tools* (C++
  workload) and CMake first, then `set CMAKE_ARGS=-DGGML_NATIVE=OFF` and pip install.
- Worker interpreters: `"python_executable": "C:\\path\\.venv-paddle\\Scripts\\python.exe"`.
- Offline verification: disable the network adapter and run `python scripts\verify_offline.py --models-dir C:\dpc-models`.
- Windows installation has **not** been tested.

### GPU (not verified — no GPU was available)

- Paddle: install the CUDA build of `paddlepaddle-gpu` from Paddle's own index matching
  your CUDA version, then set `options={"paddle": {"device": "gpu:0"}}`.
- llama.cpp: `CMAKE_ARGS="-DGGML_CUDA=on"` and `options={"vlm": {"n_gpu_layers": -1}}`.
- Torch from PyPI on Linux is already a CUDA 13 build; for CPU-only machines use the CPU index.
- CPU fallback is the default everywhere (`device="cpu"`, `n_gpu_layers=0`).

## Offline operation

1. **Install + provision** (online): dependencies, then models as described in
   [`models/README.md`](models/README.md). Pin them: `python scripts/verify_models.py --write`.
2. **Run** (offline): nothing in the parsing path downloads anything.
   - `offline=True` (default) sets `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`,
     `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` before any provider is imported.
   - PaddleOCR is always given explicit local model dirs; the VLM gets local GGUF paths
     and images only as `data:` URIs (llama-cpp would `urlopen` any other URL).
   - Missing weights → a `ModelNotFoundError` naming the exact path, recorded in
     `processing.stages`, and the document routes to `review_required`.
   - `strict_network_guard=True` makes any non-loopback socket connection or DNS lookup
     in the process raise `OfflineViolationError`.
3. **Verify**: `scripts/verify_offline.sh --models-dir DIR` (Linux: runs in a network
   namespace with no network at all) or `python scripts/verify_offline.py --models-dir DIR`
   with the network disabled. It reports `INCOMPLETE` if providers could not be exercised.

## How routing works

| Level | When | What runs |
|---|---|---|
| 1 Fast | always | PDF text layer with quality checks; OCR only on pages whose text layer is empty, image-dominated, vector-dominated, garbled, reversed (visual-order Arabic) or has broken glyph maps |
| 2 Layout | required field missing, or expected table missing/inconsistent | PyMuPDF table finder; OCR cross-check of native pages (catches incomplete text layers and disagreements); Surya recorded as deferred |
| 3 Vision | required field still missing / unreadable / conflicting / invalid | Local VLM on at most `max_vlm_pages` pages, only for those fields |

Each level runs at most once; VLM retries are capped (`vlm.max_retries`). VLM answers are
accepted only when their evidence text or value is found in the page's text, otherwise
the field is marked `requires_review`. Conflicting values are never resolved by picking
one, unless a corroborated VLM answer agrees with exactly one of them.

Thresholds are in `ParserConfig.routing` (`min_ocr_confidence`, `max_level`, ...).

## Output schema (v1.0)

`parse_document` returns `ParseResult.model_dump(mode="json")`; the JSON Schema is
available from `document_parser_core.result_json_schema()`. Key rules (see
`schemas.py` docstring):

- Field `status`: `extracted` | `missing` | `unreadable` | `explicitly_absent` | `conflict`.
  `null` values are always explained by the status.
- `validation_status`: `valid` | `warning` | `invalid` | `not_validated`.
- Boxes are `[x0, y0, x1, y1]`, top-left origin, in `pdf_points` (PDF) or
  `image_pixels` (images); `approx_bbox` marks word boxes interpolated from OCR line boxes.
- `confidence` + `confidence_kind`: provider scores, never calibrated probabilities.
- `evidence[]` keeps every occurrence/provider answer separately from the chosen value.

## Document types and custom fields

Built-in templates (`document_types.py`): `generic`, `building_permit`, `invoice`,
`coordinate_table`, `architectural_plan`. They hold only label words. No regulations,
zoning or CRS rules are included, and none are inferred. Add or override fields:

```python
parse_document(path, "generic", {"fields": [
    {"name": "surveyor_licence", "type": "identifier", "required": True,
     "labels": ["surveyor licence", "رخصة المساح"]},
]})
```
Types: `text`, `identifier`, `number`, `integer`, `currency_amount`, `date`, `area`, `length`.

## Tests and benchmarks

```bash
pytest                                   # synthetic fixtures + fake providers
DPC_MODELS_DIR=/opt/dpc-models pytest -m requires_models   # real-model tests (opt-in)
python benchmarks/evaluate.py DATASET_DIR --report report.md
```
The benchmark manifest format, metrics (exact match, CER/WER, coordinate accuracy,
coverage, false acceptance, uncertainty detection, time, memory) and the report template
are in `benchmarks/`. Entries must be marked `"split": "benchmark"`, so benchmark documents
stay apart from documents used to develop rules or prompts.

## Licensing

PyMuPDF is **AGPL-3.0 or commercial**. Read [`docs/LICENSES.md`](../docs/LICENSES.md)
before distributing anything built on this package.
