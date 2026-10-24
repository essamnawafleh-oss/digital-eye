# Dependency and Model Licensing Report

Sources: package metadata of the versions installed on 2026-10-22 (VERIFIED), plus
model-card knowledge that could **not** be re-checked because Hugging Face was
unreachable (UNVERIFIED). This is an engineering summary, not legal advice.

## Python packages

| Package | Version | Licence (metadata) | Used for | Redistribution notes |
|---|---|---|---|---|
| pymupdf | 2.0 | **AGPL-3.0 or Artifex commercial** | PDF text, rendering, table finder | Distributing software that uses it, or offering it over a network, triggers AGPL obligations (release the complete application source under AGPL) unless an Artifex commercial licence is bought. **Highest-impact item.** Isolated in `pdf_handler.py` so it can be replaced (e.g. pypdfium2, Apache-2.0/BSD) if needed. |
| pymupdf-layout | 2.0 | **not declared** | not used (installed by pymupdf 2.0) | Unknown; ask Artifex. Contains bundled ONNX models. |
| pymupdf4llm | 2.0 | **not declared** | not used (installed by pymupdf 2.0) | Unknown. |
| onnxruntime | (dep of pymupdf-layout) | MIT | not used | — |
| pillow | 10.4–13.0 | MIT-CMU (HPND) | image I/O | Permissive. |
| numpy | 2.x | BSD-3-Clause | arrays | Permissive. |
| pydantic | 2.x | MIT | schema | Permissive. |
| paddlepaddle | 3.3.1 | Apache-2.0 | OCR runtime | Permissive; keep NOTICE. |
| paddleocr | 3.7.0 | Apache-2.0 | OCR pipeline | Permissive. |
| paddlex | 3.7.2 | Apache-2.0 | OCR backend | Permissive. Pulls `opencv-contrib-python` (Apache-2.0), `huggingface_hub`, `modelscope`, `aistudio-sdk` (download clients; not used at runtime here). |
| llama-cpp-python | 0.3.36 | MIT | VLM runtime | Permissive; bundles llama.cpp/ggml (MIT). |
| surya-ocr (deferred) | 0.22.1 | Apache-2.0 (code) | not used yet | Earlier releases were GPL-3.0: check the version you install. |
| torch (deferred) | 2.14.1 | BSD-style + others | Surya | PyPI Linux wheel bundles NVIDIA CUDA libraries under NVIDIA's EULA. |
| transformers (deferred) | 5.20.2 | Apache-2.0 | Surya | Permissive. |

## Model weights

| Model | Licence | Status | Notes |
|---|---|---|---|
| PP-OCRv5_mobile_det, arabic_PP-OCRv5_mobile_rec, en_PP-OCRv5_mobile_rec (the models in use) | **Apache-2.0** | VERIFIED 2026-10-24 on the downloaded model cards | Commercial use allowed; keep the licence notice. |
| Qwen2.5-VL-7B-Instruct (GGUF by ggml-org) | Apache-2.0 | UNVERIFIED | Default candidate. GGUF conversion inherits the base model licence. |
| Qwen2.5-VL-3B-Instruct | **Qwen Research License (non-commercial)** | UNVERIFIED | Do not use commercially without checking. |
| Qwen2.5-VL-72B-Instruct | Qwen License (usage threshold terms) | UNVERIFIED | |
| Surya models (layout2, detection, OCR) | **UNVERIFIED**; older versions used a modified AI Pubs Open RAIL-M with commercial revenue limits | UNVERIFIED | Required before enabling the Surya provider. |
| PyMuPDF-layout bundled ONNX models | not declared | UNVERIFIED | Not used. |

## Fonts used in tests

Test fixtures use a system font at runtime (DejaVu Sans on Linux: Bitstream Vera /
public-domain-derived licence). `docs/samples/permit_mixed.pdf` embeds a DejaVu Sans
subset, which that licence permits.

## Actions before distribution

1. Decide on PyMuPDF: AGPL compliance, a commercial licence, or replacing the PDF backend.
2. Confirm every model card licence for the exact files provisioned, and record them next to
   `models/manifest.json`.
3. Re-run this inventory (`pip-licenses` or package metadata) for the final pinned environments.
