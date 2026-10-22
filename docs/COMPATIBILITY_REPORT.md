# Dependency Feasibility & Compatibility Report

Date of inspection: 2026-10-22. Everything below is marked **VERIFIED** (checked in
this environment by installing / importing / running), **SOURCE-VERIFIED** (read
from the installed package source or official docs, not executed with real
models), or **UNVERIFIED** (could not be checked; must be confirmed before use).

## 1. Inspection environment

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.5 LTS, x86_64 (Linux container) |
| Python | 3.13.16 |
| CPU / RAM | 4 vCPU, 15 GB |
| GPU | **None** (no `nvidia-smi`, no `/dev/nvidia*`) — all checks are CPU-only |
| Build tools | gcc 13.3, cmake (system) |
| Network | PyPI + GitHub reachable. **Blocked:** huggingface.co, hf-mirror.com, `*.bcebos.com` (PaddleOCR models), modelscope.cn, aistudio.baidu.com, models.datalab.to (Surya), download.pytorch.org |

Consequence: **no model weights could be downloaded in this environment.** Every
statement about OCR/VLM *accuracy or end-to-end behaviour with real weights* is
UNVERIFIED. Windows was not available for testing.

## 2. Library versions tested (latest on PyPI on the inspection date)

| Package | Version | Py3.13 wheel | License (from package metadata) | Status |
|---|---|---|---|---|
| pymupdf | 2.0 | cp311-abi3 | **AGPL-3.0 or Artifex commercial** | VERIFIED import + text/word/render API |
| pymupdf-layout, pymupdf4llm | 2.0 | pulled in **automatically** by pymupdf 2.0 | **not declared in metadata** | Installed transitively (adds onnxruntime, networkx); NOT used by this package |
| paddlepaddle (CPU) | 3.3.1 | cp313 | Apache-2.0 | VERIFIED import |
| paddleocr | 3.7.0 | py3 | Apache-2.0 | VERIFIED import; API SOURCE-VERIFIED |
| paddlex | 3.7.2 | py3 | Apache-2.0 | VERIFIED import (dependency of paddleocr) |
| surya-ocr | 0.22.1 | py3 | Apache-2.0 (code; earlier releases were GPL-3.0) | VERIFIED import of `surya.recognition/layout/detection` |
| torch | 2.14.1 (+cu130 wheel from PyPI) | cp313 | BSD-style/Apache mix | VERIFIED import (CPU) |
| transformers | 5.20.2 | py3 | Apache-2.0 | installed by surya |
| llama-cpp-python | 0.3.36 | **sdist only** – compiled from source | MIT | VERIFIED CPU build (`GGML_NATIVE=OFF`) incl. `libmtmd.so` |

## 3. Cross-library compatibility findings

1. **All four stacks resolve and import in one Python 3.13 environment** (VERIFIED, both
   `paddle → torch` and `torch → paddle` import orders). No hard resolver conflict.
2. **OpenCV collision (VERIFIED):** paddlex pins `opencv-contrib-python==4.10.0.84`,
   surya pulls `opencv-python-headless==4.11.0.86`. Both get installed and overwrite
   the same `cv2/` package directory; the result is a mixed install (`cv2.__version__`
   reports 4.11 while contrib modules from 4.10 remain). It imports, but it is
   fragile and undefined behaviour. **→ Do not install Paddle and Surya in the same
   environment.**
3. **Pillow downgrade (VERIFIED):** surya pins `pillow==10.4.0`; alone, paddle gets 13.0.
4. **numpy:** paddle env resolves 2.3.5, surya env 2.5.4; combined 2.3.5. Works on import.
5. **Torch wheel from PyPI is the CUDA 13 build** (`2.14.1+cu130`, several GB with
   `nvidia-*` libs). For CPU-only machines install from the PyTorch CPU index
   (not reachable here, UNVERIFIED).
6. **GPU memory contention** (UNVERIFIED, no GPU): Paddle, Torch and llama.cpp each
   allocate their own CUDA memory pools; running them in one process on a small GPU is
   a known risk. Separate worker processes avoid it.

**Decision:** separate dependency manifests (`core`, `paddle`, `vlm`, `surya`) and an
optional **subprocess worker** adapter so each provider can run with its own Python
interpreter while `parse_document()` stays the single public API.

## 4. Provider-specific findings

### PyMuPDF 2.0 (VERIFIED)
- `pymupdf.open`, `page.get_text("text"|"words")`, `page.get_pixmap(dpi=...)`,
  `page.find_tables()` all work; `import fitz` still works but emits a
  `FutureWarning` (deprecated) → use `import pymupdf`.
- **Arabic text-layer defects reproduced:**
  - Text written in *visual order* (as many legacy/CAD PDF producers do) is
    extracted **reversed**: `ةصخرلا مقر` instead of `رقم الرخصة`, `٤٢٠٢` instead of `٢٠٢٤`.
  - Shaped text comes back as **Arabic Presentation Forms** (U+FB50–U+FEFF), and in
    the tested font the lam-alef ligature was mapped to `Ǆ` (U+01C4) — a broken
    ToUnicode map.
  → A non-empty text layer is not trustworthy. The direct-text provider normalises
  presentation forms (NFKC) and scores pages for reversed-Arabic and
  foreign-script-inside-Arabic-word signals, escalating to OCR when suspicious.

### PaddleOCR 3.7.0 (SOURCE-VERIFIED, not run with weights)
- `PaddleOCR(lang=..., text_detection_model_dir=..., text_recognition_model_dir=...)`.
  When model dirs are given, `lang/ocr_version` are ignored.
- `lang="ar"` maps to **`PP-OCRv5_server_det` + `arabic_PP-OCRv5_mobile_rec`**.
  Whether this recogniser's dictionary covers Latin letters well enough for mixed
  Arabic/English lines is **UNVERIFIED**; the design allows running a second
  (`en`) recogniser and keeping the higher-confidence result per line.
- Default model source is **Hugging Face** (`PADDLE_PDX_MODEL_SOURCE`), with BOS,
  ModelScope and AIStudio fallbacks. It runs a connectivity check unless
  `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`. Our provider sets that flag and
  **always passes explicit local model directories**, refusing to start if they are
  missing — no implicit download.
- Basic OCR pipeline returns text lines, polygons and scores; it does **not** do
  table-structure recognition. `PP-StructureV3` / `table_recognition_v2` pipelines
  exist in this version but are out of scope for the first slice (UNVERIFIED).

### Surya 0.22.1 (SOURCE-VERIFIED, not run with weights)
- **Architecture changed significantly vs. older Surya:** text recognition
  ("Surya2") is now **VLM-backed via an external inference server** —
  vLLM (Docker image, GPU) or a `llama-server` binary (llama.cpp) spawned as a
  subprocess. CPU-only use means a llama.cpp server.
- Layout + reading order ("fast layout", RF-DETR) runs on CPU in a shared local
  server process; checkpoints default to `hf://datalab-to/surya_layout2`, detection
  and OCR-error models to `s3://...` on `models.datalab.to`. All are overridable by
  env vars (`FAST_LAYOUT_MODEL_CHECKPOINT`, `DETECTOR_MODEL_CHECKPOINT`, ...) to local paths.
- **Model-weight license: UNVERIFIED** (model cards on HF unreachable). Older Surya
  weights used a modified AI Pubs Open RAIL-M license with commercial-use revenue
  limits — must be checked before any commercial use.
- Per the task's scope control, Surya routing is **deferred** until the first
  slice passes; only an interface stub and availability check are provided.

### llama-cpp-python 0.3.36 + Qwen2.5-VL (SOURCE-VERIFIED, not run with weights)
- Builds from source on CPU; bundles `libmtmd` (llama.cpp multimodal library).
- Provides `Qwen25VLChatHandler(clip_model_path=<mmproj.gguf>)`.
- Image input: `image_url` must be a `data:` URI — any other URL is fetched with
  `urllib.request.urlopen`, i.e. a network code path. **Our provider only ever sends
  base64 `data:` URIs of locally rendered pages.**
- JSON-schema constrained decoding via `response_format={"type":"json_object","schema":...}`
  is implemented in the handler (grammar from JSON schema).
- llama.cpp docs (`docs/multimodal.md`, fetched from GitHub) list
  `ggml-org/Qwen2.5-VL-{3B,7B,32B,72B}-Instruct-GGUF` and `ggml-org/Qwen2-VL-{2B,7B}-Instruct-GGUF`
  as supported.
- **Selected candidate:** `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF`, quantisation
  **Q4_K_M**, plus its **mmproj (f16)** projector file. Exact file names, sizes and
  checksums are **UNVERIFIED** (HF blocked) — `models/README.md` lists them as
  “to confirm at provisioning time”.
- **Qwen licenses: UNVERIFIED here.** From prior knowledge (must be re-checked on the
  model cards): 7B is Apache-2.0; **3B is under the Qwen Research License
  (non-commercial)**; 72B under the Qwen License. Do not ship the 3B for commercial use
  without checking.
- CPU feasibility: a 7B Q4_K_M model needs ~5–6 GB RAM; per-page latency on 4 CPU
  cores is expected in the tens of seconds to minutes (UNVERIFIED; measure with the
  benchmark script). This is why VLM is Level 3 only, per field/page.

## 5. Licensing summary / redistribution risks

| Component | License | Risk |
|---|---|---|
| PyMuPDF | AGPL-3.0 / commercial | **High.** Distributing an application that links PyMuPDF requires AGPL compliance (source of the whole app for network users too) or an Artifex commercial license. The PDF handler is isolated behind an interface so a permissively licensed backend (e.g. pypdfium2, Apache/BSD) could replace it. |
| pymupdf-layout / pymupdf4llm | not declared in metadata | Unknown; not used; installed transitively by pymupdf 2.0. Check with Artifex. |
| PaddlePaddle / PaddleOCR / PaddleX | Apache-2.0 | Low. Official PP-OCRv5 weights: Apache-2.0 per PaddleOCR repo (UNVERIFIED for each model card). |
| Surya code | Apache-2.0 (0.22.1) | Weights: UNVERIFIED, historically restricted. |
| llama-cpp-python | MIT | Low. |
| Qwen2.5-VL weights | size-dependent (see above) | **Must check per model.** |
| torch | BSD-style | Low; CUDA libs carry NVIDIA EULA terms. |

## 6. Implementation plan (smallest vertical slice first)

1. `schemas.py`, `exceptions.py`, `config.py` — stable result schema (pydantic v2),
   explicit field states (`extracted`, `missing`, `unreadable`, `explicitly_absent`,
   `conflict`), local model path config, offline flags.
2. `pdf_handler.py` + `core/direct_text_handler.py` — native text/words with page
   refs; text-quality scoring (empty, garbage, reversed Arabic, presentation forms,
   broken glyph maps); rendering at configurable DPI.
3. `core/paddle_handler.py` — PaddleOCR adapter with explicit model dirs and
   missing-model errors; optional subprocess worker (`core/worker.py`).
4. `validators.py` — numbers (Arabic-Indic digits, `٫`/`,`/`.` decimal separators),
   units, coordinate tables, cross-provider conflicts. No CRS/regulatory inference.
5. `core/vlm_handler.py` — llama-cpp + Qwen2.5-VL, data-URI only, JSON-schema
   output, malformed-output retries, evidence corroboration against OCR/native text.
6. `router.py` — Level 1/2/3 with thresholds, per-page escalation, retry caps,
   `review_required` when uncertain.
7. Tests with synthetic fixtures + fake providers; real-model tests marked
   `requires_models` and skipped when weights are absent. Offline test with a socket
   guard (all OSes) and `unshare -n` script (Linux).
8. `benchmarks/evaluate.py` + report template.
9. Surya (layout/reading order) only after 1–8 pass — stub for now.

## 7. Additional findings during implementation (VERIFIED with PyMuPDF 2.0)

- `page.get_text("words")` splits Arabic words at non-joining letters (`ا لرخصة`),
  even with `TEXT_INHIBIT_SPACES`. Words are therefore built from `rawdict`
  characters, splitting only on real whitespace glyphs.
- MuPDF's logical reordering of RTL lines reversed an Arabic-Indic number
  (`١٢٥٠٫٥` → `٥٫٠٥٢١`). Purely numeric words are re-ordered by glyph x-position
  (numbers are always laid out left-to-right).
- In RTL paragraphs, `2024-03-05` is split into separate bidi runs/lines; fields are
  therefore read from *visual rows* (words grouped by vertical overlap, ordered
  right-to-left for Arabic-dominant rows), not from MuPDF lines.
- A font with a broken ToUnicode map produced CJK characters inside Arabic words
  (`ا㒡رخصة`); this is detected as `broken_glyph_mapping_suspected`.
- `page.get_cdrawings()` exists and is used as a cheap vector-density signal.
- llama-cpp-python 0.3.36: `llama_grammar.json_schema_to_gbnf` accepts the VLM output
  schema (VERIFIED in the vlm environment).
- Subprocess workers: core-venv parent → paddle-venv and vlm-venv workers start, and
  errors propagate with their type (VERIFIED, without model weights).

## 8. Test status at time of writing

Ran `pytest` in three separate Python 3.13 environments: core (`pymupdf` only),
paddle (`paddleocr 3.7.0`) and vlm (`llama-cpp-python 0.3.36`). See the README for how to run the tests.

| Environment | Passed | Skipped | Skip reasons |
|---|---|---|---|
| core | 90 | 3 | 2 × no model weights; 1 × `llama_cpp` not installed (grammar test) |
| paddle | 90 | 3 | same as core |
| vlm | 91 | 2 | 2 × no model weights |

The Linux network-namespace test (`unshare -rn`) ran and passed in all three.
Real-model tests (`-m requires_models`) were **skipped** everywhere: no weights were available.
