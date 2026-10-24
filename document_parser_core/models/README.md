# Local models

Model weights are **never** stored in this repository and **never** downloaded during
parsing. Download them once (needs internet), then everything runs offline:

```bash
pip install -r document_parser_core/requirements-paddle.txt   # includes huggingface_hub
python scripts/download_models.py --models-dir /opt/dpc-models # ~20 MB, writes manifest.json
export DPC_MODELS_DIR=/opt/dpc-models                          # Windows: set DPC_MODELS_DIR=C:\dpc-models
```
or pass `options={"models_dir": "/opt/dpc-models"}` to `parse_document`.

## Layout

```
$DPC_MODELS_DIR/
├── paddle/
│   ├── PP-OCRv5_mobile_det/          # text detection           4.8 MB
│   ├── arabic_PP-OCRv5_mobile_rec/   # Arabic + Latin + digits  7.9 MB
│   └── en_PP-OCRv5_mobile_rec/       # Latin/digit cross-check  7.8 MB
│       (each: inference.yml + inference.pdiparams [checked], inference.json, config.json)
├── vlm/                              # OPTIONAL (level 3), ~6 GB, not verified end-to-end
│   ├── Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
│   └── mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf
└── manifest.json                     # SHA-256 of every file (scripts/verify_models.py)
```

Every path can be overridden (`paddle.det_model_dir`, `paddle.rec_model_dir`,
`paddle.cross_check_rec_model_dir`, `vlm.model_path`, `vlm.mmproj_path`).

## Why these models (measured 2026-10-24, 4 CPU cores, no GPU)

| Choice | Evidence |
|---|---|
| `PP-OCRv5_mobile_det` instead of the server detector paddleocr picks for `lang="ar"` | Same text on the test page; 2.0 s vs 20 s per page; 690 MB vs 4.0 GB RSS |
| `arabic_PP-OCRv5_mobile_rec` | What paddleocr 3.7.0 maps `lang="ar"` to; its dictionary (747 chars) includes Latin letters, ASCII and Arabic-Indic digits |
| `en_PP-OCRv5_mobile_rec` as a second reader | The Arabic model dropped digits/letters in Latin runs inside Arabic lines (`128` → `18`, `BP-2025/9070` → `BP-2025/907`) with high scores; the English model read all those cases correctly. Disagreements are flagged for review; it is also the primary reader for lines without Arabic |
| oneDNN off (`paddle.enable_mkldnn=False`) | With paddlepaddle 3.3.1 it crashed on both detectors (`ConvertPirAttribute2RuntimeAttribute not support`) |
| `det_limit_side_len=960` | Without it the server detector tried to allocate 20 GB for one 200-dpi page |

Benchmark details: `docs/BENCHMARK_RESULTS.md`.

## Licences (checked on the downloaded model cards, 2026-10-24)

| Model | Licence |
|---|---|
| PP-OCRv5_mobile_det, arabic_PP-OCRv5_mobile_rec, en_PP-OCRv5_mobile_rec | **Apache-2.0** (model card front matter) |
| Qwen2.5-VL GGUF (optional) | not checked: model card not downloaded. From prior knowledge: 7B Apache-2.0, 3B non-commercial |

## Optional vision model

`python scripts/download_models.py --with-vlm` adds the Qwen2.5-VL-7B GGUF files (~6 GB)
and prints their exact names; set `vlm.model_file` / `vlm.mmproj_file` if they differ
from the defaults. It needs `requirements-vlm.txt` (compiles llama.cpp). This path has
**not** been run with real weights yet.
