# Local models

Model weights are **never** stored in this repository and **never** downloaded during
parsing. Provision them once, on a machine with internet access, then copy the folder
to the offline machine and point the parser at it:

```bash
export DPC_MODELS_DIR=/opt/dpc-models          # Windows: set DPC_MODELS_DIR=C:\dpc-models
# or per call: parse_document(path, options={"models_dir": "/opt/dpc-models"})
```

## Expected layout

```
$DPC_MODELS_DIR/
├── paddle/
│   ├── PP-OCRv5_server_det/          # text detection
│   │   ├── inference.yml             # required (checked)
│   │   ├── inference.pdiparams       # required (checked)
│   │   └── inference.json            # program file shipped with PaddleX 3.x models
│   └── arabic_PP-OCRv5_mobile_rec/   # Arabic recognition (lang="ar" in paddleocr 3.7.0)
│       ├── inference.yml
│       ├── inference.pdiparams
│       └── inference.json
├── vlm/
│   ├── Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf        # language model (name UNVERIFIED, see below)
│   └── mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf    # multimodal projector (name UNVERIFIED)
└── manifest.json                     # written by scripts/verify_models.py --write
```

Every path can also be set explicitly (`paddle.det_model_dir`, `paddle.rec_model_dir`,
`vlm.model_path`, `vlm.mmproj_path`), e.g. to use a different quantisation.

## What is verified and what is not

| Item | Status |
|---|---|
| PaddleOCR model names for `lang="ar"` | Source-verified in paddleocr 3.7.0 (`_get_ocr_model_names`) |
| PaddleX model dir needs `inference.yml` | Source-verified (`paddlex/inference/models/utils/model_config.py`) |
| PaddleX hosts: HF repo `PaddlePaddle/<model_name>`; BOS `.../<model_name>_infer.tar` | Source-verified (`official_models.py`) |
| Qwen2.5-VL GGUF repo `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF` supported by llama.cpp | Verified in llama.cpp `docs/multimodal.md` |
| Exact GGUF file names, sizes, checksums | **UNVERIFIED** — Hugging Face was unreachable from the dev environment |
| End-to-end inference with any of these weights | **UNVERIFIED** |

## Provisioning (run on an online machine; commands UNVERIFIED in the dev environment)

PaddleOCR — either let PaddleOCR download into its cache once, then copy:

```bash
python - <<'EOF'
from paddleocr import PaddleOCR
PaddleOCR(lang="ar", use_doc_orientation_classify=False, use_doc_unwarping=False,
          use_textline_orientation=False)
EOF
cp -r ~/.paddlex/official_models/PP-OCRv5_server_det        $DPC_MODELS_DIR/paddle/
cp -r ~/.paddlex/official_models/arabic_PP-OCRv5_mobile_rec $DPC_MODELS_DIR/paddle/
```

or download the repos directly:

```bash
huggingface-cli download PaddlePaddle/PP-OCRv5_server_det        --local-dir $DPC_MODELS_DIR/paddle/PP-OCRv5_server_det
huggingface-cli download PaddlePaddle/arabic_PP-OCRv5_mobile_rec --local-dir $DPC_MODELS_DIR/paddle/arabic_PP-OCRv5_mobile_rec
```

VLM — list the repo's files first, choose the Q4_K_M model and the mmproj file, and set
the config to the exact file names you downloaded:

```bash
huggingface-cli download ggml-org/Qwen2.5-VL-7B-Instruct-GGUF --local-dir $DPC_MODELS_DIR/vlm \
    --include "*Q4_K_M*.gguf" "mmproj*.gguf"
```

Then pin what you provisioned and check the layout:

```bash
python scripts/verify_models.py --write     # records SHA-256 of every model file
python scripts/verify_models.py             # later: fails if files are missing or changed
```

## Licences (check every model card before use)

- PP-OCRv5 models: Apache-2.0 per the PaddleOCR project (confirm on each model card).
- Qwen2.5-VL: **licence depends on size** (from prior knowledge, re-check): 7B Apache-2.0,
  3B *Qwen Research License* (non-commercial), 72B *Qwen License*. The 7B is the default
  candidate for that reason.
- Surya weights (deferred provider): licence UNVERIFIED; older releases restricted
  commercial use above a revenue threshold.

## Hardware notes (estimates, not measured)

- PaddleOCR PP-OCRv5 server det + mobile rec on CPU: seconds per page.
- Qwen2.5-VL-7B Q4_K_M: ~5–6 GB RAM; on a 4-core CPU expect tens of seconds to minutes
  per page. Measure with `benchmarks/evaluate.py` before relying on it.
