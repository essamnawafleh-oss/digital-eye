"""Download the OCR models once (needs internet), then everything runs offline.

    python scripts/download_models.py [--models-dir DIR] [--with-vlm]

Default DIR: $DPC_MODELS_DIR, else ./models_local (git-ignored).
Downloads from Hugging Face (verified 2026-10-24 with huggingface_hub, all Apache-2.0):
  PaddlePaddle/PP-OCRv5_mobile_det        ~4.8 MB  text detection
  PaddlePaddle/arabic_PP-OCRv5_mobile_rec ~7.9 MB  Arabic(+Latin, digits) recognition
  PaddlePaddle/en_PP-OCRv5_mobile_rec     ~7.8 MB  Latin/digit cross-check recognition
--with-vlm also fetches the optional vision model (~6 GB, NOT verified end-to-end):
  ggml-org/Qwen2.5-VL-7B-Instruct-GGUF  (*Q4_K_M*.gguf + mmproj*.gguf)
Afterwards it writes manifest.json with SHA-256 checksums (scripts/verify_models.py).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PADDLE = ["PP-OCRv5_mobile_det", "arabic_PP-OCRv5_mobile_rec", "en_PP-OCRv5_mobile_rec"]
VLM_REPO = "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=os.environ.get("DPC_MODELS_DIR") or "models_local")
    ap.add_argument("--with-vlm", action="store_true")
    args = ap.parse_args(argv)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is missing: pip install huggingface_hub")
        return 1
    root = Path(args.models_dir).resolve()
    for name in PADDLE:
        target = root / "paddle" / name
        print(f"downloading PaddlePaddle/{name} -> {target}")
        snapshot_download(repo_id=f"PaddlePaddle/{name}", local_dir=str(target))
        if not (target / "inference.yml").is_file() or not (target / "inference.pdiparams").is_file():
            print(f"ERROR: {name} is incomplete (inference.yml / inference.pdiparams missing); run again")
            return 1
    if args.with_vlm:
        target = root / "vlm"
        print(f"downloading {VLM_REPO} (Q4_K_M + mmproj, ~6 GB) -> {target}")
        snapshot_download(repo_id=VLM_REPO, local_dir=str(target), allow_patterns=["*Q4_K_M*.gguf", "mmproj*.gguf"])
        print("VLM files:", sorted(p.name for p in target.glob("*.gguf")),
              "- set vlm.model_file / vlm.mmproj_file to these names if they differ from the defaults")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.verify_models import main as verify  # noqa: E402
    print(f"\nmodels are in {root}\nuse: DPC_MODELS_DIR={root}  or  options={{'models_dir': r'{root}'}}")
    return verify(["--models-dir", str(root), "--write"] + (["--require-vlm"] if args.with_vlm else []))


if __name__ == "__main__":
    sys.exit(main())
