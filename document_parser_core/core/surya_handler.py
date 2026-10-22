"""Surya layout / reading-order provider — DEFERRED (interface + readiness check only).

Per the task's scope control, Surya is added only after the first vertical slice
passes. Findings for surya-ocr 0.22.1 (source-verified, see docs/COMPATIBILITY_REPORT.md):
* Text recognition is VLM-backed via an external vLLM (Docker/GPU) or llama-server
  process; this package does not use Surya for OCR.
* "Fast layout" (RF-DETR) + reading order run on CPU inside a shared local server
  process. Checkpoints default to `hf://datalab-to/surya_layout2` and must be
  overridden with local paths via FAST_LAYOUT_MODEL_CHECKPOINT /
  FAST_ORDER_MODEL_CHECKPOINT for offline use.
* Do not install it in the same environment as PaddleOCR (OpenCV package collision);
  run it through the subprocess worker instead.
"""

from __future__ import annotations

import os

from ..exceptions import ModelNotFoundError, ProviderUnavailableError
from .base_handler import LayoutProvider


class SuryaLayoutProvider(LayoutProvider):
    name = "surya_layout"

    def check_ready(self) -> None:
        for var in ("FAST_LAYOUT_MODEL_CHECKPOINT", "FAST_ORDER_MODEL_CHECKPOINT"):
            value = os.environ.get(var)
            if not value or value.startswith(("hf://", "s3://", "http")) or not os.path.exists(value):
                raise ModelNotFoundError(
                    "surya", value or "<unset>", f"checkpoint ({var})",
                    f"Set {var} to a local directory; remote checkpoints are not allowed offline.",
                )
        try:
            import surya  # noqa: F401
        except ImportError as exc:
            raise ProviderUnavailableError("surya", str(exc), "see requirements-surya.txt") from exc

    def analyze(self, image, page_number, scale):
        raise NotImplementedError(
            "Surya layout analysis is deferred until the first vertical slice is validated "
            "with real models; see docs/KNOWN_LIMITATIONS.md."
        )
