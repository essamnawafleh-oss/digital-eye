"""Configuration: local model paths, routing thresholds and offline behaviour.

Precedence (lowest to highest): defaults < environment variables < `options` passed
to `parse_document`. Model paths are only ever *local*; nothing here can make the
package download a model.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import ConfigError

ENV_MODELS_DIR = "DPC_MODELS_DIR"

# Environment flags that stop third-party libraries from reaching model hubs.
# Verified names: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE (huggingface_hub, transformers),
# PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK (paddlex/utils/flags.py, v3.7.2).
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
}


def apply_offline_env() -> None:
    for key, value in OFFLINE_ENV.items():
        os.environ[key] = value


@dataclass
class PaddleConfig:
    enabled: bool = True
    lang: str = "ar"
    # Recogniser: what paddleocr 3.7.0 maps lang="ar" to. Detector: the mobile model
    # instead of the server one paddleocr picks by default. Measured 2026-10-24 on a
    # 200-dpi A4 page, 4 CPU cores: mobile 2.0 s / 690 MB RSS, server 20 s / 4.0 GB,
    # identical text on the synthetic permit page. Set "PP-OCRv5_server_det" to compare.
    det_model_name: str = "PP-OCRv5_mobile_det"
    rec_model_name: str = "arabic_PP-OCRv5_mobile_rec"
    det_model_dir: str | None = None  # default: <models_dir>/paddle/<det_model_name>
    rec_model_dir: str | None = None  # default: <models_dir>/paddle/<rec_model_name>
    # Second recogniser that re-reads every line containing Latin letters or digits.
    # Lines without Arabic use it as the primary reading; elsewhere its Latin/digit
    # tokens must agree with the Arabic model's, or the value is flagged for review.
    # Benchmark 2026-10-24: the Arabic model misread digits in Arabic lines
    # ('رقم القطعة: 128' -> '18', score 0.94); this model read all such cases right.
    # None disables the cross-check.
    cross_check_rec_model_name: str | None = "en_PP-OCRv5_mobile_rec"
    # In Arabic lines, take Latin/digit tokens from the Latin model when its line score
    # is at least this (see paddle_handler.merge_line_readings for the measurements).
    latin_merge_min_score: float = 0.80
    cross_check_rec_model_dir: str | None = None
    device: str = "cpu"
    # oneDNN (MKL-DNN) CPU acceleration. Off by default: with paddlepaddle 3.3.1 on CPU,
    # PP-OCRv5 detection crashed with "ConvertPirAttribute2RuntimeAttribute not support"
    # inside onednn_instruction.cc (reproduced 2026-10-24).
    enable_mkldnn: bool = False
    # Detection input size. Without a cap, the server detector tried to allocate 20 GB
    # for one 200-dpi page. "max" + 960 limits the longer image side.
    det_limit_type: str = "max"
    det_limit_side_len: int = 960
    # Run in a separate interpreter (e.g. a dedicated venv). None = in-process.
    python_executable: str | None = None
    worker_timeout_s: float = 600.0


@dataclass
class VLMConfig:
    enabled: bool = True
    backend: str = "llama_cpp"
    chat_handler: str = "qwen2.5-vl"
    model_path: str | None = None  # default: <models_dir>/vlm/<model_file>
    mmproj_path: str | None = None  # default: <models_dir>/vlm/<mmproj_file>
    # Candidate files (names UNVERIFIED until provisioning, see models/README.md).
    model_file: str = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
    mmproj_file: str = "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"
    n_ctx: int = 8192
    n_threads: int | None = None
    n_gpu_layers: int = 0
    max_tokens: int = 1024
    temperature: float = 0.0
    max_image_side_px: int = 1600
    max_retries: int = 1  # extra attempts after malformed output
    python_executable: str | None = None
    worker_timeout_s: float = 1800.0


@dataclass
class RoutingConfig:
    # Level 1 text-layer quality thresholds
    min_chars_per_page: int = 25
    max_garbage_ratio: float = 0.10
    max_suspicious_token_ratio: float = 0.15
    reversed_arabic_threshold: float = 0.6
    min_image_coverage_for_scan: float = 0.5
    min_drawings_for_vector_page: int = 500
    # OCR thresholds (PaddleOCR rec_scores are model scores, not calibrated probabilities)
    min_ocr_confidence: float = 0.80  # below: value is "unreadable" and not returned
    # Between min_ocr_confidence and this: value returned but flagged for review.
    # Chosen on the synthetic benchmark of 2026-10-24 (seed 2026): at 0.90 it caught
    # 8/14 wrong OCR values for 5/169 correct ones sent to review.
    review_ocr_confidence: float = 0.90
    # Escalation
    max_level: int = 3  # 1 = fast only, 2 = + layout/table, 3 = + VLM
    max_vlm_pages: int = 3
    max_vlm_fields_per_call: int = 12
    vlm_for_conflicts: bool = True


@dataclass
class ParserConfig:
    models_dir: str | None = None
    render_dpi: int = 200
    offline: bool = True
    # Install a socket guard during parsing that raises on any non-local connection.
    strict_network_guard: bool = False
    include_page_text: bool = True
    extract_tables: bool | None = None  # None = only if the document type expects tables
    vlm_for_optional_fields: bool = False
    trace_python_memory: bool = False
    date_order: str = "DMY"  # DMY or MDY, used for ambiguous numeric dates
    decimal_separator: str = "auto"  # auto | "." | ","
    # Square metres per dunum. None = do not convert (the dunum is not universal).
    dunum_m2: float | None = None
    ocr_provider: str | None = "paddle"  # "paddle" or None
    paddle: PaddleConfig = field(default_factory=PaddleConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)

    # ---- construction -------------------------------------------------
    @classmethod
    def from_env(cls) -> "ParserConfig":
        cfg = cls()
        if os.environ.get(ENV_MODELS_DIR):
            cfg.models_dir = os.environ[ENV_MODELS_DIR]
        return cfg

    @classmethod
    def from_options(cls, options: dict | None) -> "ParserConfig":
        options = dict(options or {})
        base = options.pop("config", None)
        if isinstance(base, ParserConfig):
            cfg = dataclasses.replace(base)
        elif base is None:
            cfg = cls.from_env()
        else:
            raise ConfigError("options['config'] must be a ParserConfig instance")
        # Non-config options handled elsewhere (fields, providers for tests, etc.)
        for key in ("fields", "providers"):
            options.pop(key, None)
        _merge(cfg, options, "options")
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not 36 <= self.render_dpi <= 600:
            raise ConfigError(f"render_dpi must be in [36, 600], got {self.render_dpi}")
        if self.date_order not in ("DMY", "MDY"):
            raise ConfigError("date_order must be 'DMY' or 'MDY'")
        if self.decimal_separator not in ("auto", ".", ","):
            raise ConfigError("decimal_separator must be 'auto', '.' or ','")
        if self.routing.max_level not in (1, 2, 3):
            raise ConfigError("routing.max_level must be 1, 2 or 3")
        if self.ocr_provider not in (None, "paddle"):
            raise ConfigError("ocr_provider must be 'paddle' or None")

    # ---- resolved model paths ----------------------------------------
    def _models_root(self) -> Path | None:
        return Path(self.models_dir).expanduser() if self.models_dir else None

    def paddle_det_dir(self) -> Path | None:
        return _resolve(self.paddle.det_model_dir, self._models_root(), "paddle", self.paddle.det_model_name)

    def paddle_rec_dir(self) -> Path | None:
        return _resolve(self.paddle.rec_model_dir, self._models_root(), "paddle", self.paddle.rec_model_name)

    def paddle_cross_check_dir(self) -> Path | None:
        name = self.paddle.cross_check_rec_model_name
        if not name:
            return None
        return _resolve(self.paddle.cross_check_rec_model_dir, self._models_root(), "paddle", name)

    def vlm_model_path(self) -> Path | None:
        return _resolve(self.vlm.model_path, self._models_root(), "vlm", self.vlm.model_file)

    def vlm_mmproj_path(self) -> Path | None:
        return _resolve(self.vlm.mmproj_path, self._models_root(), "vlm", self.vlm.mmproj_file)

    def snapshot(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _resolve(explicit: str | None, root: Path | None, sub: str, name: str) -> Path | None:
    if explicit:
        return Path(explicit).expanduser()
    if root is None:
        return None
    return root / sub / name


def _merge(obj: Any, updates: dict, where: str) -> None:
    valid = {f.name: f for f in dataclasses.fields(obj)}
    for key, value in updates.items():
        if key not in valid:
            raise ConfigError(f"Unknown option '{where}.{key}'. Valid: {sorted(valid)}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            sub = dataclasses.replace(current)
            _merge(sub, value, f"{where}.{key}")
            setattr(obj, key, sub)
        else:
            setattr(obj, key, value)
