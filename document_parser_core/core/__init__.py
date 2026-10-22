"""Provider implementations and factories."""

from __future__ import annotations

from ..config import ParserConfig
from .base_handler import LayoutProvider, OCRProvider, Provider, VLMProvider


def create_ocr_provider(config: ParserConfig) -> OCRProvider | None:
    if config.ocr_provider is None or not config.paddle.enabled:
        return None
    if config.paddle.python_executable:
        from .worker import SubprocessOCRProvider
        return SubprocessOCRProvider(config, config.paddle.python_executable)
    from .paddle_handler import PaddleOCRProvider
    return PaddleOCRProvider(config)


def create_vlm_provider(config: ParserConfig) -> VLMProvider | None:
    if not config.vlm.enabled or config.routing.max_level < 3:
        return None
    if config.vlm.python_executable:
        from .worker import SubprocessVLMProvider
        return SubprocessVLMProvider(config, config.vlm.python_executable)
    from .vlm_handler import LlamaCppVLMProvider
    return LlamaCppVLMProvider(config)


__all__ = [
    "LayoutProvider", "OCRProvider", "Provider", "VLMProvider",
    "create_ocr_provider", "create_vlm_provider",
]
