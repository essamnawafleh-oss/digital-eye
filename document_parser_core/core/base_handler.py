"""Provider interfaces. Providers are swappable without changing `parse_document`.

Coordinates returned by providers are always in the *page* coordinate space
(PDF points for PDF pages, pixels for image inputs); OCR/VLM providers receive the
`scale` that maps rendered-image pixels to page coordinates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from ..schemas import Line


class Provider(ABC):
    name: str = "provider"

    def check_ready(self) -> None:
        """Raise ProviderUnavailableError / ModelNotFoundError if the provider cannot run.

        Must be cheap and must never download anything.
        """

    def close(self) -> None:
        """Release models/processes."""


class OCRProvider(Provider):
    """Recognises text lines (with boxes and scores) on a page image."""

    @abstractmethod
    def recognize(self, image: Image.Image, page_number: int, scale: float) -> list[Line]:
        ...


@dataclass
class VLMFieldAnswer:
    name: str
    status: str  # "found" | "not_found" | "illegible" | "explicitly_absent"
    value: str | None
    evidence_text: str | None
    self_reported_confidence: float | None = None


@dataclass
class VLMResponse:
    answers: dict[str, VLMFieldAnswer]
    raw_output: str
    attempts: int
    model: str
    extra: dict[str, Any] = field(default_factory=dict)


class VLMProvider(Provider):
    """Answers a narrowly scoped extraction task about one page image."""

    @abstractmethod
    def extract_fields(
        self,
        image: Image.Image,
        page_number: int,
        fields: list[dict[str, Any]],
        document_hint: str | None = None,
    ) -> VLMResponse:
        ...


@dataclass
class LayoutRegion:
    label: str  # e.g. "table", "text", "title", "figure"
    bbox: list[float]
    page: int
    reading_order: int | None = None
    confidence: float | None = None


class LayoutProvider(Provider):
    """Layout / reading-order analysis (e.g. Surya). Not used by the first slice."""

    @abstractmethod
    def analyze(self, image: Image.Image, page_number: int, scale: float) -> list[LayoutRegion]:
        ...


# Extension point: a CAD provider (DWG/DXF) would implement a separate interface that
# yields Lines/TableResults directly from vector entities, bypassing rendering + OCR.
