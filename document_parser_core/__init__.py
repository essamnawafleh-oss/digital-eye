"""document_parser_core: offline-first parsing of engineering documents, permits,
plans, coordinate tables, invoices and scans (PDF, PNG, JPEG)."""

__version__ = "0.1.0"

from .main import parse_document, parse_document_model  # noqa: E402
from .config import ParserConfig  # noqa: E402
from .schemas import SCHEMA_VERSION, result_json_schema  # noqa: E402

__all__ = ["parse_document", "parse_document_model", "ParserConfig", "SCHEMA_VERSION", "result_json_schema", "__version__"]
