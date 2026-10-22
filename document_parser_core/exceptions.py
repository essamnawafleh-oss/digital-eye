"""Exception hierarchy. Every error carries an actionable message."""

from __future__ import annotations


class DocumentParserError(Exception):
    """Base class for all errors raised by document_parser_core."""

    code = "parser_error"


class ConfigError(DocumentParserError):
    code = "config_error"


class InputFileError(DocumentParserError):
    """The input file is missing, unreadable, encrypted, corrupted or unsupported."""

    code = "input_file_error"


class ProviderUnavailableError(DocumentParserError):
    """A provider's Python dependency is not installed in this environment."""

    code = "provider_unavailable"

    def __init__(self, provider: str, detail: str, hint: str = ""):
        self.provider = provider
        msg = f"Provider '{provider}' is unavailable: {detail}"
        if hint:
            msg += f" -> {hint}"
        super().__init__(msg)


class ModelNotFoundError(DocumentParserError):
    """A required local model file or directory is missing. Never triggers a download."""

    code = "model_not_found"

    def __init__(self, provider: str, path: str, what: str, hint: str = ""):
        self.provider = provider
        self.path = path
        msg = (
            f"Provider '{provider}': required local model {what} not found at '{path}'. "
            "Models are never downloaded during parsing. "
        )
        msg += hint or "Provision the model files first (see document_parser_core/models/README.md)."
        super().__init__(msg)


class ProviderError(DocumentParserError):
    """A provider failed at runtime."""

    code = "provider_error"


class MalformedModelOutputError(ProviderError):
    """A model returned output that does not match the requested schema."""

    code = "malformed_model_output"


class OfflineViolationError(DocumentParserError):
    """A network connection was attempted while the strict offline guard was active."""

    code = "offline_violation"
