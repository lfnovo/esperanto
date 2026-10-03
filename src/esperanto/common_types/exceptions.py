"""Exceptions for Esperanto common types."""

from typing import List, Optional


class EsperantoError(Exception):
    """Base class for all Esperanto-raised errors.

    This is the root of Esperanto's normalized exception hierarchy. Catching
    ``EsperantoError`` catches any error the library raises deliberately (as
    opposed to a raw provider/SDK exception). More specific error types are
    built on top of this root — see issue #227 for the full hierarchy.
    """


class ProviderCapabilityError(EsperantoError):
    """Raised when a provider is asked for a modality it does not support.

    For example, requesting an embedding model from an OpenAI-compatible profile
    that only declares ``language`` support.
    """


class EmptyCompletionError(EsperantoError):
    """Raised when structured output was requested but the model returned no content.

    Typically the output budget was consumed by reasoning before any text was
    produced (``finish_reason='length'``), or the model refused
    (``finish_reason='content_filter'``).

    Attributes:
        model: Name of the model that returned no content.
        finish_reason: Normalized finish reason reported for the choice.
    """

    def __init__(self, model: str, finish_reason: Optional[str] = None):
        self.model = model
        self.finish_reason = finish_reason
        message = (
            f"Model '{model}' returned no content for a structured output request "
            f"(finish_reason={finish_reason!r})."
        )
        if finish_reason == "length":
            message += (
                " The output budget may have been consumed by reasoning; "
                "increase max_tokens."
            )
        elif finish_reason == "content_filter":
            message += " The model refused or its output was filtered."
        super().__init__(message)


class ToolCallValidationError(Exception):
    """Raised when tool call arguments fail JSON schema validation.

    Attributes:
        tool_name: Name of the tool that failed validation.
        errors: List of validation error messages.
    """

    def __init__(self, tool_name: str, errors: List[str]):
        self.tool_name = tool_name
        self.errors = errors
        error_msg = "; ".join(errors)
        super().__init__(f"Tool '{tool_name}' validation failed: {error_msg}")


class StructuredOutputValidationError(Exception):
    """Raised when schema-driven structured output validation fails.

    Attributes:
        schema_name: Name of the schema that failed validation.
        errors: List of validation error messages.
    """

    def __init__(self, schema_name: str, errors: List[str]):
        self.schema_name = schema_name
        self.errors = errors
        error_msg = "; ".join(errors)
        super().__init__(f"Structured output '{schema_name}' validation failed: {error_msg}")
