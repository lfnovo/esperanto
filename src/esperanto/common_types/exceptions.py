"""Exceptions for Esperanto common types."""

from typing import List


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
