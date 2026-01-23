"""Exceptions for Esperanto common types."""

from typing import List


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
