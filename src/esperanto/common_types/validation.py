"""Validation utilities for tool calls."""

import json
from typing import List, Optional

from .exceptions import ToolCallValidationError
from .response import Tool, ToolCall


def validate_tool_call(tool_call: ToolCall, tool: Tool) -> None:
    """Validate tool call arguments against the tool's JSON schema.

    Args:
        tool_call: The tool call from the model response.
        tool: The tool definition with parameters schema.

    Raises:
        ToolCallValidationError: If validation fails.
        ImportError: If jsonschema is not installed.
    """
    try:
        import jsonschema
    except ImportError:
        raise ImportError(
            "jsonschema is required for tool call validation. "
            "Install it with: pip install esperanto[validation]"
        )

    # Parse the arguments JSON
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        raise ToolCallValidationError(
            tool_call.function.name, [f"Invalid JSON in arguments: {e}"]
        )

    # Validate against the schema
    try:
        jsonschema.validate(arguments, tool.function.parameters)
    except jsonschema.ValidationError as e:
        raise ToolCallValidationError(tool_call.function.name, [e.message])


def validate_tool_calls(
    tool_calls: List[ToolCall], tools: List[Tool]
) -> None:
    """Validate multiple tool calls against their corresponding tools.

    Args:
        tool_calls: List of tool calls from the model response.
        tools: List of tool definitions.

    Raises:
        ToolCallValidationError: If any validation fails.
        ImportError: If jsonschema is not installed.
    """
    # Build a lookup dict for tools by name
    tool_lookup = {tool.function.name: tool for tool in tools}

    for tool_call in tool_calls:
        func_name = tool_call.function.name
        if func_name not in tool_lookup:
            raise ToolCallValidationError(
                func_name, [f"Unknown tool: '{func_name}' not in provided tools"]
            )
        validate_tool_call(tool_call, tool_lookup[func_name])


def find_tool_by_name(tools: List[Tool], name: str) -> Optional[Tool]:
    """Find a tool by its function name.

    Args:
        tools: List of tool definitions.
        name: Name of the function to find.

    Returns:
        The matching Tool, or None if not found.
    """
    for tool in tools:
        if tool.function.name == name:
            return tool
    return None
