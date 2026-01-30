"""Tests for tool-related types and validation."""

import json
import pytest

from esperanto.common_types import (
    FunctionCall,
    Message,
    Tool,
    ToolCall,
    ToolCallValidationError,
    ToolFunction,
    find_tool_by_name,
    validate_tool_call,
    validate_tool_calls,
)


class TestFunctionCall:
    """Tests for FunctionCall type."""

    def test_create_function_call(self):
        """Test creating a FunctionCall."""
        fc = FunctionCall(name="get_weather", arguments='{"location": "NYC"}')
        assert fc.name == "get_weather"
        assert fc.arguments == '{"location": "NYC"}'

    def test_function_call_immutable(self):
        """Test that FunctionCall is immutable (frozen)."""
        fc = FunctionCall(name="test", arguments="{}")
        with pytest.raises(Exception):  # Pydantic raises ValidationError for frozen
            fc.name = "other"


class TestToolCall:
    """Tests for ToolCall type."""

    def test_create_tool_call(self):
        """Test creating a ToolCall."""
        tc = ToolCall(
            id="call_123",
            type="function",
            function=FunctionCall(name="get_weather", arguments='{"location": "NYC"}'),
        )
        assert tc.id == "call_123"
        assert tc.type == "function"
        assert tc.function.name == "get_weather"
        assert tc.index is None

    def test_tool_call_with_index(self):
        """Test creating a ToolCall with index for streaming."""
        tc = ToolCall(
            id="call_123",
            type="function",
            function=FunctionCall(name="test", arguments="{}"),
            index=0,
        )
        assert tc.index == 0

    def test_tool_call_default_type(self):
        """Test that type defaults to 'function'."""
        tc = ToolCall(
            id="call_123",
            function=FunctionCall(name="test", arguments="{}"),
        )
        assert tc.type == "function"


class TestToolFunction:
    """Tests for ToolFunction type."""

    def test_create_tool_function(self):
        """Test creating a ToolFunction."""
        tf = ToolFunction(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        )
        assert tf.name == "get_weather"
        assert tf.description == "Get weather for a location"
        assert tf.parameters["type"] == "object"
        assert tf.strict is None

    def test_tool_function_with_strict(self):
        """Test creating a ToolFunction with strict mode."""
        tf = ToolFunction(
            name="test",
            description="Test function",
            parameters={"type": "object", "properties": {}},
            strict=True,
        )
        assert tf.strict is True

    def test_tool_function_default_parameters(self):
        """Test that parameters has a sensible default."""
        tf = ToolFunction(name="test", description="Test function")
        assert tf.parameters == {"type": "object", "properties": {}}


class TestTool:
    """Tests for Tool type."""

    def test_create_tool(self):
        """Test creating a Tool."""
        tool = Tool(
            type="function",
            function=ToolFunction(
                name="get_weather",
                description="Get weather",
                parameters={"type": "object", "properties": {}},
            ),
        )
        assert tool.type == "function"
        assert tool.function.name == "get_weather"

    def test_tool_default_type(self):
        """Test that type defaults to 'function'."""
        tool = Tool(
            function=ToolFunction(name="test", description="Test"),
        )
        assert tool.type == "function"


class TestMessageWithToolCalls:
    """Tests for Message with tool_calls."""

    def test_message_with_typed_tool_calls(self):
        """Test creating a Message with typed ToolCall objects."""
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=FunctionCall(name="get_weather", arguments='{"location": "NYC"}'),
        )
        msg = Message(
            role="assistant",
            content=None,
            tool_calls=[tool_call],
        )
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert isinstance(msg.tool_calls[0], ToolCall)
        assert msg.tool_calls[0].function.name == "get_weather"

    def test_message_with_dict_tool_calls_backward_compat(self):
        """Test that Message accepts dict tool_calls for backward compatibility."""
        msg = Message(
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "NYC"}',
                    },
                }
            ],
        )
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        # Should be converted to ToolCall
        assert isinstance(msg.tool_calls[0], ToolCall)
        assert msg.tool_calls[0].id == "call_123"
        assert msg.tool_calls[0].function.name == "get_weather"

    def test_message_with_empty_tool_calls(self):
        """Test Message with empty tool_calls list."""
        msg = Message(role="assistant", content="Hello", tool_calls=[])
        # Empty list should remain as empty list (not converted to None)
        assert msg.tool_calls == []

    def test_message_with_none_tool_calls(self):
        """Test Message with None tool_calls."""
        msg = Message(role="assistant", content="Hello", tool_calls=None)
        assert msg.tool_calls is None

    def test_message_dict_access_tool_calls(self):
        """Test dict-like access for tool_calls."""
        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="test", arguments="{}"),
        )
        msg = Message(role="assistant", tool_calls=[tool_call])
        assert msg["tool_calls"] is not None
        assert len(msg["tool_calls"]) == 1


class TestValidation:
    """Tests for tool call validation."""

    @pytest.fixture
    def weather_tool(self):
        """Create a sample weather tool."""
        return Tool(
            function=ToolFunction(
                name="get_weather",
                description="Get weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            )
        )

    @pytest.fixture
    def valid_tool_call(self):
        """Create a valid tool call."""
        return ToolCall(
            id="call_123",
            function=FunctionCall(
                name="get_weather",
                arguments='{"location": "NYC", "unit": "celsius"}',
            ),
        )

    def test_validate_tool_call_valid(self, weather_tool, valid_tool_call):
        """Test validation passes for valid tool call."""
        pytest.importorskip("jsonschema")
        # Should not raise
        validate_tool_call(valid_tool_call, weather_tool)

    def test_validate_tool_call_missing_required(self, weather_tool):
        """Test validation fails when required field is missing."""
        pytest.importorskip("jsonschema")
        invalid_call = ToolCall(
            id="call_123",
            function=FunctionCall(
                name="get_weather",
                arguments='{"unit": "celsius"}',  # Missing required 'location'
            ),
        )
        with pytest.raises(ToolCallValidationError) as exc_info:
            validate_tool_call(invalid_call, weather_tool)
        assert "get_weather" in str(exc_info.value)
        assert exc_info.value.tool_name == "get_weather"

    def test_validate_tool_call_invalid_json(self, weather_tool):
        """Test validation fails for invalid JSON arguments."""
        pytest.importorskip("jsonschema")
        invalid_call = ToolCall(
            id="call_123",
            function=FunctionCall(
                name="get_weather",
                arguments="not valid json",
            ),
        )
        with pytest.raises(ToolCallValidationError) as exc_info:
            validate_tool_call(invalid_call, weather_tool)
        assert "Invalid JSON" in str(exc_info.value)

    def test_validate_tool_call_wrong_type(self, weather_tool):
        """Test validation fails for wrong argument type."""
        pytest.importorskip("jsonschema")
        invalid_call = ToolCall(
            id="call_123",
            function=FunctionCall(
                name="get_weather",
                arguments='{"location": 123}',  # Should be string, not int
            ),
        )
        with pytest.raises(ToolCallValidationError):
            validate_tool_call(invalid_call, weather_tool)

    def test_validate_tool_call_invalid_enum(self, weather_tool):
        """Test validation fails for invalid enum value."""
        pytest.importorskip("jsonschema")
        invalid_call = ToolCall(
            id="call_123",
            function=FunctionCall(
                name="get_weather",
                arguments='{"location": "NYC", "unit": "kelvin"}',  # Invalid enum
            ),
        )
        with pytest.raises(ToolCallValidationError):
            validate_tool_call(invalid_call, weather_tool)


class TestValidateToolCalls:
    """Tests for validate_tool_calls function."""

    @pytest.fixture
    def tools(self):
        """Create sample tools."""
        return [
            Tool(
                function=ToolFunction(
                    name="get_weather",
                    description="Get weather",
                    parameters={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                )
            ),
            Tool(
                function=ToolFunction(
                    name="get_time",
                    description="Get current time",
                    parameters={
                        "type": "object",
                        "properties": {"timezone": {"type": "string"}},
                    },
                )
            ),
        ]

    def test_validate_multiple_tool_calls(self, tools):
        """Test validating multiple tool calls."""
        pytest.importorskip("jsonschema")
        tool_calls = [
            ToolCall(
                id="call_1",
                function=FunctionCall(
                    name="get_weather", arguments='{"location": "NYC"}'
                ),
            ),
            ToolCall(
                id="call_2",
                function=FunctionCall(
                    name="get_time", arguments='{"timezone": "UTC"}'
                ),
            ),
        ]
        # Should not raise
        validate_tool_calls(tool_calls, tools)

    def test_validate_unknown_tool(self, tools):
        """Test validation fails for unknown tool."""
        pytest.importorskip("jsonschema")
        tool_calls = [
            ToolCall(
                id="call_1",
                function=FunctionCall(
                    name="unknown_function", arguments="{}"
                ),
            ),
        ]
        with pytest.raises(ToolCallValidationError) as exc_info:
            validate_tool_calls(tool_calls, tools)
        assert "Unknown tool" in str(exc_info.value)


class TestFindToolByName:
    """Tests for find_tool_by_name function."""

    def test_find_existing_tool(self):
        """Test finding an existing tool."""
        tools = [
            Tool(function=ToolFunction(name="tool_a", description="A")),
            Tool(function=ToolFunction(name="tool_b", description="B")),
        ]
        found = find_tool_by_name(tools, "tool_b")
        assert found is not None
        assert found.function.name == "tool_b"

    def test_find_nonexistent_tool(self):
        """Test finding a nonexistent tool returns None."""
        tools = [
            Tool(function=ToolFunction(name="tool_a", description="A")),
        ]
        found = find_tool_by_name(tools, "nonexistent")
        assert found is None

    def test_find_in_empty_list(self):
        """Test finding in empty list returns None."""
        found = find_tool_by_name([], "anything")
        assert found is None


class TestToolCallValidationError:
    """Tests for ToolCallValidationError exception."""

    def test_error_message_format(self):
        """Test that error message is formatted correctly."""
        error = ToolCallValidationError("my_tool", ["error 1", "error 2"])
        assert error.tool_name == "my_tool"
        assert error.errors == ["error 1", "error 2"]
        assert "my_tool" in str(error)
        assert "error 1" in str(error)
        assert "error 2" in str(error)

    def test_single_error(self):
        """Test error with single error message."""
        error = ToolCallValidationError("my_tool", ["single error"])
        assert "single error" in str(error)


class TestToolSerialization:
    """Tests for tool serialization/deserialization."""

    def test_tool_to_dict(self):
        """Test converting Tool to dict."""
        tool = Tool(
            function=ToolFunction(
                name="test",
                description="Test function",
                parameters={"type": "object", "properties": {}},
            )
        )
        d = tool.model_dump()
        assert d["type"] == "function"
        assert d["function"]["name"] == "test"

    def test_tool_call_to_dict(self):
        """Test converting ToolCall to dict."""
        tc = ToolCall(
            id="call_123",
            function=FunctionCall(name="test", arguments='{"a": 1}'),
        )
        d = tc.model_dump()
        assert d["id"] == "call_123"
        assert d["function"]["name"] == "test"
        assert d["function"]["arguments"] == '{"a": 1}'

    def test_message_with_tool_calls_to_dict(self):
        """Test converting Message with tool_calls to dict."""
        msg = Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_123",
                    function=FunctionCall(name="test", arguments="{}"),
                )
            ],
        )
        d = msg.model_dump()
        assert d["tool_calls"] is not None
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["id"] == "call_123"
