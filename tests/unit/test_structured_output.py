"""Unit tests for the shared structured-output machinery.

Covers the resolver (`resolve_structured_output`), the content parser
(`parse_structured_output_content`), the response wiring helper
(`apply_structured_output`), and the unsupported-error detector
(`is_json_schema_unsupported_error`) in isolation from any provider.
"""

import pytest
from pydantic import BaseModel

from esperanto.common_types import (
    ChatCompletion,
    Choice,
    FunctionCall,
    Message,
    StructuredOutputValidationError,
    ToolCall,
)
from esperanto.providers.llm.structured_output import (
    ResolvedStructuredOutput,
    apply_structured_output,
    is_json_schema_unsupported_error,
    parse_structured_output_content,
    resolve_structured_output,
)


class Capital(BaseModel):
    city: str
    country: str


DICT_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
    "required": ["city", "country"],
}


def _completion(content=None, tool_calls=None, n=1):
    """Build a ChatCompletion with n choices, all sharing content/tool_calls."""
    choices = [
        Choice(
            index=i,
            message=Message(role="assistant", content=content, tool_calls=tool_calls),
            finish_reason="stop",
        )
        for i in range(n)
    ]
    return ChatCompletion(
        id="c1", choices=choices, model="m", provider="p", created=1
    )


# --------------------------------------------------------------------------- #
# resolve_structured_output                                                    #
# --------------------------------------------------------------------------- #

def test_resolve_none_returns_none():
    assert resolve_structured_output(None) is None


def test_resolve_json_string_alias_enabled():
    resolved = resolve_structured_output("json", allow_string_json_alias=True)
    assert resolved is not None
    assert resolved.mode == "json_object"
    assert resolved.is_schema_mode is False
    assert resolved.response_format == {"type": "json_object"}


def test_resolve_json_string_alias_disabled_raises():
    with pytest.raises(TypeError):
        resolve_structured_output("json")


def test_resolve_arbitrary_string_raises():
    with pytest.raises(TypeError):
        resolve_structured_output("nope", allow_string_json_alias=True)


def test_resolve_non_dict_raises():
    with pytest.raises(TypeError):
        resolve_structured_output(123)


@pytest.mark.parametrize("type_value", ["json", "json_object"])
def test_resolve_json_object_modes(type_value):
    resolved = resolve_structured_output({"type": type_value})
    assert resolved.mode == "json_object"
    assert resolved.is_schema_mode is False
    assert resolved.schema_source is None


def test_resolve_json_schema_pydantic():
    resolved = resolve_structured_output({"type": "json_schema", "schema": Capital})
    assert resolved.is_schema_mode is True
    assert resolved.schema_source is Capital
    assert resolved.schema_name == "Capital"  # defaults to class name
    js = resolved.response_format["json_schema"]
    assert js["name"] == "Capital"
    assert js["strict"] is True
    assert js["schema"]["type"] == "object"


def test_resolve_json_schema_dict_defaults_name():
    resolved = resolve_structured_output({"type": "json_schema", "schema": DICT_SCHEMA})
    assert resolved.is_schema_mode is True
    assert resolved.schema_source == DICT_SCHEMA
    assert resolved.schema_name == "structured_output"


def test_resolve_json_schema_requires_schema():
    with pytest.raises(ValueError):
        resolve_structured_output({"type": "json_schema"})


def test_resolve_custom_name_used():
    resolved = resolve_structured_output(
        {"type": "json_schema", "schema": Capital, "name": "my_cap"}
    )
    assert resolved.schema_name == "my_cap"


@pytest.mark.parametrize("bad_name", ["", "   ", "has spaces", "a" * 65, "bad!chars"])
def test_resolve_invalid_name_raises(bad_name):
    with pytest.raises(ValueError):
        resolve_structured_output(
            {"type": "json_schema", "schema": Capital, "name": bad_name}
        )


@pytest.mark.parametrize("bad_strict", ["true", 1, None])
def test_resolve_non_bool_strict_raises(bad_strict):
    with pytest.raises((ValueError, TypeError)):
        resolve_structured_output(
            {"type": "json_schema", "schema": Capital, "strict": bad_strict}
        )


def test_resolve_unknown_type_raises():
    with pytest.raises(TypeError):
        resolve_structured_output({"type": "xml"})


# --------------------------------------------------------------------------- #
# parse_structured_output_content                                              #
# --------------------------------------------------------------------------- #

def test_parse_not_schema_mode_returns_none():
    resolved = resolve_structured_output({"type": "json_object"})
    assert parse_structured_output_content('{"a": 1}', resolved) is None


def test_parse_pydantic_valid():
    resolved = resolve_structured_output({"type": "json_schema", "schema": Capital})
    parsed = parse_structured_output_content(
        '{"city": "Paris", "country": "France"}', resolved
    )
    assert isinstance(parsed, Capital)
    assert parsed.city == "Paris"


def test_parse_dict_schema_valid_returns_dict():
    resolved = resolve_structured_output({"type": "json_schema", "schema": DICT_SCHEMA})
    parsed = parse_structured_output_content(
        '{"city": "Rome", "country": "Italy"}', resolved
    )
    assert parsed == {"city": "Rome", "country": "Italy"}
    assert not isinstance(parsed, BaseModel)


def test_parse_invalid_json_raises():
    resolved = resolve_structured_output({"type": "json_schema", "schema": Capital})
    with pytest.raises(StructuredOutputValidationError):
        parse_structured_output_content("not json", resolved)


def test_parse_pydantic_validation_failure_raises():
    resolved = resolve_structured_output({"type": "json_schema", "schema": Capital})
    with pytest.raises(StructuredOutputValidationError):
        parse_structured_output_content('{"city": "Paris"}', resolved)  # missing country


def test_parse_dict_schema_validation_failure_raises():
    resolved = resolve_structured_output({"type": "json_schema", "schema": DICT_SCHEMA})
    with pytest.raises(StructuredOutputValidationError):
        parse_structured_output_content('{"city": "Paris"}', resolved)  # missing country


# --------------------------------------------------------------------------- #
# apply_structured_output                                                      #
# --------------------------------------------------------------------------- #

def test_apply_no_op_when_not_schema_mode():
    resolved = resolve_structured_output({"type": "json_object"})
    result = _completion(content='{"city": "Paris", "country": "France"}')
    out = apply_structured_output(result, resolved)
    assert out.structured is None


def test_apply_no_op_when_resolved_none():
    result = _completion(content="hello")
    assert apply_structured_output(result, None) is result


def test_apply_populates_message_and_response_property():
    resolved = resolve_structured_output({"type": "json_schema", "schema": Capital})
    result = _completion(content='{"city": "Paris", "country": "France"}')
    out = apply_structured_output(result, resolved)
    # Source of truth on the message
    assert isinstance(out.choices[0].message.structured, Capital)
    # Response-level property mirrors choices[0]
    assert out.structured is out.choices[0].message.structured
    assert out.structured.city == "Paris"


def test_apply_tool_calls_guard_leaves_structured_none():
    resolved = resolve_structured_output({"type": "json_schema", "schema": Capital})
    tc = [ToolCall(id="1", type="function", function=FunctionCall(name="f", arguments="{}"))]
    result = _completion(content="", tool_calls=tc)
    out = apply_structured_output(result, resolved)
    # Guard: tool-call response is not parsed, no crash
    assert out.structured is None


def test_apply_multi_choice_parses_each():
    resolved = resolve_structured_output({"type": "json_schema", "schema": Capital})
    result = _completion(content='{"city": "Paris", "country": "France"}', n=3)
    out = apply_structured_output(result, resolved)
    assert len(out.choices) == 3
    for choice in out.choices:
        assert isinstance(choice.message.structured, Capital)
        assert choice.message.structured.city == "Paris"
    # Top-level property surfaces the first choice
    assert out.structured is out.choices[0].message.structured


def test_apply_invalid_json_raises():
    resolved = resolve_structured_output({"type": "json_schema", "schema": Capital})
    result = _completion(content="not json")
    with pytest.raises(StructuredOutputValidationError):
        apply_structured_output(result, resolved)


# --------------------------------------------------------------------------- #
# is_json_schema_unsupported_error                                             #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "msg,expected",
    [
        ("response_format json_schema is not supported", True),
        ("json_schema unsupported by this model", True),
        ("response_format.type must be 'text'", False),  # no json_schema token
        ("json_schema type must be 'text'", True),
        ("HTTP 500: Internal Server Error", False),
        ("invalid json_schema: extra field", False),  # json_schema but no unsupported pattern
    ],
)
def test_is_json_schema_unsupported_error(msg, expected):
    assert is_json_schema_unsupported_error(RuntimeError(msg)) is expected


def test_resolved_is_schema_mode_property():
    assert ResolvedStructuredOutput("json_schema", {}).is_schema_mode is True
    assert ResolvedStructuredOutput("json_object", {}).is_schema_mode is False
