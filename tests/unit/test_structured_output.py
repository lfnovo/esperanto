"""Unit tests for the shared structured-output machinery.

Covers the resolver (`resolve_structured_output`), the content parser
(`parse_structured_output_content`), the response wiring helper
(`apply_structured_output`), and the unsupported-error detector
(`is_json_schema_unsupported_error`) in isolation from any provider.
"""

import copy

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


# ---------------------------------------------------------------------------
# OpenAI strict-mode schema normalization
# ---------------------------------------------------------------------------


class Address(BaseModel):
    street: str
    city: str


class Person(BaseModel):
    name: str
    address: Address
    tags: list[str]


class Loose(BaseModel):
    city: str
    nickname: str = "none"


def test_strict_schema_gets_additional_properties_false():
    """OpenAI rejects strict json_schema without additionalProperties: false."""
    resolved = resolve_structured_output({"type": "json_schema", "schema": Capital})

    schema = resolved.response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert resolved.response_format["json_schema"]["strict"] is True


def test_strict_schema_normalizes_nested_defs():
    """Nested models live in $defs and each needs the flag too."""
    resolved = resolve_structured_output({"type": "json_schema", "schema": Person})

    schema = resolved.response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Address"]["additionalProperties"] is False
    assert resolved.response_format["json_schema"]["strict"] is True


def test_optional_properties_downgrade_strict():
    """Strict also demands every property be required — we don't fabricate that.

    Promoting an optional field to required would change the caller's schema,
    so the request drops to strict=false instead. Local validation is unchanged.
    """
    resolved = resolve_structured_output({"type": "json_schema", "schema": Loose})

    js = resolved.response_format["json_schema"]
    assert js["strict"] is False
    # The flag is still added — it is semantically free and harmless.
    assert js["schema"]["additionalProperties"] is False


def test_explicit_strict_false_is_preserved():
    resolved = resolve_structured_output(
        {"type": "json_schema", "schema": Capital, "strict": False}
    )
    assert resolved.response_format["json_schema"]["strict"] is False


def test_caller_additional_properties_is_not_overwritten():
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": True,
    }
    resolved = resolve_structured_output({"type": "json_schema", "schema": schema})

    assert resolved.response_format["json_schema"]["schema"]["additionalProperties"] is True


def test_caller_schema_dict_is_not_mutated():
    """The caller may reuse their dict — normalization must not leak into it."""
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    original = copy.deepcopy(schema)

    resolve_structured_output({"type": "json_schema", "schema": schema})

    assert schema == original


def test_normalization_applies_to_every_provider_shape():
    """One normalized schema serves every provider, not just the OpenAI family.

    Anthropic's output_config rejects an object schema without
    additionalProperties: false exactly like OpenAI's strict mode does, and the
    other native-schema providers accept it, so there is no per-provider split.
    """
    resolved = resolve_structured_output({"type": "json_schema", "schema": Capital})

    assert resolved.response_format["json_schema"]["schema"]["additionalProperties"] is False
