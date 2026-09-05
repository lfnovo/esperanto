"""Real-API integration tests for schema-driven structured output.

These make REAL API calls and cost money. They are gated behind the ``release``
marker (see ``pyproject.toml`` addopts ``-m 'not release'``) so they NEVER run in
a normal unit/CI run, and each is additionally skipped when the provider's API
key env var is absent.

Run explicitly with: uv run pytest -m release tests/integration/test_structured_output_release.py -v
"""

import os

import pytest
from pydantic import BaseModel

from esperanto import AIFactory


class Capital(BaseModel):
    city: str
    country: str


PROMPT = [
    {
        "role": "user",
        "content": "Return the capital of France as JSON with keys 'city' and 'country'.",
    }
]

STRUCTURED_CONFIG = {"structured": {"type": "json_schema", "schema": Capital}}

# Several current models (gemini-2.5-flash, groq's gpt-oss-120b) spend reasoning
# tokens from the same budget as the answer, so a tight cap truncates the JSON
# mid-string. The failure then reads like a schema problem when it is a budget
# problem — keep the budget generous enough that these tests only fail for real.
MAX_TOKENS = 800


def _assert_capital(response):
    """Every provider must surface a validated ``Capital`` on ``response.structured``."""
    assert isinstance(response.structured, Capital)
    assert response.structured.city
    assert response.structured.country


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not configured"
)
def test_openai_structured_output_real():
    model = AIFactory.create_language(
        "openai", "gpt-4o-mini", config=STRUCTURED_CONFIG
    )
    response = model.chat_complete(PROMPT, max_tokens=MAX_TOKENS)
    _assert_capital(response)


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not configured"
)
def test_anthropic_structured_output_real():
    model = AIFactory.create_language(
        "anthropic", "claude-haiku-4-5-20251001", config=STRUCTURED_CONFIG
    )
    response = model.chat_complete(PROMPT, max_tokens=MAX_TOKENS)
    _assert_capital(response)


@pytest.mark.release
@pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="GOOGLE_API_KEY or GEMINI_API_KEY not configured",
)
def test_google_structured_output_real():
    model = AIFactory.create_language(
        "google", "gemini-2.5-flash", config=STRUCTURED_CONFIG
    )
    response = model.chat_complete(PROMPT, max_tokens=MAX_TOKENS)
    _assert_capital(response)


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not configured"
)
def test_groq_structured_output_real():
    # llama-3.3-70b-versatile does not support response_format json_schema on
    # Groq; gpt-oss-120b does. See https://console.groq.com/docs/structured-outputs
    model = AIFactory.create_language(
        "groq", "openai/gpt-oss-120b", config=STRUCTURED_CONFIG
    )
    response = model.chat_complete(PROMPT, max_tokens=MAX_TOKENS)
    _assert_capital(response)


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("MISTRAL_API_KEY"), reason="MISTRAL_API_KEY not configured"
)
def test_mistral_structured_output_real():
    model = AIFactory.create_language(
        "mistral", "mistral-large-latest", config=STRUCTURED_CONFIG
    )
    response = model.chat_complete(PROMPT, max_tokens=MAX_TOKENS)
    _assert_capital(response)


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("COHERE_API_KEY"), reason="COHERE_API_KEY not configured"
)
def test_cohere_structured_output_real():
    model = AIFactory.create_language(
        "cohere", "command-a-03-2025", config=STRUCTURED_CONFIG
    )
    response = model.chat_complete(PROMPT, max_tokens=MAX_TOKENS)
    _assert_capital(response)


# ---------------------------------------------------------------------------
# OpenAI strict-mode schema shapes
#
# The Capital tests above are the simplest possible schema: flat, all-required.
# These cover the two shapes that used to fail before the strict-mode
# normalization — a nested model (which puts objects under $defs) and a model
# with an optional field (which strict mode forbids outright).
# ---------------------------------------------------------------------------


class Address(BaseModel):
    street: str
    city: str


class Person(BaseModel):
    name: str
    address: Address


class LooseCapital(BaseModel):
    city: str
    nickname: str = "unknown"


NESTED_PROMPT = [
    {
        "role": "user",
        "content": "Invent a person with a name and an address (street and city). Return JSON.",
    }
]


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not configured"
)
def test_openai_structured_output_nested_schema_real():
    """A nested model puts its child under $defs, which also needs the flag."""
    model = AIFactory.create_language(
        "openai", "gpt-4o-mini", config={"structured": {"type": "json_schema", "schema": Person}}
    )
    response = model.chat_complete(NESTED_PROMPT, max_tokens=MAX_TOKENS)

    assert isinstance(response.structured, Person)
    assert response.structured.address.city


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not configured"
)
def test_openai_structured_output_optional_field_real():
    """An optional field can't satisfy strict mode; the call must still work."""
    model = AIFactory.create_language(
        "openai",
        "gpt-4o-mini",
        config={"structured": {"type": "json_schema", "schema": LooseCapital}},
    )
    response = model.chat_complete(PROMPT, max_tokens=MAX_TOKENS)

    assert isinstance(response.structured, LooseCapital)
    assert response.structured.city


@pytest.mark.release
@pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="GOOGLE_API_KEY or GEMINI_API_KEY not configured",
)
def test_google_structured_output_nested_schema_real():
    """Google accepts the same normalized schema the OpenAI family needs."""
    model = AIFactory.create_language(
        "google",
        "gemini-2.5-flash",
        config={
            "api_key": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            "structured": {"type": "json_schema", "schema": Person},
        },
    )
    response = model.chat_complete(NESTED_PROMPT, max_tokens=MAX_TOKENS)

    assert isinstance(response.structured, Person)
    assert response.structured.address.city
