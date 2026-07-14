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
    response = model.chat_complete(PROMPT, max_tokens=100)
    _assert_capital(response)


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not configured"
)
def test_anthropic_structured_output_real():
    model = AIFactory.create_language(
        "anthropic", "claude-3-5-haiku-latest", config=STRUCTURED_CONFIG
    )
    response = model.chat_complete(PROMPT, max_tokens=100)
    _assert_capital(response)


@pytest.mark.release
@pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="GOOGLE_API_KEY or GEMINI_API_KEY not configured",
)
def test_google_structured_output_real():
    model = AIFactory.create_language(
        "google", "gemini-2.0-flash", config=STRUCTURED_CONFIG
    )
    response = model.chat_complete(PROMPT, max_tokens=100)
    _assert_capital(response)


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not configured"
)
def test_groq_structured_output_real():
    model = AIFactory.create_language(
        "groq", "llama-3.3-70b-versatile", config=STRUCTURED_CONFIG
    )
    response = model.chat_complete(PROMPT, max_tokens=100)
    _assert_capital(response)


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("MISTRAL_API_KEY"), reason="MISTRAL_API_KEY not configured"
)
def test_mistral_structured_output_real():
    model = AIFactory.create_language(
        "mistral", "mistral-large-latest", config=STRUCTURED_CONFIG
    )
    response = model.chat_complete(PROMPT, max_tokens=100)
    _assert_capital(response)


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("COHERE_API_KEY"), reason="COHERE_API_KEY not configured"
)
def test_cohere_structured_output_real():
    model = AIFactory.create_language(
        "cohere", "command-a-03-2025", config=STRUCTURED_CONFIG
    )
    response = model.chat_complete(PROMPT, max_tokens=100)
    _assert_capital(response)
