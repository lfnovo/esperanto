"""Cross-provider parity tests for structured output.

The point of this module is the *public contract*: the SAME structured config
(``{"type": "json_schema", "schema": Capital}``) must yield a uniformly-typed
``response.structured`` — a validated ``Capital`` instance — across every HTTP
provider, regardless of each provider's wildly different native request/response
shape. Everything is mocked; no real API calls are made.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from esperanto.providers.llm.anthropic import AnthropicLanguageModel
from esperanto.providers.llm.azure import AzureLanguageModel
from esperanto.providers.llm.cohere import CohereLanguageModel
from esperanto.providers.llm.google import GoogleLanguageModel
from esperanto.providers.llm.groq import GroqLanguageModel
from esperanto.providers.llm.mistral import MistralLanguageModel
from esperanto.providers.llm.ollama import OllamaLanguageModel
from esperanto.providers.llm.openai import OpenAILanguageModel
from esperanto.providers.llm.openai_compatible import OpenAICompatibleLanguageModel
from esperanto.providers.llm.openrouter import OpenRouterLanguageModel
from esperanto.providers.llm.perplexity import PerplexityLanguageModel
from esperanto.providers.llm.vertex import VertexLanguageModel


class Capital(BaseModel):
    city: str
    country: str


# The exact JSON string every mocked provider "returns" as its text content.
CAPITAL_JSON = '{"city": "Paris", "country": "France"}'


# --- Native-response builders (one per response shape) --------------------- #


def _openai_style_response(content: str) -> dict:
    """OpenAI-compatible chat completion (openai/azure/groq/mistral/perplexity/openrouter)."""
    return {
        "id": "chatcmpl-parity",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "parity-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _anthropic_response(content: str) -> dict:
    return {
        "id": "msg_parity",
        "content": [{"type": "text", "text": content}],
        "model": "claude-parity",
        "role": "assistant",
        "stop_reason": "end_turn",
        "type": "message",
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }


def _google_response(content: str) -> dict:
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": content}], "role": "model"},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
    }


def _ollama_response(content: str) -> dict:
    return {
        "model": "gemma2",
        "created_at": "2024-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "eval_count": 10,
        "prompt_eval_count": 5,
    }


def _cohere_response(content: str) -> dict:
    return {
        "id": "chat-parity",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": content}],
        },
        "finish_reason": "COMPLETE",
        "usage": {"tokens": {"input_tokens": 12, "output_tokens": 8}},
    }


# --- Provider builders (construct + mock the sync HTTP client) ------------- #


def _mock_client(response_data: dict) -> Mock:
    client = Mock()
    resp = Mock()
    resp.status_code = 200
    resp.json.return_value = response_data
    client.post.return_value = resp
    return client


def _build_openai(response_data):
    model = OpenAILanguageModel(api_key="test-key", model_name="gpt-4o")
    model.client = _mock_client(response_data)
    model.async_client = AsyncMock()
    return model


def _build_azure(response_data):
    model = AzureLanguageModel(
        model_name="test-deployment",
        api_key="test-key",
        config={
            "azure_endpoint": "https://test.openai.azure.com/",
            "api_version": "2023-12-01-preview",
        },
    )
    model.client = _mock_client(response_data)
    model.async_client = AsyncMock()
    return model


def _build_groq(response_data):
    model = GroqLanguageModel(api_key="test-key", model_name="llama-3.1-8b-instant")
    model.client = _mock_client(response_data)
    model.async_client = AsyncMock()
    return model


def _build_mistral(response_data):
    model = MistralLanguageModel(api_key="test-key", model_name="mistral-large-latest")
    model.client = _mock_client(response_data)
    model.async_client = AsyncMock()
    return model


def _build_perplexity(response_data):
    model = PerplexityLanguageModel(api_key="test-key", model_name="sonar")
    model.client = _mock_client(response_data)
    model.async_client = AsyncMock()
    return model


def _build_openrouter(response_data):
    model = OpenRouterLanguageModel(api_key="test-key", model_name="openai/gpt-4")
    model.client = _mock_client(response_data)
    model.async_client = AsyncMock()
    return model


def _build_anthropic(response_data):
    model = AnthropicLanguageModel(api_key="test-key", model_name="claude-3-opus-20240229")
    model.client = _mock_client(response_data)
    model.async_client = AsyncMock()
    return model


def _build_google(response_data):
    model = GoogleLanguageModel(api_key="test-key", model_name="gemini-2.0-flash")
    model.client = _mock_client(response_data)
    model.async_client = AsyncMock()
    return model


def _build_ollama(response_data):
    # Patch the ollama SDK clients so construction never touches the network.
    with patch("ollama.Client"), patch("ollama.AsyncClient"):
        model = OllamaLanguageModel(model_name="gemma2")
    model.client = _mock_client(response_data)
    model.async_client = AsyncMock()
    return model


def _build_cohere(response_data):
    with patch.object(CohereLanguageModel, "_create_http_clients", lambda self: None):
        model = CohereLanguageModel(model_name="command-a-03-2025", api_key="test-key", config={})
    model.client = _mock_client(response_data)
    model.async_client = AsyncMock()
    return model


def _build_openai_compatible(response_data):
    model = OpenAICompatibleLanguageModel(
        api_key="test-key",
        base_url="http://localhost:8080/v1",
        model_name="test-model",
    )
    model.client = _mock_client(response_data)
    model.async_client = AsyncMock()
    return model


def _build_vertex(response_data):
    with patch.object(
        VertexLanguageModel,
        "_load_credentials",
        lambda self: setattr(self, "_credentials", None),
    ):
        model = VertexLanguageModel(
            model_name="gemini-2.0-flash",
            vertex_project="test-project",
            vertex_location="us-central1",
        )
    # Shadow token retrieval persistently (calls happen during chat_complete).
    model._get_access_token = lambda: "tok"
    model.client = _mock_client(response_data)
    model.async_client = AsyncMock()
    return model


# (name, builder, native_response_dict)
PARITY_PROVIDERS = [
    ("openai", _build_openai, _openai_style_response(CAPITAL_JSON)),
    ("azure", _build_azure, _openai_style_response(CAPITAL_JSON)),
    ("groq", _build_groq, _openai_style_response(CAPITAL_JSON)),
    ("mistral", _build_mistral, _openai_style_response(CAPITAL_JSON)),
    ("perplexity", _build_perplexity, _openai_style_response(CAPITAL_JSON)),
    ("openrouter", _build_openrouter, _openai_style_response(CAPITAL_JSON)),
    ("openai_compatible", _build_openai_compatible, _openai_style_response(CAPITAL_JSON)),
    ("anthropic", _build_anthropic, _anthropic_response(CAPITAL_JSON)),
    ("google", _build_google, _google_response(CAPITAL_JSON)),
    ("vertex", _build_vertex, _google_response(CAPITAL_JSON)),
    ("ollama", _build_ollama, _ollama_response(CAPITAL_JSON)),
    ("cohere", _build_cohere, _cohere_response(CAPITAL_JSON)),
]


@pytest.mark.parametrize(
    "provider_name, builder, response_data",
    PARITY_PROVIDERS,
    ids=[p[0] for p in PARITY_PROVIDERS],
)
def test_structured_output_parity(provider_name, builder, response_data):
    """Same structured config → validated ``Capital`` instance for every provider."""
    model = builder(response_data)
    model.structured = {"type": "json_schema", "schema": Capital}

    response = model.chat_complete(
        [{"role": "user", "content": "What is the capital of France?"}]
    )

    # The public contract is identical everywhere.
    assert isinstance(response.structured, Capital), (
        f"{provider_name}: expected Capital, got {type(response.structured)!r}"
    )
    assert response.structured.city == "Paris"
    assert response.structured.country == "France"


def test_structured_output_parity_multi_choice():
    """n>1: every choice carries its own validated instance; top-level is choice[0]."""
    two_choice_response = {
        "id": "chatcmpl-multi",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": CAPITAL_JSON},
                "finish_reason": "stop",
            },
            {
                "index": 1,
                "message": {"role": "assistant", "content": CAPITAL_JSON},
                "finish_reason": "stop",
            },
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }
    model = _build_openai(two_choice_response)
    model.structured = {"type": "json_schema", "schema": Capital}

    response = model.chat_complete([{"role": "user", "content": "Capital of France?"}])

    assert len(response.choices) == 2
    assert isinstance(response.choices[0].message.structured, Capital)
    assert isinstance(response.choices[1].message.structured, Capital)
    # Top-level ``structured`` is sourced from the first choice.
    assert response.structured is response.choices[0].message.structured
    assert response.choices[1].message.structured.city == "Paris"
