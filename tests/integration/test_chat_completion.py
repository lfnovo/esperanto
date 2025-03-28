from unittest.mock import AsyncMock, MagicMock

import pytest

from esperanto.providers.llm.anthropic import AnthropicLanguageModel
from esperanto.providers.llm.google import GoogleLanguageModel
from esperanto.providers.llm.openai import OpenAILanguageModel
from esperanto.providers.llm.openrouter import OpenRouterLanguageModel
from esperanto.providers.llm.xai import XAILanguageModel

# Mock responses for each provider
OPENAI_RESPONSE = {"choices": [{"message": {"content": "OpenAI response"}}]}

ANTHROPIC_RESPONSE = {"content": [{"text": "Anthropic response"}]}

GOOGLE_RESPONSE = {
    "candidates": [
        {
            "content": {"parts": [{"text": "Google response"}]},
            "finish_reason": "STOP",
        }
    ]
}


@pytest.fixture
def openai_model():
    model = OpenAILanguageModel(api_key="test-key")
    model.chat_complete = MagicMock(return_value=OPENAI_RESPONSE)
    model.achat_complete = AsyncMock(return_value=OPENAI_RESPONSE)
    return model


@pytest.fixture
def anthropic_model():
    model = AnthropicLanguageModel(api_key="test-key")
    model.chat_complete = MagicMock(return_value=ANTHROPIC_RESPONSE)
    model.achat_complete = AsyncMock(return_value=ANTHROPIC_RESPONSE)
    return model


@pytest.fixture
def openrouter_model():
    model = OpenRouterLanguageModel(api_key="test-key")
    model.chat_complete = MagicMock(return_value=OPENAI_RESPONSE)
    model.achat_complete = AsyncMock(return_value=OPENAI_RESPONSE)
    return model


@pytest.fixture
def xai_model():
    model = XAILanguageModel(api_key="test-key")
    model.chat_complete = MagicMock(return_value=OPENAI_RESPONSE)
    model.achat_complete = AsyncMock(return_value=OPENAI_RESPONSE)
    return model


@pytest.fixture
def google_model():
    model = GoogleLanguageModel(api_key="test-key")
    model.chat_complete = MagicMock(return_value=GOOGLE_RESPONSE)
    model.achat_complete = AsyncMock(return_value=GOOGLE_RESPONSE)
    return model


def test_openai_chat_completion(openai_model):
    messages = [{"role": "user", "content": "Hello!"}]
    response = openai_model.chat_complete(messages)
    assert response == OPENAI_RESPONSE
    openai_model.chat_complete.assert_called_once_with(messages)


@pytest.mark.asyncio
async def test_openai_async_chat_completion(openai_model):
    messages = [{"role": "user", "content": "Hello!"}]
    response = await openai_model.achat_complete(messages)
    assert response == OPENAI_RESPONSE
    openai_model.achat_complete.assert_called_once_with(messages)


def test_anthropic_chat_completion(anthropic_model):
    messages = [{"role": "user", "content": "Hello!"}]
    response = anthropic_model.chat_complete(messages)
    assert response == ANTHROPIC_RESPONSE
    anthropic_model.chat_complete.assert_called_once_with(messages)


@pytest.mark.asyncio
async def test_anthropic_async_chat_completion(anthropic_model):
    messages = [{"role": "user", "content": "Hello!"}]
    response = await anthropic_model.achat_complete(messages)
    assert response == ANTHROPIC_RESPONSE
    anthropic_model.achat_complete.assert_called_once_with(messages)


def test_openrouter_chat_completion(openrouter_model):
    messages = [{"role": "user", "content": "Hello!"}]
    response = openrouter_model.chat_complete(messages)
    assert response == OPENAI_RESPONSE
    openrouter_model.chat_complete.assert_called_once_with(messages)


@pytest.mark.asyncio
async def test_openrouter_async_chat_completion(openrouter_model):
    messages = [{"role": "user", "content": "Hello!"}]
    response = await openrouter_model.achat_complete(messages)
    assert response == OPENAI_RESPONSE
    openrouter_model.achat_complete.assert_called_once_with(messages)


def test_xai_chat_completion(xai_model):
    messages = [{"role": "user", "content": "Hello!"}]
    response = xai_model.chat_complete(messages)
    assert response == OPENAI_RESPONSE
    xai_model.chat_complete.assert_called_once_with(messages)


@pytest.mark.asyncio
async def test_xai_async_chat_completion(xai_model):
    messages = [{"role": "user", "content": "Hello!"}]
    response = await xai_model.achat_complete(messages)
    assert response == OPENAI_RESPONSE
    xai_model.achat_complete.assert_called_once_with(messages)


def test_google_chat_completion(google_model):
    messages = [{"role": "user", "content": "Hello!"}]
    response = google_model.chat_complete(messages)
    assert response == GOOGLE_RESPONSE
    google_model.chat_complete.assert_called_once_with(messages)


@pytest.mark.asyncio
async def test_google_async_chat_completion(google_model):
    messages = [{"role": "user", "content": "Hello!"}]
    response = await google_model.achat_complete(messages)
    assert response == GOOGLE_RESPONSE
    google_model.achat_complete.assert_called_once_with(messages)
