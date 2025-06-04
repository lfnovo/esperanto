import os
from unittest.mock import Mock, patch
import pytest
from esperanto.providers.llm.mistral import MistralLanguageModel

@pytest.fixture
def mock_mistral_client():
    class FakeMessage:
        def __init__(self, content, role):
            self.content = content
            self.role = role
    class FakeChoice:
        def __init__(self):
            self.index = 0
            self.message = FakeMessage("Hello!", "assistant")
            self.finish_reason = "stop"
    class FakeUsage:
        def __init__(self):
            self.prompt_tokens = 2
            self.completion_tokens = 3
            self.total_tokens = 5
    class FakeResponse:
        def __init__(self):
            self.choices = [FakeChoice()]
            self.created = 123
            self.model = "mistral-large-latest"
            self.id = "cmpl-123"
            self.provider = "mistral"
            self.usage = FakeUsage()
    fake_response = FakeResponse()
    client = Mock()
    client.chat_completions.create.return_value = fake_response
    class Chat:
        def complete(self, *args, **kwargs):
            return fake_response
        async def complete_async(self, *args, **kwargs):
            return fake_response
    client.chat = Chat()
    return client

@pytest.fixture
def mistral_model(monkeypatch, mock_mistral_client):
    client = mock_mistral_client
    with patch("esperanto.providers.llm.mistral.Mistral", return_value=client):
        model = MistralLanguageModel(api_key="test-key", model_name="mistral-large-latest")
        model.client = client
        model.async_client = client
        return model

def test_provider_name(mistral_model):
    assert mistral_model.provider == "mistral"

def test_initialization_with_api_key():
    model = MistralLanguageModel(api_key="test-key")
    assert model.api_key == "test-key"

def test_initialization_with_env_var(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "env-test-key")
    model = MistralLanguageModel()
    assert model.api_key == "env-test-key"

def test_initialization_without_api_key(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Mistral API key not found"):
        MistralLanguageModel()

def test_chat_complete(mistral_model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    response = mistral_model.chat_complete(messages)
    assert response.choices[0].message["content"] == "Hello!"

def test_achat_complete(mistral_model):
    import asyncio
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    async def run():
        response = await mistral_model.achat_complete(messages)
        assert response.choices[0].message["content"] == "Hello!"
    asyncio.run(run())

def test_to_langchain(mistral_model):
    # Only run if langchain_mistralai is installed
    try:
        lc = mistral_model.to_langchain()
        assert lc is not None
    except ImportError:
        pytest.skip("langchain_mistralai not installed")
