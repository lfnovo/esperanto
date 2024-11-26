import os
from unittest.mock import patch

import pytest

from esperanto.providers.llm.openrouter import OpenRouterLanguageModel


def test_provider_name():
    model = OpenRouterLanguageModel(api_key="test-key")
    assert model.provider == "openrouter"

def test_initialization_with_api_key():
    model = OpenRouterLanguageModel(api_key="test-key")
    assert model.api_key == "test-key"
    assert model.base_url == "https://openrouter.ai/api/v1"

def test_initialization_with_env_var():
    with patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "env-test-key",
        "OPENROUTER_BASE_URL": "https://custom.openrouter.ai/v1"
    }):
        model = OpenRouterLanguageModel()
        assert model.api_key == "env-test-key"
        assert model.base_url == "https://custom.openrouter.ai/v1"

def test_initialization_without_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="OpenRouter API key not found"):
            OpenRouterLanguageModel()

def test_custom_base_url():
    model = OpenRouterLanguageModel(
        api_key="test-key",
        base_url="https://custom.openrouter.ai/v1"
    )
    assert model.base_url == "https://custom.openrouter.ai/v1"
