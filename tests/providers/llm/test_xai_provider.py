import os
from unittest.mock import patch

import pytest

from esperanto.providers.llm.xai import XAILanguageModel


def test_provider_name():
    model = XAILanguageModel(api_key="test-key")
    assert model.provider == "xai"

def test_initialization_with_api_key():
    model = XAILanguageModel(api_key="test-key")
    assert model.api_key == "test-key"
    assert model.base_url == "https://api.x.ai/v1"

def test_initialization_with_env_var():
    with patch.dict(os.environ, {
        "XAI_API_KEY": "env-test-key",
        "XAI_BASE_URL": "https://custom.x.ai/v1"
    }):
        model = XAILanguageModel()
        assert model.api_key == "env-test-key"
        assert model.base_url == "https://custom.x.ai/v1"

def test_initialization_without_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="XAI API key not found"):
            XAILanguageModel()

def test_custom_base_url():
    model = XAILanguageModel(
        api_key="test-key",
        base_url="https://custom.x.ai/v1"
    )
    assert model.base_url == "https://custom.x.ai/v1"
