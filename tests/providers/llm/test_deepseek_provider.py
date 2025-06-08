import os
from unittest.mock import patch

import pytest

from esperanto.providers.llm.deepseek import DeepSeekLanguageModel


def test_provider_name():
    model = DeepSeekLanguageModel(api_key="test-key")
    assert model.provider == "deepseek"


def test_client_properties():
    model = DeepSeekLanguageModel(api_key="test-key")
    assert model.client is not None
    assert model.async_client is not None
    # Check HTTP client properties
    assert hasattr(model.client, "post")
    assert hasattr(model.async_client, "post")
    assert hasattr(model.client, "get")
    assert hasattr(model.async_client, "get")


def test_initialization_with_api_key():
    model = DeepSeekLanguageModel(api_key="test-key")
    assert model.api_key == "test-key"


def test_initialization_with_env_var():
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-test-key"}):
        model = DeepSeekLanguageModel()
        assert model.api_key == "env-test-key"


def test_initialization_without_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="DeepSeek API key not found"):
            DeepSeekLanguageModel()


def test_default_model_name():
    model = DeepSeekLanguageModel(api_key="test-key")
    assert model.model_name == "deepseek-chat"


def test_to_langchain():
    model = DeepSeekLanguageModel(api_key="test-key")
    lc = model.to_langchain()
    # Should be ChatDeepSeek or compatible
    assert lc is not None
