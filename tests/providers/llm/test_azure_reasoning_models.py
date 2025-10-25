"""Tests for Azure reasoning model support."""

from unittest.mock import Mock, patch

import pytest

from esperanto.providers.llm.azure import AzureLanguageModel


@pytest.fixture
def azure_config():
    """Azure configuration for testing."""
    return {
        "api_key": "test-key",
        "config": {
            "azure_endpoint": "https://test.openai.azure.com/",
            "api_version": "2024-02-01",
        }
    }


class TestAzureReasoningModels:
    """Test Azure reasoning model support."""

    @pytest.mark.parametrize("model_name", ["o1-preview", "o1-mini", "o3-mini", "o4-mini", "gpt-5"])
    def test_is_reasoning_model_detection(self, azure_config, model_name):
        """Test reasoning model detection."""
        with patch('httpx.Client'), patch('httpx.AsyncClient'):
            model = AzureLanguageModel(model_name=model_name, **azure_config)
            assert model._is_reasoning_model() is True

    @pytest.mark.parametrize("model_name", ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"])
    def test_is_not_reasoning_model_detection(self, azure_config, model_name):
        """Test non-reasoning model detection."""
        with patch('httpx.Client'), patch('httpx.AsyncClient'):
            model = AzureLanguageModel(model_name=model_name, **azure_config)
            assert model._is_reasoning_model() is False

    def test_reasoning_model_api_kwargs(self, azure_config):
        """Test API kwargs for reasoning models."""
        with patch('httpx.Client'), patch('httpx.AsyncClient'):
            model = AzureLanguageModel(
                model_name="o1-preview",
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                **azure_config
            )

            api_kwargs = model._get_api_kwargs()

            # For reasoning models, should use max_completion_tokens
            assert "max_completion_tokens" in api_kwargs
            assert api_kwargs["max_completion_tokens"] == 1000
            assert "max_tokens" not in api_kwargs

            # Should not include temperature or top_p for reasoning models
            assert "temperature" not in api_kwargs
            assert "top_p" not in api_kwargs

            # Other params should still be present
            assert api_kwargs["model"] == "o1-preview"
            assert "stream" in api_kwargs

    def test_reasoning_model_api_kwargs_default_max_tokens(self, azure_config):
        """Test API kwargs for reasoning models with default max_tokens."""
        with patch('httpx.Client'), patch('httpx.AsyncClient'):

            model = AzureLanguageModel(
                model_name="o1-preview",
                max_tokens=850,  # Default value
                **azure_config
            )

            api_kwargs = model._get_api_kwargs()

            # Should skip max_completion_tokens when using default value
            assert "max_completion_tokens" not in api_kwargs
            assert "max_tokens" not in api_kwargs

    def test_non_reasoning_model_api_kwargs(self, azure_config):
        """Test API kwargs for non-reasoning models."""
        with patch('httpx.Client'), patch('httpx.AsyncClient'):

            model = AzureLanguageModel(
                model_name="gpt-4",
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                **azure_config
            )

            api_kwargs = model._get_api_kwargs()

            # For non-reasoning models, should use max_tokens
            assert "max_tokens" in api_kwargs
            assert api_kwargs["max_tokens"] == 1000
            assert "max_completion_tokens" not in api_kwargs

            # Should include temperature and top_p for non-reasoning models
            assert api_kwargs["temperature"] == 0.7
            assert api_kwargs["top_p"] == 0.9

    def test_reasoning_model_to_langchain(self, azure_config):
        """Test LangChain conversion for reasoning models."""
        with patch('httpx.Client'), \
             patch('httpx.AsyncClient'), \
             patch('langchain_openai.AzureChatOpenAI') as mock_azure_chat, \
             patch.object(AzureLanguageModel, '_clean_config', side_effect=lambda x: x):

            model = AzureLanguageModel(
                model_name="o1-preview",
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                **azure_config
            )

            model.to_langchain()

            # Verify the call was made with correct kwargs
            call_kwargs = mock_azure_chat.call_args[1]

            # Should use max_completion_tokens in model_kwargs for reasoning models
            assert "model_kwargs" in call_kwargs
            assert "max_completion_tokens" in call_kwargs["model_kwargs"]
            assert call_kwargs["model_kwargs"]["max_completion_tokens"] == 1000
            assert "max_tokens" not in call_kwargs

            # Should set specific values for reasoning models
            assert call_kwargs["temperature"] == 1
            # top_p should not be present (filtered out since it's None)
            assert "top_p" not in call_kwargs

    def test_reasoning_model_to_langchain_default_max_tokens(self, azure_config):
        """Test LangChain conversion for reasoning models with default max_tokens."""
        with patch('httpx.Client'), \
             patch('httpx.AsyncClient'), \
             patch('langchain_openai.AzureChatOpenAI') as mock_azure_chat, \
             patch.object(AzureLanguageModel, '_clean_config', side_effect=lambda x: x):

            model = AzureLanguageModel(
                model_name="o1-preview",
                max_tokens=850,  # Default value
                temperature=0.7,
                top_p=0.9,
                **azure_config
            )

            model.to_langchain()

            # Verify the call was made with correct kwargs
            call_kwargs = mock_azure_chat.call_args[1]

            # Should not include max_completion_tokens when using default value
            if "model_kwargs" in call_kwargs:
                assert "max_completion_tokens" not in call_kwargs["model_kwargs"]
            assert "max_tokens" not in call_kwargs

            # Should set specific values for reasoning models
            assert call_kwargs["temperature"] == 1
            # top_p should not be present (filtered out since it's None)
            assert "top_p" not in call_kwargs

    def test_non_reasoning_model_to_langchain(self, azure_config):
        """Test LangChain conversion for non-reasoning models."""
        with patch('httpx.Client'), \
             patch('httpx.AsyncClient'), \
             patch('langchain_openai.AzureChatOpenAI') as mock_azure_chat, \
             patch.object(AzureLanguageModel, '_clean_config', side_effect=lambda x: x):

            model = AzureLanguageModel(
                model_name="gpt-4",
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                **azure_config
            )

            model.to_langchain()

            # Verify the call was made with correct kwargs
            call_kwargs = mock_azure_chat.call_args[1]

            # Should use max_tokens for non-reasoning models
            assert "max_tokens" in call_kwargs
            assert call_kwargs["max_tokens"] == 1000
            assert "max_completion_tokens" not in call_kwargs

            # Should preserve original values for non-reasoning models
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["top_p"] == 0.9