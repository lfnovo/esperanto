"""Tests for OpenAI-compatible language model provider."""

import os
from unittest.mock import Mock, patch

import pytest

from esperanto.providers.llm.openai_compatible import OpenAICompatibleLanguageModel


class TestOpenAICompatibleLanguageModel:
    """Test suite for OpenAI-compatible language model."""

    def test_provider_name(self):
        """Test that provider name is correctly returned."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:1234"
        )
        assert model.provider == "openai-compatible"

    def test_initialization_with_direct_params(self):
        """Test initialization with direct parameters."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:1234"
        )
        assert model.api_key == "test-key"
        assert model.base_url == "http://localhost:1234"

    def test_initialization_with_env_vars(self):
        """Test initialization with environment variables."""
        with patch.dict(os.environ, {
            "OPENAI_COMPATIBLE_API_KEY": "env-test-key",
            "OPENAI_COMPATIBLE_BASE_URL": "http://env-localhost:8080"
        }):
            model = OpenAICompatibleLanguageModel()
            assert model.api_key == "env-test-key"
            assert model.base_url == "http://env-localhost:8080"

    def test_initialization_with_config(self):
        """Test initialization with config dictionary."""
        config = {
            "api_key": "config-key",
            "base_url": "http://config-localhost:9090"
        }
        model = OpenAICompatibleLanguageModel(config=config)
        assert model.api_key == "config-key"
        assert model.base_url == "http://config-localhost:9090"

    def test_configuration_precedence(self):
        """Test that configuration precedence works correctly."""
        # Factory config should override environment variables
        with patch.dict(os.environ, {
            "OPENAI_COMPATIBLE_API_KEY": "env-key",
            "OPENAI_COMPATIBLE_BASE_URL": "http://env-localhost:8080"
        }):
            config = {
                "api_key": "config-key",
                "base_url": "http://config-localhost:9090"
            }
            model = OpenAICompatibleLanguageModel(config=config)
            assert model.api_key == "config-key"
            assert model.base_url == "http://config-localhost:9090"

    def test_initialization_without_base_url(self):
        """Test that initialization fails without base URL."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI-compatible base URL is required"):
                OpenAICompatibleLanguageModel(api_key="test-key")

    def test_initialization_without_api_key(self):
        """Test that initialization succeeds without API key using default value."""
        with patch.dict(os.environ, {}, clear=True):
            model = OpenAICompatibleLanguageModel(base_url="http://localhost:1234")
            assert model.api_key == "not-required"

    def test_initialization_without_both_required_params(self):
        """Test that initialization fails without base URL (only base URL is required)."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI-compatible base URL is required"):
                OpenAICompatibleLanguageModel()

    def test_base_url_trailing_slash_removal(self):
        """Test that trailing slashes are removed from base URL."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:1234/"
        )
        assert model.base_url == "http://localhost:1234"

    def test_base_url_multiple_trailing_slashes(self):
        """Test that multiple trailing slashes are removed."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:1234///"
        )
        assert model.base_url == "http://localhost:1234"

    def test_get_default_model(self):
        """Test that default model is returned correctly."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:1234"
        )
        assert model._get_default_model() == "gpt-3.5-turbo"

    def test_get_api_kwargs(self):
        """Test that API kwargs are returned correctly."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:1234",
            temperature=0.7,
            max_tokens=1000
        )
        kwargs = model._get_api_kwargs()
        assert "temperature" in kwargs
        assert "max_tokens" in kwargs
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 1000

    def test_get_api_kwargs_exclude_stream(self):
        """Test that streaming is excluded when requested."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:1234",
            streaming=True
        )
        kwargs = model._get_api_kwargs(exclude_stream=True)
        assert "streaming" not in kwargs
        assert "stream" not in kwargs

    def test_models_property_success(self):
        """Test successful models property call."""
        # Mock the HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "test-model-1",
                    "owned_by": "test-provider",
                    "context_window": 4096
                },
                {
                    "id": "test-model-2",
                    "owned_by": "test-provider"
                }
            ]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.get.return_value = mock_response
        
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:1234"
        )
        model.client = mock_client_instance
        
        models = model.models
        assert len(models) == 2
        assert models[0].id == "test-model-1"
        assert models[0].owned_by == "test-provider"
        assert models[0].context_window == 4096
        assert models[1].id == "test-model-2"
        assert models[1].owned_by == "test-provider"

    def test_models_property_failure(self):
        """Test models property graceful failure."""
        # Mock the HTTP response to raise an exception
        mock_client_instance = Mock()
        mock_client_instance.get.side_effect = Exception("Network error")
        
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:1234"
        )
        model.client = mock_client_instance
        
        # Should return empty list on failure
        models = model.models
        assert models == []

    def test_handle_error_openai_format(self):
        """Test error handling with OpenAI-format error response."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:1234"
        )
        
        # Mock response with OpenAI-format error
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {
                "message": "Invalid request format",
                "type": "invalid_request_error"
            }
        }
        
        with pytest.raises(RuntimeError, match="OpenAI-compatible endpoint error: Invalid request format"):
            model._handle_error(mock_response)

    def test_handle_error_http_only(self):
        """Test error handling with HTTP-only error response."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:1234"
        )
        
        # Mock response with non-JSON error
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.json.side_effect = Exception("Not JSON")
        
        with pytest.raises(RuntimeError, match="OpenAI-compatible endpoint error: HTTP 500: Internal Server Error"):
            model._handle_error(mock_response)

    def test_handle_error_success_response(self):
        """Test that successful responses don't raise errors."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:1234"
        )
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        
        # Should not raise any exception
        model._handle_error(mock_response)

    def test_langchain_integration(self):
        """Test LangChain integration."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:1234",
            model_name="test-model",
            temperature=0.5,
            max_tokens=500
        )
        
        # Mock the langchain import inside the method
        with patch('langchain_openai.ChatOpenAI') as mock_chat_openai:
            mock_instance = Mock()
            mock_chat_openai.return_value = mock_instance
            
            result = model.to_langchain()
            
            # Verify ChatOpenAI was called with correct parameters
            mock_chat_openai.assert_called_once()
            call_args = mock_chat_openai.call_args[1]
            assert call_args["api_key"] == "test-key"
            assert call_args["base_url"] == "http://localhost:1234"
            assert call_args["model"] == "test-model"
            assert call_args["temperature"] == 0.5
            assert call_args["max_tokens"] == 500
            assert result == mock_instance

    def test_langchain_integration_with_structured_output(self):
        """Test LangChain integration with structured output."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:1234",
            structured={"type": "json"}
        )
        
        with patch('langchain_openai.ChatOpenAI') as mock_chat_openai:
            model.to_langchain()
            
            call_args = mock_chat_openai.call_args[1]
            assert "model_kwargs" in call_args
            assert call_args["model_kwargs"]["response_format"] == {"type": "json_object"}

    def test_langchain_integration_reasoning_model(self):
        """Test LangChain integration with reasoning model (o1)."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:1234",
            model_name="o1-preview",
            max_tokens=1000,
            temperature=0.7
        )
        
        with patch('langchain_openai.ChatOpenAI') as mock_chat_openai:
            model.to_langchain()
            
            call_args = mock_chat_openai.call_args[1]
            # For reasoning models, max_tokens becomes max_completion_tokens
            assert "max_completion_tokens" in call_args
            assert "max_tokens" not in call_args
            assert call_args["max_completion_tokens"] == 1000
            # Temperature should be set to 1 for reasoning models
            assert call_args["temperature"] == 1
            # top_p should be None for reasoning models
            assert call_args.get("top_p") is None

    def test_langchain_integration_import_error(self):
        """Test LangChain integration with missing import."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:1234"
        )
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'langchain_openai'")):
            with pytest.raises(ImportError, match="Langchain integration requires langchain_openai"):
                model.to_langchain()