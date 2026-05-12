"""Tests for OpenAI-compatible language model provider."""

import os
from unittest.mock import AsyncMock, Mock, patch

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
        """Test LangChain integration with structured output (non-LM Studio port)."""
        # Use port 8080 (not 1234) to test that response_format IS set
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:8080",
            structured={"type": "json"}
        )

        with patch('langchain_openai.ChatOpenAI') as mock_chat_openai:
            model.to_langchain()

            call_args = mock_chat_openai.call_args[1]
            assert "model_kwargs" in call_args
            assert call_args["model_kwargs"]["response_format"] == {"type": "json_object"}

    def test_langchain_integration_lmstudio_skips_response_format(self):
        """Test LangChain integration skips response_format for LM Studio (port 1234)."""
        # Port 1234 is the default LM Studio port - response_format should be skipped
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:1234",
            structured={"type": "json"}
        )

        with patch('langchain_openai.ChatOpenAI') as mock_chat_openai:
            model.to_langchain()

            call_args = mock_chat_openai.call_args[1]
            assert "model_kwargs" in call_args
            # response_format should NOT be set for LM Studio
            assert "response_format" not in call_args["model_kwargs"]

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

    def test_provider_specific_env_var_precedence(self):
        """Test that provider-specific env vars take precedence over generic ones."""
        with patch.dict(os.environ, {
            "OPENAI_COMPATIBLE_BASE_URL_LLM": "http://llm-specific:1234",
            "OPENAI_COMPATIBLE_BASE_URL": "http://generic:5678",
            "OPENAI_COMPATIBLE_API_KEY_LLM": "llm-specific-key",
            "OPENAI_COMPATIBLE_API_KEY": "generic-key"
        }):
            model = OpenAICompatibleLanguageModel()
            assert model.base_url == "http://llm-specific:1234"
            assert model.api_key == "llm-specific-key"

    def test_fallback_to_generic_env_var(self):
        """Test fallback to generic env vars when provider-specific ones are not set."""
        with patch.dict(os.environ, {
            "OPENAI_COMPATIBLE_BASE_URL": "http://generic:5678",
            "OPENAI_COMPATIBLE_API_KEY": "generic-key"
        }, clear=True):
            model = OpenAICompatibleLanguageModel()
            assert model.base_url == "http://generic:5678"
            assert model.api_key == "generic-key"

    def test_config_overrides_provider_specific_env_vars(self):
        """Test that config parameters override provider-specific env vars."""
        with patch.dict(os.environ, {
            "OPENAI_COMPATIBLE_BASE_URL_LLM": "http://llm-env:1234",
            "OPENAI_COMPATIBLE_API_KEY_LLM": "llm-env-key"
        }):
            config = {
                "base_url": "http://config:9090",
                "api_key": "config-key"
            }
            model = OpenAICompatibleLanguageModel(config=config)
            assert model.base_url == "http://config:9090"
            assert model.api_key == "config-key"

    def test_direct_params_override_all_env_vars(self):
        """Test that direct parameters override all environment variables."""
        with patch.dict(os.environ, {
            "OPENAI_COMPATIBLE_BASE_URL_LLM": "http://llm-env:1234",
            "OPENAI_COMPATIBLE_BASE_URL": "http://generic-env:5678",
            "OPENAI_COMPATIBLE_API_KEY_LLM": "llm-env-key",
            "OPENAI_COMPATIBLE_API_KEY": "generic-env-key"
        }):
            model = OpenAICompatibleLanguageModel(
                base_url="http://direct:3000",
                api_key="direct-key"
            )
            assert model.base_url == "http://direct:3000"
            assert model.api_key == "direct-key"

    def test_error_message_mentions_both_env_vars(self):
        """Test that error message mentions both provider-specific and generic env vars."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                OpenAICompatibleLanguageModel(api_key="test-key")
            error_message = str(exc_info.value)
            assert "OPENAI_COMPATIBLE_BASE_URL_LLM" in error_message
            assert "OPENAI_COMPATIBLE_BASE_URL" in error_message

    def test_is_likely_lmstudio_port_1234(self):
        """Test that port 1234 is detected as likely LM Studio."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:1234/v1"
        )
        assert model._is_likely_lmstudio() is True

    def test_is_likely_lmstudio_other_port(self):
        """Test that other ports are not detected as LM Studio."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:8080/v1"
        )
        assert model._is_likely_lmstudio() is False

    def test_is_likely_lmstudio_port_12345_not_matched(self):
        """Test that port 12345 is NOT detected as LM Studio (regression test)."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:12345/v1"
        )
        assert model._is_likely_lmstudio() is False

    def test_is_likely_lmstudio_port_12346_not_matched(self):
        """Test that port 12346 is NOT detected as LM Studio (regression test)."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:12346/v1"
        )
        assert model._is_likely_lmstudio() is False

    def test_is_likely_lmstudio_127_0_0_1(self):
        """Test that 127.0.0.1:1234 is detected as likely LM Studio."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://127.0.0.1:1234/v1"
        )
        assert model._is_likely_lmstudio() is True

    def test_response_format_skipped_for_lmstudio(self):
        """Test that response_format is skipped for LM Studio (port 1234)."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:1234/v1",
            structured={"type": "json_object"}
        )
        kwargs = model._get_api_kwargs()
        assert "response_format" not in kwargs

    def test_response_format_included_for_other_ports(self):
        """Test that response_format is included for non-LM Studio endpoints."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:8080/v1",
            structured={"type": "json_object"}
        )
        kwargs = model._get_api_kwargs()
        assert "response_format" in kwargs
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_is_response_format_error(self):
        """Test detection of response_format error message."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:8080/v1"
        )
        # Test the specific error from LM Studio
        error = RuntimeError("'response_format.type' must be 'json_schema' or 'text'")
        assert model._is_response_format_error(error) is True

        # Test other errors
        other_error = RuntimeError("Some other error")
        assert model._is_response_format_error(other_error) is False

    def test_response_format_unsupported_flag(self):
        """Test that _response_format_unsupported flag is properly set."""
        model = OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:8080/v1",
            structured={"type": "json_object"}
        )
        # Initially should be False
        assert model._response_format_unsupported is False

        # After setting the flag, response_format should be skipped
        model._response_format_unsupported = True
        kwargs = model._get_api_kwargs()
        assert "response_format" not in kwargs


def _make_mock_response(content="Hello!"):
    """Build a minimal mock chat completion HTTP response."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "created": 1234567890,
        "model": "test-model",
        "usage": {"completion_tokens": 5, "prompt_tokens": 10, "total_tokens": 15},
    }
    return mock_response


class TestExtraBody:
    """Tests for extra_body passthrough in chat_complete / achat_complete."""

    MESSAGES = [{"role": "user", "content": "hi"}]

    def _model(self, **kwargs):
        return OpenAICompatibleLanguageModel(
            api_key="test-key",
            base_url="http://localhost:8080",
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Sync                                                                 #
    # ------------------------------------------------------------------ #

    def test_instance_extra_body_sent_in_payload(self):
        """Instance-level extra_body keys appear in the outgoing JSON payload."""
        model = self._model(config={"extra_body": {"top_k": 40}})
        mock_resp = _make_mock_response()
        model.client = Mock()
        model.client.post.return_value = mock_resp

        model.chat_complete(self.MESSAGES)

        payload = model.client.post.call_args[1]["json"]
        assert payload["top_k"] == 40
        assert payload["model"] is not None
        assert payload["messages"] == self.MESSAGES

    def test_per_call_extra_body_sent_in_payload(self):
        """Per-call extra_body keys appear in the outgoing JSON payload."""
        model = self._model()
        mock_resp = _make_mock_response()
        model.client = Mock()
        model.client.post.return_value = mock_resp

        model.chat_complete(self.MESSAGES, extra_body={"guided_json": {}})

        payload = model.client.post.call_args[1]["json"]
        assert payload["guided_json"] == {}

    def test_merge_semantics_per_call_wins(self):
        """Per-call extra_body wins on key collision with instance-level extras."""
        model = self._model(config={"extra_body": {"top_k": 40}})
        mock_resp = _make_mock_response()
        model.client = Mock()
        model.client.post.return_value = mock_resp

        model.chat_complete(self.MESSAGES, extra_body={"top_k": 99, "guided_json": {}})

        payload = model.client.post.call_args[1]["json"]
        assert payload["top_k"] == 99
        assert payload["guided_json"] == {}

    def test_none_per_call_uses_only_instance_extras(self):
        """Passing extra_body=None leaves only instance-level extras in payload."""
        model = self._model(config={"extra_body": {"top_k": 40}})
        mock_resp = _make_mock_response()
        model.client = Mock()
        model.client.post.return_value = mock_resp

        model.chat_complete(self.MESSAGES, extra_body=None)

        payload = model.client.post.call_args[1]["json"]
        assert payload["top_k"] == 40

    def test_empty_per_call_uses_only_instance_extras(self):
        """Passing extra_body={} leaves only instance-level extras in payload."""
        model = self._model(config={"extra_body": {"top_k": 40}})
        mock_resp = _make_mock_response()
        model.client = Mock()
        model.client.post.return_value = mock_resp

        model.chat_complete(self.MESSAGES, extra_body={})

        payload = model.client.post.call_args[1]["json"]
        assert payload["top_k"] == 40

    def test_no_extra_body_at_all(self):
        """With no instance or per-call extra_body, payload contains only standard keys."""
        model = self._model()
        mock_resp = _make_mock_response()
        model.client = Mock()
        model.client.post.return_value = mock_resp

        model.chat_complete(self.MESSAGES)

        payload = model.client.post.call_args[1]["json"]
        assert "top_k" not in payload
        assert "guided_json" not in payload

    def test_streaming_inherits_merged_extras(self):
        """Streaming chat_complete includes merged extra_body in the outgoing payload."""
        model = self._model(config={"extra_body": {"top_k": 40}})

        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.iter_text.return_value = iter([
            'data: {"id":"c","choices":[{"index":0,"delta":{"content":"hi","role":"assistant"},"finish_reason":null}],"created":1,"model":"m"}\n',
            "data: [DONE]\n",
        ])
        model.client = Mock()
        model.client.post.return_value = mock_resp

        gen = model.chat_complete(self.MESSAGES, stream=True, extra_body={"guided_json": {}})
        list(gen)  # exhaust generator to ensure post was called

        payload = model.client.post.call_args[1]["json"]
        assert payload["top_k"] == 40
        assert payload["guided_json"] == {}

    # ------------------------------------------------------------------ #
    # Async                                                                #
    # ------------------------------------------------------------------ #

    @pytest.mark.asyncio
    async def test_async_instance_extra_body_sent_in_payload(self):
        """Async: instance-level extra_body keys appear in the outgoing JSON payload."""
        model = self._model(config={"extra_body": {"top_k": 40}})
        mock_resp = _make_mock_response()
        model.async_client = AsyncMock()
        model.async_client.post.return_value = mock_resp

        await model.achat_complete(self.MESSAGES)

        payload = model.async_client.post.call_args[1]["json"]
        assert payload["top_k"] == 40

    @pytest.mark.asyncio
    async def test_async_per_call_extra_body_sent_in_payload(self):
        """Async: per-call extra_body keys appear in the outgoing JSON payload."""
        model = self._model()
        mock_resp = _make_mock_response()
        model.async_client = AsyncMock()
        model.async_client.post.return_value = mock_resp

        await model.achat_complete(self.MESSAGES, extra_body={"guided_json": {}})

        payload = model.async_client.post.call_args[1]["json"]
        assert payload["guided_json"] == {}

    @pytest.mark.asyncio
    async def test_async_merge_semantics_per_call_wins(self):
        """Async: per-call extra_body wins on key collision with instance-level extras."""
        model = self._model(config={"extra_body": {"top_k": 40}})
        mock_resp = _make_mock_response()
        model.async_client = AsyncMock()
        model.async_client.post.return_value = mock_resp

        await model.achat_complete(self.MESSAGES, extra_body={"top_k": 99, "guided_json": {}})

        payload = model.async_client.post.call_args[1]["json"]
        assert payload["top_k"] == 99
        assert payload["guided_json"] == {}

    @pytest.mark.asyncio
    async def test_async_streaming_inherits_merged_extras(self):
        """Async streaming: merged extra_body appears in the outgoing payload."""
        model = self._model(config={"extra_body": {"top_k": 40}})

        mock_resp = Mock()
        mock_resp.status_code = 200

        async def _aiter():
            yield 'data: {"id":"c","choices":[{"index":0,"delta":{"content":"hi","role":"assistant"},"finish_reason":null}],"created":1,"model":"m"}\n'
            yield "data: [DONE]\n"

        mock_resp.aiter_text = _aiter
        model.async_client = AsyncMock()
        model.async_client.post.return_value = mock_resp

        gen = await model.achat_complete(self.MESSAGES, stream=True, extra_body={"guided_json": {}})
        async for _ in gen:
            pass

        payload = model.async_client.post.call_args[1]["json"]
        assert payload["top_k"] == 40
        assert payload["guided_json"] == {}

    def test_extra_body_cannot_override_stream(self):
        """`stream` is reserved — extra_body cannot flip request mode from the response-parsing branch."""
        model = self._model()
        mock_resp = _make_mock_response()
        model.client = Mock()
        model.client.post.return_value = mock_resp

        # Caller asked for non-streaming, but tries to flip via extra_body.
        model.chat_complete(self.MESSAGES, stream=False, extra_body={"stream": True, "top_k": 40})

        payload = model.client.post.call_args[1]["json"]
        assert payload["stream"] is False  # request mode preserved
        assert payload["top_k"] == 40  # benign extras still pass through

    @pytest.mark.asyncio
    async def test_extra_body_cannot_override_stream_async(self):
        """Async counterpart: `stream` is reserved on async path too."""
        model = self._model()
        mock_resp = _make_mock_response()
        model.async_client = AsyncMock()
        model.async_client.post.return_value = mock_resp

        await model.achat_complete(
            self.MESSAGES, stream=False, extra_body={"stream": True, "top_k": 40}
        )

        payload = model.async_client.post.call_args[1]["json"]
        assert payload["stream"] is False
        assert payload["top_k"] == 40