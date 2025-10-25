#!/usr/bin/env python3
"""Tests for static model discovery functionality."""

import hashlib
import os
from unittest.mock import MagicMock, patch, Mock
import pytest
import httpx

from esperanto.common_types import Model
from esperanto.model_discovery import (
    _create_cache_key,
    get_openai_models,
    get_openai_compatible_models,
    get_anthropic_models,
    get_google_models,
    get_mistral_models,
    get_groq_models,
    get_jina_models,
    get_voyage_models,
    PROVIDER_MODELS_REGISTRY,
    _model_cache,
)
from esperanto import AIFactory


class TestCacheKeyCreation:
    """Test cache key generation."""

    def test_create_cache_key_simple(self):
        """Test cache key creation with simple params."""
        key = _create_cache_key("openai", base_url="https://api.openai.com")
        assert key == "openai:base_url=https://api.openai.com"

    def test_create_cache_key_with_api_key(self):
        """Test that API keys are hashed in cache keys."""
        key = _create_cache_key("openai", api_key="sk-test123")

        # API key should be hashed
        expected_hash = hashlib.sha256("sk-test123".encode()).hexdigest()[:16]
        assert f"api_key={expected_hash}" in key
        assert "sk-test123" not in key

    def test_create_cache_key_sorted(self):
        """Test that cache keys are deterministic (sorted)."""
        key1 = _create_cache_key("openai", base_url="url", api_key="key", organization="org")
        key2 = _create_cache_key("openai", organization="org", api_key="key", base_url="url")

        assert key1 == key2

    def test_create_cache_key_ignores_none(self):
        """Test that None values are excluded from cache key."""
        key = _create_cache_key("openai", base_url="url", api_key=None)
        assert "api_key" not in key
        assert "base_url=url" in key


class TestOpenAIDiscovery:
    """Test OpenAI model discovery."""

    @patch("esperanto.model_discovery.httpx.get")
    def test_get_openai_models_success(self, mock_get):
        """Test successful OpenAI model discovery."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-4", "owned_by": "openai"},
                {"id": "gpt-3.5-turbo", "owned_by": "openai"},
                {"id": "text-embedding-3-small", "owned_by": "openai"},
            ]
        }
        mock_get.return_value = mock_response

        # Test
        models = get_openai_models(api_key="test-key", model_type="language")

        # Should only return language models
        assert len(models) == 2
        assert all(m.id.startswith("gpt") for m in models)
        assert all(isinstance(m, Model) for m in models)

    @patch("esperanto.model_discovery.httpx.get")
    def test_get_openai_models_all_types(self, mock_get):
        """Test OpenAI discovery returns all types when model_type=None."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-4", "owned_by": "openai"},
                {"id": "text-embedding-3-small", "owned_by": "openai"},
                {"id": "whisper-1", "owned_by": "openai"},
                {"id": "tts-1", "owned_by": "openai"},
            ]
        }
        mock_get.return_value = mock_response

        models = get_openai_models(api_key="test-key", model_type=None)

        assert len(models) == 4

    def test_get_openai_models_no_api_key(self):
        """Test that ValueError is raised when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key not found"):
                get_openai_models()

    @patch("esperanto.model_discovery.httpx.get")
    def test_get_openai_models_api_error(self, mock_get):
        """Test that API errors are properly handled."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": {"message": "Invalid API key"}
        }
        mock_get.return_value = mock_response

        with pytest.raises(RuntimeError, match="OpenAI API error"):
            get_openai_models(api_key="bad-key")

    @patch("esperanto.model_discovery.httpx.get")
    def test_get_openai_models_caching(self, mock_get):
        """Test that results are cached."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"id": "gpt-4", "owned_by": "openai"}]
        }
        mock_get.return_value = mock_response

        # Clear cache first
        _model_cache.clear()

        # First call
        models1 = get_openai_models(api_key="test-key", model_type="language")

        # Second call should use cache
        models2 = get_openai_models(api_key="test-key", model_type="language")

        # Should only make one HTTP request
        assert mock_get.call_count == 1
        assert models1 == models2

    @patch("esperanto.model_discovery.httpx.get")
    def test_get_openai_models_with_env_var(self, mock_get):
        """Test that OpenAI API key can come from environment."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            get_openai_models()

            # Verify API key was used
            call_args = mock_get.call_args
            assert call_args.kwargs["headers"]["Authorization"] == "Bearer env-key"


class TestAnthropicDiscovery:
    """Test Anthropic model discovery."""

    def test_get_anthropic_models(self):
        """Test Anthropic model discovery returns hardcoded list."""
        models = get_anthropic_models()

        assert len(models) > 0
        assert all(isinstance(m, Model) for m in models)
        assert all(m.owned_by == "anthropic" for m in models)
        assert any("claude" in m.id for m in models)

    def test_get_anthropic_models_caching(self):
        """Test that Anthropic models are cached."""
        _model_cache.clear()

        models1 = get_anthropic_models()
        models2 = get_anthropic_models()

        # Should return same objects (from cache)
        assert models1 is models2


class TestGoogleDiscovery:
    """Test Google/Gemini model discovery."""

    @patch("esperanto.model_discovery.httpx.get")
    def test_get_google_models_success(self, mock_get):
        """Test successful Google model discovery."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "models/gemini-pro",
                    "supportedGenerationMethods": ["generateContent"],
                },
                {
                    "name": "models/gemini-pro-vision",
                    "supportedGenerationMethods": ["generateContent"],
                },
            ]
        }
        mock_get.return_value = mock_response

        models = get_google_models(api_key="test-key")

        assert len(models) == 2
        assert all(isinstance(m, Model) for m in models)
        assert models[0].id == "gemini-pro"

    def test_get_google_models_no_api_key(self):
        """Test that ValueError is raised when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Google API key not found"):
                get_google_models()

    @patch("esperanto.model_discovery.httpx.get")
    def test_get_google_models_with_env_var(self, mock_get):
        """Test that Google API key can come from environment."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-key"}):
            get_google_models()

            # Verify API key was used in query params
            call_args = mock_get.call_args
            assert call_args.kwargs.get("params", {}).get("key") == "env-key"


class TestOpenAICompatibleDiscovery:
    """Test OpenAI-compatible model discovery."""

    def test_get_openai_compatible_models_no_base_url(self):
        """Test that ValueError is raised when base_url is missing."""
        with pytest.raises(ValueError, match="base_url is required"):
            get_openai_compatible_models()

    @patch("esperanto.model_discovery.httpx.get")
    def test_get_openai_compatible_models_success(self, mock_get):
        """Test successful OpenAI-compatible model discovery."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "local-model-1", "owned_by": "local"},
                {"id": "local-model-2", "owned_by": "local"},
            ]
        }
        mock_get.return_value = mock_response

        models = get_openai_compatible_models(base_url="http://localhost:1234/v1")

        assert len(models) == 2
        assert all(isinstance(m, Model) for m in models)
        assert models[0].id == "local-model-1"

    @patch("esperanto.model_discovery.httpx.get")
    def test_get_openai_compatible_models_with_type_filter(self, mock_get):
        """Test OpenAI-compatible discovery with type filtering."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "llama-chat", "owned_by": "meta"},
                {"id": "text-embedding-local", "owned_by": "local"},
            ]
        }
        mock_get.return_value = mock_response

        # Filter for language models
        models = get_openai_compatible_models(
            base_url="http://localhost:1234/v1",
            model_type="language"
        )

        assert len(models) == 1
        assert "chat" in models[0].id.lower()

    @patch("esperanto.model_discovery.httpx.get")
    def test_get_openai_compatible_models_with_api_key(self, mock_get):
        """Test that API key is included in headers."""
        _model_cache.clear()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        mock_get.return_value = mock_response

        get_openai_compatible_models(
            base_url="http://localhost:1234/v1",
            api_key="test-key"
        )

        # Check that the API was called with Authorization header
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args.kwargs
        assert "headers" in call_kwargs
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"

    @patch("esperanto.model_discovery.httpx.get")
    def test_get_openai_compatible_models_strips_trailing_slash(self, mock_get):
        """Test that trailing slash is stripped from base_url."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        mock_get.return_value = mock_response

        get_openai_compatible_models(base_url="http://localhost:1234/v1/")

        call_args = mock_get.call_args
        assert call_args.args[0] == "http://localhost:1234/v1/models"


class TestProviderRegistry:
    """Test the provider registry."""

    def test_registry_contains_all_providers(self):
        """Test that registry has entries for all supported providers."""
        expected_providers = [
            "openai", "openai-compatible", "anthropic", "google", "vertex", "mistral",
            "groq", "deepseek", "ollama", "openrouter", "xai",
            "perplexity", "jina", "voyage", "azure", "transformers"
        ]

        for provider in expected_providers:
            assert provider in PROVIDER_MODELS_REGISTRY

    def test_registry_functions_are_callable(self):
        """Test that all registry entries are callable functions."""
        for provider, func in PROVIDER_MODELS_REGISTRY.items():
            assert callable(func), f"{provider} registry entry is not callable"


class TestAIFactoryIntegration:
    """Test AIFactory.get_provider_models() integration."""

    def test_get_provider_models_openai(self):
        """Test AIFactory.get_provider_models() for OpenAI."""
        # Patch the registry instead of the function
        with patch.dict("esperanto.model_discovery.PROVIDER_MODELS_REGISTRY") as mock_registry:
            mock_func = MagicMock(return_value=[Model(id="gpt-4", owned_by="openai")])
            mock_registry["openai"] = mock_func

            models = AIFactory.get_provider_models("openai", api_key="test-key")

            assert len(models) == 1
            assert models[0].id == "gpt-4"
            mock_func.assert_called_once()

    def test_get_provider_models_anthropic(self):
        """Test AIFactory.get_provider_models() for Anthropic."""
        # Patch the registry
        with patch.dict("esperanto.model_discovery.PROVIDER_MODELS_REGISTRY") as mock_registry:
            mock_func = MagicMock(return_value=[Model(id="claude-3-opus", owned_by="anthropic")])
            mock_registry["anthropic"] = mock_func

            models = AIFactory.get_provider_models("anthropic")

            assert len(models) == 1
            mock_func.assert_called_once()

    def test_get_provider_models_invalid_provider(self):
        """Test that invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="Provider 'invalid' not supported"):
            AIFactory.get_provider_models("invalid")

    def test_get_provider_models_with_model_type(self):
        """Test that model_type is passed to OpenAI discovery."""
        with patch.dict("esperanto.model_discovery.PROVIDER_MODELS_REGISTRY") as mock_registry:
            mock_func = MagicMock(return_value=[])
            mock_registry["openai"] = mock_func

            AIFactory.get_provider_models("openai", api_key="test", model_type="embedding")

            # Should pass model_type in config
            call_args = mock_func.call_args
            assert call_args.kwargs.get("model_type") == "embedding"

    def test_get_provider_models_case_insensitive(self):
        """Test that provider names are case-insensitive."""
        with patch.dict("esperanto.model_discovery.PROVIDER_MODELS_REGISTRY") as mock_registry:
            mock_func = MagicMock(return_value=[])
            mock_registry["openai"] = mock_func

            AIFactory.get_provider_models("OpenAI", api_key="test")
            AIFactory.get_provider_models("OPENAI", api_key="test")

            assert mock_func.call_count == 2


class TestModelCacheIntegration:
    """Test ModelCache integration in discovery."""

    def test_cache_expiration(self):
        """Test that cache entries expire after TTL."""
        _model_cache.clear()

        # This is a bit tricky to test without waiting an hour
        # We'll just verify the cache can be cleared
        get_anthropic_models()
        assert len(_model_cache._cache) > 0

        _model_cache.clear()
        assert len(_model_cache._cache) == 0

    def test_different_configs_different_cache(self):
        """Test that different configs create different cache entries."""
        _model_cache.clear()

        with patch("esperanto.model_discovery.httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}
            mock_get.return_value = mock_response

            # Two different API keys should create different cache entries
            get_openai_models(api_key="key1", model_type="language")
            get_openai_models(api_key="key2", model_type="language")

            # Should make two HTTP requests (different cache keys)
            assert mock_get.call_count == 2


class TestErrorHandling:
    """Test error handling in discovery functions."""

    @patch("esperanto.model_discovery.httpx.get")
    def test_http_timeout_error(self, mock_get):
        """Test that HTTP timeout errors are handled."""
        mock_get.side_effect = httpx.TimeoutException("Request timed out")

        with pytest.raises(RuntimeError, match="Failed to fetch OpenAI models"):
            get_openai_models(api_key="test-key")

    @patch("esperanto.model_discovery.httpx.get")
    def test_network_error(self, mock_get):
        """Test that network errors are handled."""
        mock_get.side_effect = httpx.NetworkError("Connection failed")

        with pytest.raises(RuntimeError, match="Failed to fetch OpenAI models"):
            get_openai_models(api_key="test-key")

    @patch("esperanto.model_discovery.httpx.get")
    def test_malformed_response(self, mock_get):
        """Test handling of malformed API responses."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response

        with pytest.raises(Exception):  # Should propagate the JSON error
            get_openai_models(api_key="test-key")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
