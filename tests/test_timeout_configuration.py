"""Tests for timeout configuration functionality."""

import os
import pytest
from unittest.mock import patch

from esperanto.utils.timeout import TimeoutMixin, DEFAULT_TIMEOUTS, TIMEOUT_ENV_VARS


class MockTimeoutModel(TimeoutMixin):
    """Mock model for testing timeout functionality."""

    def __init__(self, provider_type: str, config: dict = None):
        self.provider_type = provider_type
        self._config = config or {}

    def _get_provider_type(self) -> str:
        return self.provider_type


class TestTimeoutValidation:
    """Test timeout validation logic."""

    def test_valid_timeout_values(self):
        """Test that valid timeout values are accepted."""
        model = MockTimeoutModel("language")

        # Test positive integers
        assert model._validate_timeout(30) == 30.0
        assert model._validate_timeout(60) == 60.0
        assert model._validate_timeout(3600) == 3600.0

        # Test positive floats
        assert model._validate_timeout(30.5) == 30.5
        assert model._validate_timeout(1.5) == 1.5

    def test_invalid_timeout_types(self):
        """Test that invalid timeout types are rejected."""
        model = MockTimeoutModel("language")

        with pytest.raises(ValueError, match="Timeout must be a number"):
            model._validate_timeout("30")

        with pytest.raises(ValueError, match="Timeout must be a number"):
            model._validate_timeout(None)

        with pytest.raises(ValueError, match="Timeout must be a number"):
            model._validate_timeout([30])

    def test_invalid_timeout_values(self):
        """Test that invalid timeout values are rejected."""
        model = MockTimeoutModel("language")

        # Test negative values
        with pytest.raises(ValueError, match="Timeout must be positive"):
            model._validate_timeout(-1)

        with pytest.raises(ValueError, match="Timeout must be positive"):
            model._validate_timeout(0)

        # Test values exceeding maximum
        with pytest.raises(ValueError, match="Timeout cannot exceed 3600 seconds"):
            model._validate_timeout(3601)

        with pytest.raises(ValueError, match="Timeout cannot exceed 3600 seconds"):
            model._validate_timeout(7200)


class TestDefaultTimeouts:
    """Test default timeout values by provider type."""

    def test_language_provider_default(self):
        """Test that language providers get 60s default timeout."""
        model = MockTimeoutModel("language")
        assert model._get_default_timeout() == 60.0

    def test_embedding_provider_default(self):
        """Test that embedding providers get 60s default timeout."""
        model = MockTimeoutModel("embedding")
        assert model._get_default_timeout() == 60.0

    def test_reranker_provider_default(self):
        """Test that reranker providers get 60s default timeout."""
        model = MockTimeoutModel("reranker")
        assert model._get_default_timeout() == 60.0

    def test_stt_provider_default(self):
        """Test that STT providers get 300s default timeout."""
        model = MockTimeoutModel("speech_to_text")
        assert model._get_default_timeout() == 300.0

    def test_tts_provider_default(self):
        """Test that TTS providers get 300s default timeout."""
        model = MockTimeoutModel("text_to_speech")
        assert model._get_default_timeout() == 300.0

    def test_invalid_provider_type(self):
        """Test that invalid provider types raise errors."""
        model = MockTimeoutModel("invalid_type")

        with pytest.raises(ValueError, match="Unknown provider type"):
            model._get_default_timeout()


class TestEnvironmentVariables:
    """Test environment variable timeout configuration."""

    def test_environment_variable_names(self):
        """Test that environment variable names are correct."""
        model = MockTimeoutModel("language")
        assert model._get_timeout_env_var() == "ESPERANTO_LLM_TIMEOUT"

        model = MockTimeoutModel("embedding")
        assert model._get_timeout_env_var() == "ESPERANTO_EMBEDDING_TIMEOUT"

        model = MockTimeoutModel("reranker")
        assert model._get_timeout_env_var() == "ESPERANTO_RERANKER_TIMEOUT"

        model = MockTimeoutModel("speech_to_text")
        assert model._get_timeout_env_var() == "ESPERANTO_STT_TIMEOUT"

        model = MockTimeoutModel("text_to_speech")
        assert model._get_timeout_env_var() == "ESPERANTO_TTS_TIMEOUT"

    def test_invalid_provider_type_env_var(self):
        """Test that invalid provider types raise errors for env vars."""
        model = MockTimeoutModel("invalid_type")

        with pytest.raises(ValueError, match="Unknown provider type"):
            model._get_timeout_env_var()


class TestTimeoutResolution:
    """Test timeout resolution hierarchy."""

    def test_config_dict_priority(self):
        """Test that config dict has highest priority."""
        model = MockTimeoutModel("language", {"timeout": 120})

        with patch.dict(os.environ, {"ESPERANTO_LLM_TIMEOUT": "90"}):
            assert model._get_timeout() == 120.0

    def test_environment_variable_priority(self):
        """Test that environment variables have second priority."""
        model = MockTimeoutModel("language", {})

        with patch.dict(os.environ, {"ESPERANTO_LLM_TIMEOUT": "90"}):
            assert model._get_timeout() == 90.0

    def test_default_priority(self):
        """Test that defaults are used when no config or env var."""
        model = MockTimeoutModel("language", {})

        with patch.dict(os.environ, {}, clear=True):
            assert model._get_timeout() == 60.0

    def test_invalid_environment_variable(self):
        """Test that invalid environment variables raise clear errors."""
        model = MockTimeoutModel("language", {})

        with patch.dict(os.environ, {"ESPERANTO_LLM_TIMEOUT": "invalid"}):
            with pytest.raises(ValueError, match="Invalid timeout value in environment variable"):
                model._get_timeout()

    def test_negative_environment_variable(self):
        """Test that negative environment variables raise clear errors."""
        model = MockTimeoutModel("language", {})

        with patch.dict(os.environ, {"ESPERANTO_LLM_TIMEOUT": "-10"}):
            with pytest.raises(ValueError, match="Invalid timeout value in environment variable"):
                model._get_timeout()


class TestConstants:
    """Test timeout constants are properly defined."""

    def test_default_timeouts_constant(self):
        """Test that DEFAULT_TIMEOUTS has all required provider types."""
        expected_types = {"language", "embedding", "reranker", "speech_to_text", "text_to_speech"}
        assert set(DEFAULT_TIMEOUTS.keys()) == expected_types

        # Test expected values
        assert DEFAULT_TIMEOUTS["language"] == 60.0
        assert DEFAULT_TIMEOUTS["embedding"] == 60.0
        assert DEFAULT_TIMEOUTS["reranker"] == 60.0
        assert DEFAULT_TIMEOUTS["speech_to_text"] == 300.0
        assert DEFAULT_TIMEOUTS["text_to_speech"] == 300.0

    def test_timeout_env_vars_constant(self):
        """Test that TIMEOUT_ENV_VARS has all required provider types."""
        expected_types = {"language", "embedding", "reranker", "speech_to_text", "text_to_speech"}
        assert set(TIMEOUT_ENV_VARS.keys()) == expected_types

        # Test expected variable names
        assert TIMEOUT_ENV_VARS["language"] == "ESPERANTO_LLM_TIMEOUT"
        assert TIMEOUT_ENV_VARS["embedding"] == "ESPERANTO_EMBEDDING_TIMEOUT"
        assert TIMEOUT_ENV_VARS["reranker"] == "ESPERANTO_RERANKER_TIMEOUT"
        assert TIMEOUT_ENV_VARS["speech_to_text"] == "ESPERANTO_STT_TIMEOUT"
        assert TIMEOUT_ENV_VARS["text_to_speech"] == "ESPERANTO_TTS_TIMEOUT"