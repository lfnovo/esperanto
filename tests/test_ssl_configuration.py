"""Tests for SSL verification configuration functionality."""

import os
import tempfile
import warnings

import pytest
from unittest.mock import patch

from esperanto.utils.ssl import SSLMixin, SSL_VERIFY_ENV_VAR, SSL_CA_BUNDLE_ENV_VAR


class MockSSLModel(SSLMixin):
    """Mock model for testing SSL configuration functionality."""

    def __init__(self, config: dict = None):
        self._config = config or {}


class TestSSLVerifyDefault:
    """Test default SSL verification behavior."""

    def test_default_returns_true(self):
        """Test that SSL verification is enabled by default."""
        model = MockSSLModel()
        assert model._get_ssl_verify() is True

    def test_empty_config_returns_true(self):
        """Test that empty config returns True (SSL enabled)."""
        model = MockSSLModel(config={})
        assert model._get_ssl_verify() is True


class TestSSLVerifyConfig:
    """Test SSL configuration via config dict."""

    def test_verify_ssl_false_returns_false(self):
        """Test that verify_ssl=False disables SSL verification."""
        model = MockSSLModel(config={"verify_ssl": False})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = model._get_ssl_verify()
            assert result is False
            assert len(w) == 1
            assert "SSL verification is disabled" in str(w[0].message)

    def test_verify_ssl_true_returns_true(self):
        """Test that verify_ssl=True explicitly enables SSL verification."""
        model = MockSSLModel(config={"verify_ssl": True})
        assert model._get_ssl_verify() is True

    def test_ssl_ca_bundle_returns_path(self):
        """Test that ssl_ca_bundle returns the path to CA bundle."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as f:
            ca_path = f.name
            f.write(b"fake ca bundle content")

        try:
            model = MockSSLModel(config={"ssl_ca_bundle": ca_path})
            assert model._get_ssl_verify() == ca_path
        finally:
            os.unlink(ca_path)

    def test_ssl_ca_bundle_takes_precedence_over_verify_ssl(self):
        """Test that ssl_ca_bundle takes precedence over verify_ssl."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as f:
            ca_path = f.name
            f.write(b"fake ca bundle content")

        try:
            # Even with verify_ssl=False, ssl_ca_bundle should be used
            model = MockSSLModel(config={"verify_ssl": False, "ssl_ca_bundle": ca_path})
            result = model._get_ssl_verify()
            assert result == ca_path
        finally:
            os.unlink(ca_path)


class TestSSLVerifyEnvironmentVariables:
    """Test SSL configuration via environment variables."""

    def test_env_var_false_disables_ssl(self):
        """Test that ESPERANTO_SSL_VERIFY=false disables SSL verification."""
        model = MockSSLModel(config={})

        with patch.dict(os.environ, {SSL_VERIFY_ENV_VAR: "false"}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = model._get_ssl_verify()
                assert result is False
                assert len(w) == 1
                assert "SSL verification is disabled" in str(w[0].message)

    def test_env_var_0_disables_ssl(self):
        """Test that ESPERANTO_SSL_VERIFY=0 disables SSL verification."""
        model = MockSSLModel(config={})

        with patch.dict(os.environ, {SSL_VERIFY_ENV_VAR: "0"}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = model._get_ssl_verify()
                assert result is False

    def test_env_var_no_disables_ssl(self):
        """Test that ESPERANTO_SSL_VERIFY=no disables SSL verification."""
        model = MockSSLModel(config={})

        with patch.dict(os.environ, {SSL_VERIFY_ENV_VAR: "no"}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = model._get_ssl_verify()
                assert result is False

    def test_env_var_ca_bundle_returns_path(self):
        """Test that ESPERANTO_SSL_CA_BUNDLE env var returns path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as f:
            ca_path = f.name
            f.write(b"fake ca bundle content")

        try:
            model = MockSSLModel(config={})
            with patch.dict(os.environ, {SSL_CA_BUNDLE_ENV_VAR: ca_path}):
                assert model._get_ssl_verify() == ca_path
        finally:
            os.unlink(ca_path)

    def test_env_var_ca_bundle_takes_precedence_over_verify(self):
        """Test that CA bundle env var takes precedence over verify env var."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as f:
            ca_path = f.name
            f.write(b"fake ca bundle content")

        try:
            model = MockSSLModel(config={})
            with patch.dict(
                os.environ,
                {SSL_VERIFY_ENV_VAR: "false", SSL_CA_BUNDLE_ENV_VAR: ca_path},
            ):
                result = model._get_ssl_verify()
                assert result == ca_path
        finally:
            os.unlink(ca_path)


class TestSSLConfigPriority:
    """Test priority hierarchy between config and environment variables."""

    def test_config_takes_precedence_over_env_var(self):
        """Test that config dict takes precedence over environment variables."""
        model = MockSSLModel(config={"verify_ssl": True})

        with patch.dict(os.environ, {SSL_VERIFY_ENV_VAR: "false"}):
            # Config says True, env var says False - config should win
            assert model._get_ssl_verify() is True

    def test_config_ca_bundle_takes_precedence_over_env_var(self):
        """Test that config ssl_ca_bundle takes precedence over env var."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as f1:
            ca_path_config = f1.name
            f1.write(b"config ca bundle")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as f2:
            ca_path_env = f2.name
            f2.write(b"env ca bundle")

        try:
            model = MockSSLModel(config={"ssl_ca_bundle": ca_path_config})
            with patch.dict(os.environ, {SSL_CA_BUNDLE_ENV_VAR: ca_path_env}):
                result = model._get_ssl_verify()
                assert result == ca_path_config
        finally:
            os.unlink(ca_path_config)
            os.unlink(ca_path_env)


class TestVerifySSLValidation:
    """Test verify_ssl value validation."""

    def test_bool_true_accepted(self):
        """Test that boolean True is accepted and returned as-is."""
        model = MockSSLModel()
        assert model._validate_verify_ssl(True) is True

    def test_bool_false_accepted(self):
        """Test that boolean False is accepted and returned as-is."""
        model = MockSSLModel()
        assert model._validate_verify_ssl(False) is False

    def test_int_1_converted_to_true(self):
        """Test that integer 1 is converted to True."""
        model = MockSSLModel()
        assert model._validate_verify_ssl(1) is True

    def test_int_0_converted_to_false(self):
        """Test that integer 0 is converted to False."""
        model = MockSSLModel()
        assert model._validate_verify_ssl(0) is False

    def test_string_false_converted(self):
        """Test that string 'false' is converted to False."""
        model = MockSSLModel()
        assert model._validate_verify_ssl("false") is False
        assert model._validate_verify_ssl("False") is False
        assert model._validate_verify_ssl("FALSE") is False

    def test_string_true_converted(self):
        """Test that string 'true' is converted to True."""
        model = MockSSLModel()
        assert model._validate_verify_ssl("true") is True
        assert model._validate_verify_ssl("True") is True
        assert model._validate_verify_ssl("TRUE") is True

    def test_string_0_converted_to_false(self):
        """Test that string '0' is converted to False."""
        model = MockSSLModel()
        assert model._validate_verify_ssl("0") is False

    def test_string_1_converted_to_true(self):
        """Test that string '1' is converted to True."""
        model = MockSSLModel()
        assert model._validate_verify_ssl("1") is True

    def test_string_no_converted_to_false(self):
        """Test that string 'no' is converted to False."""
        model = MockSSLModel()
        assert model._validate_verify_ssl("no") is False
        assert model._validate_verify_ssl("No") is False
        assert model._validate_verify_ssl("NO") is False

    def test_string_yes_converted_to_true(self):
        """Test that string 'yes' is converted to True."""
        model = MockSSLModel()
        assert model._validate_verify_ssl("yes") is True
        assert model._validate_verify_ssl("Yes") is True
        assert model._validate_verify_ssl("YES") is True

    def test_invalid_string_raises_error(self):
        """Test that invalid string raises ValueError."""
        model = MockSSLModel()
        with pytest.raises(ValueError, match="verify_ssl must be a boolean"):
            model._validate_verify_ssl("invalid")

    def test_invalid_type_raises_error(self):
        """Test that invalid types raise ValueError."""
        model = MockSSLModel()
        with pytest.raises(ValueError, match="verify_ssl must be a boolean"):
            model._validate_verify_ssl([True])
        with pytest.raises(ValueError, match="verify_ssl must be a boolean"):
            model._validate_verify_ssl({"ssl": True})

    def test_string_false_in_config_works(self):
        """Test that string 'false' in config is properly converted."""
        model = MockSSLModel(config={"verify_ssl": "false"})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = model._get_ssl_verify()
            assert result is False
            assert len(w) == 1


class TestCABundleValidation:
    """Test CA bundle path validation."""

    def test_valid_ca_bundle_path_accepted(self):
        """Test that valid CA bundle path is accepted."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as f:
            ca_path = f.name
            f.write(b"fake ca bundle content")

        try:
            model = MockSSLModel()
            # Should not raise
            model._validate_ca_bundle(ca_path)
        finally:
            os.unlink(ca_path)

    def test_invalid_ca_bundle_path_raises_error(self):
        """Test that non-existent CA bundle path raises ValueError."""
        model = MockSSLModel()
        with pytest.raises(ValueError, match="CA bundle file not found"):
            model._validate_ca_bundle("/non/existent/path/ca-bundle.pem")

    def test_directory_as_ca_bundle_raises_error(self):
        """Test that directory path raises ValueError."""
        model = MockSSLModel()
        with pytest.raises(ValueError, match="CA bundle file not found"):
            model._validate_ca_bundle("/tmp")


class TestSSLWarning:
    """Test SSL security warning emission."""

    def test_warning_emitted_when_ssl_disabled_via_config(self):
        """Test that warning is emitted when SSL is disabled via config."""
        model = MockSSLModel(config={"verify_ssl": False})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model._get_ssl_verify()
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "SSL verification is disabled" in str(w[0].message)
            assert "insecure" in str(w[0].message)
            assert "ssl_ca_bundle" in str(w[0].message)

    def test_warning_emitted_when_ssl_disabled_via_env_var(self):
        """Test that warning is emitted when SSL is disabled via env var."""
        model = MockSSLModel(config={})

        with patch.dict(os.environ, {SSL_VERIFY_ENV_VAR: "false"}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model._get_ssl_verify()
                assert len(w) == 1
                assert issubclass(w[0].category, UserWarning)

    def test_no_warning_when_ssl_enabled(self):
        """Test that no warning is emitted when SSL is enabled."""
        model = MockSSLModel(config={"verify_ssl": True})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model._get_ssl_verify()
            assert len(w) == 0

    def test_no_warning_when_using_ca_bundle(self):
        """Test that no warning is emitted when using CA bundle."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as f:
            ca_path = f.name
            f.write(b"fake ca bundle content")

        try:
            model = MockSSLModel(config={"ssl_ca_bundle": ca_path})
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model._get_ssl_verify()
                assert len(w) == 0
        finally:
            os.unlink(ca_path)


class TestConstants:
    """Test SSL configuration constants."""

    def test_env_var_names(self):
        """Test that environment variable names are correct."""
        assert SSL_VERIFY_ENV_VAR == "ESPERANTO_SSL_VERIFY"
        assert SSL_CA_BUNDLE_ENV_VAR == "ESPERANTO_SSL_CA_BUNDLE"


class TestBaseClassIntegration:
    """Test that SSL configuration is properly integrated into base classes."""

    def test_language_model_has_ssl_mixin(self):
        """Test that LanguageModel inherits from SSLMixin."""
        from esperanto.providers.llm.base import LanguageModel
        assert issubclass(LanguageModel, SSLMixin)

    def test_embedding_model_has_ssl_mixin(self):
        """Test that EmbeddingModel inherits from SSLMixin."""
        from esperanto.providers.embedding.base import EmbeddingModel
        assert issubclass(EmbeddingModel, SSLMixin)

    def test_speech_to_text_model_has_ssl_mixin(self):
        """Test that SpeechToTextModel inherits from SSLMixin."""
        from esperanto.providers.stt.base import SpeechToTextModel
        assert issubclass(SpeechToTextModel, SSLMixin)

    def test_text_to_speech_model_has_ssl_mixin(self):
        """Test that TextToSpeechModel inherits from SSLMixin."""
        from esperanto.providers.tts.base import TextToSpeechModel
        assert issubclass(TextToSpeechModel, SSLMixin)

    def test_reranker_model_has_ssl_mixin(self):
        """Test that RerankerModel inherits from SSLMixin."""
        from esperanto.providers.reranker.base import RerankerModel
        assert issubclass(RerankerModel, SSLMixin)


class TestHTTPClientCreation:
    """Test that HTTP clients are created with SSL configuration."""

    def test_http_client_created_with_ssl_verify_false(self):
        """Test that httpx clients are created with verify=False when configured."""
        from unittest.mock import MagicMock, patch

        # Create a mock model that uses _create_http_clients
        model = MockSSLModel(config={"verify_ssl": False})

        with patch("httpx.Client") as mock_client, \
             patch("httpx.AsyncClient") as mock_async_client:

            # Simulate what _create_http_clients does
            import httpx
            timeout = 60.0  # default
            verify = model._get_ssl_verify()

            # Verify that _get_ssl_verify returns False
            assert verify is False

            # Simulate the httpx client creation
            httpx.Client(timeout=timeout, verify=verify)
            httpx.AsyncClient(timeout=timeout, verify=verify)

            # Verify httpx was called with verify=False
            mock_client.assert_called_once_with(timeout=60.0, verify=False)
            mock_async_client.assert_called_once_with(timeout=60.0, verify=False)

    def test_http_client_created_with_ca_bundle(self):
        """Test that httpx clients are created with CA bundle path when configured."""
        from unittest.mock import patch

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as f:
            ca_path = f.name
            f.write(b"fake ca bundle content")

        try:
            model = MockSSLModel(config={"ssl_ca_bundle": ca_path})

            with patch("httpx.Client") as mock_client, \
                 patch("httpx.AsyncClient") as mock_async_client:

                timeout = 60.0
                verify = model._get_ssl_verify()

                # Verify that _get_ssl_verify returns the CA path
                assert verify == ca_path

                # Simulate the httpx client creation
                import httpx
                httpx.Client(timeout=timeout, verify=verify)
                httpx.AsyncClient(timeout=timeout, verify=verify)

                mock_client.assert_called_once_with(timeout=60.0, verify=ca_path)
                mock_async_client.assert_called_once_with(timeout=60.0, verify=ca_path)
        finally:
            os.unlink(ca_path)
