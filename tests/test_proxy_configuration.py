"""Tests for proxy configuration functionality."""

import os
from unittest.mock import patch

from esperanto.utils.connect import HttpConnectionMixin


class MockProxyModel(HttpConnectionMixin):
    """Mock model for testing proxy configuration functionality."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def _get_provider_type(self) -> str:
        return "language"


class TestProxyDefault:
    """Test default proxy behavior."""

    def test_default_returns_none(self):
        """Test that proxy is None by default."""
        model = MockProxyModel()
        assert model._get_proxy() is None

    def test_empty_config_returns_none(self):
        """Test that empty config returns None."""
        model = MockProxyModel(config={})
        assert model._get_proxy() is None


class TestProxyConfig:
    """Test proxy configuration via config dict."""

    def test_proxy_from_config(self):
        """Test that proxy can be set via config dict."""
        model = MockProxyModel(config={"proxy": "http://proxy.example.com:8080"})
        assert model._get_proxy() == "http://proxy.example.com:8080"

    def test_proxy_https_from_config(self):
        """Test that HTTPS proxy can be set via config dict."""
        model = MockProxyModel(config={"proxy": "https://secure-proxy.example.com:443"})
        assert model._get_proxy() == "https://secure-proxy.example.com:443"

    def test_proxy_with_auth_from_config(self):
        """Test that proxy with authentication can be set via config dict."""
        model = MockProxyModel(config={"proxy": "http://user:pass@proxy.example.com:8080"})
        assert model._get_proxy() == "http://user:pass@proxy.example.com:8080"


class TestProxyEnvironmentVariable:
    """Test proxy configuration via environment variable."""

    def test_env_var_sets_proxy(self):
        """Test that ESPERANTO_PROXY environment variable sets proxy."""
        model = MockProxyModel()

        with patch.dict(os.environ, {"ESPERANTO_PROXY": "http://env-proxy.example.com:3128"}):
            assert model._get_proxy() == "http://env-proxy.example.com:3128"

    def test_env_var_with_https_proxy(self):
        """Test that ESPERANTO_PROXY works with HTTPS proxy URL."""
        model = MockProxyModel()

        with patch.dict(os.environ, {"ESPERANTO_PROXY": "https://secure-env-proxy.example.com:443"}):
            assert model._get_proxy() == "https://secure-env-proxy.example.com:443"


class TestProxyPriority:
    """Test proxy configuration priority (config > env var)."""

    def test_config_takes_precedence_over_env_var(self):
        """Test that config dict takes precedence over environment variable."""
        model = MockProxyModel(config={"proxy": "http://config-proxy.example.com:8080"})

        with patch.dict(os.environ, {"ESPERANTO_PROXY": "http://env-proxy.example.com:3128"}):
            assert model._get_proxy() == "http://config-proxy.example.com:8080"

    def test_env_var_used_when_config_proxy_is_none(self):
        """Test that env var is used when config doesn't have proxy."""
        model = MockProxyModel(config={"timeout": 120})  # config exists but no proxy

        with patch.dict(os.environ, {"ESPERANTO_PROXY": "http://env-proxy.example.com:3128"}):
            assert model._get_proxy() == "http://env-proxy.example.com:3128"

    def test_env_var_used_when_config_proxy_is_empty(self):
        """Test that env var is used when config proxy is empty string."""
        model = MockProxyModel(config={"proxy": ""})

        with patch.dict(os.environ, {"ESPERANTO_PROXY": "http://env-proxy.example.com:3128"}):
            # Empty string is falsy, so env var should be used
            assert model._get_proxy() == "http://env-proxy.example.com:3128"


class TestProxyClientCreation:
    """Test that proxy is passed to HTTP clients."""

    def test_clients_created_with_proxy(self):
        """Test that HTTP clients are created with proxy configuration."""
        model = MockProxyModel(config={"proxy": "http://proxy.example.com:8080"})
        model._create_http_clients()

        # httpx stores proxy in _proxy_url attribute (internal detail)
        # We verify by checking the client was created successfully
        assert model.client is not None
        assert model.async_client is not None

    def test_clients_created_without_proxy(self):
        """Test that HTTP clients are created when proxy is None."""
        model = MockProxyModel()
        model._create_http_clients()

        assert model.client is not None
        assert model.async_client is not None

    def test_clients_created_with_env_var_proxy(self):
        """Test that HTTP clients use proxy from environment variable."""
        model = MockProxyModel()

        with patch.dict(os.environ, {"ESPERANTO_PROXY": "http://env-proxy.example.com:3128"}):
            model._create_http_clients()

        assert model.client is not None
        assert model.async_client is not None
