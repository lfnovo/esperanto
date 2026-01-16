"""Connection utilities for Esperanto providers."""

from abc import ABC

import httpx

from .timeout import TimeoutMixin
from .ssl import SSLMixin


class HttpConnectionMixin(TimeoutMixin, SSLMixin, ABC):
    """Mixin providing HTTP connection functionality.

    This mixin provides a standardized way to configure HTTP connections across all Esperanto providers.

    This mixin already provides timeout and SSL configuration functionality as httpx client relies on `_get_timeout` and `_get_ssl_verify` methods.

    The `_create_http_clients` method should be used with classes that have:
    - client: httpx.Client and async_client: httpx.AsyncClient attributes
    - Provider-specific __post_init__() that calls super().__post_init__()
    """

    def _create_http_clients(self) -> None:
        """Create HTTP clients with configured timeout and SSL settings.

        Call this method in provider's __post_init__ after setting up
        API keys and base URLs.
        """
        timeout = self._get_timeout()
        verify = self._get_ssl_verify()
        self.client = httpx.Client(timeout=timeout, verify=verify)
        self.async_client = httpx.AsyncClient(timeout=timeout, verify=verify)

    def close(self):
        """Explicitly close HTTP clients."""
        try:
            if (
                hasattr(self, "client")
                and self.client is not None
                and not self.client.is_closed
            ):
                self.client.close()
        except Exception:
            pass  # Ignore cleanup errors

    async def aclose(self):
        """Asynchronously close HTTP clients."""
        try:
            if (
                hasattr(self, "async_client")
                and self.async_client is not None
                and not self.async_client.is_closed
            ):
                await self.async_client.aclose()
        except Exception:
            pass  # Ignore cleanup errors

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.aclose()

    def __del__(self):
        """Clean up HTTP clients on object destruction."""
        # Only handle sync client cleanup in destructor
        try:
            if (
                hasattr(self, "close")
                and callable(self.close)
            ):
                self.close()
        except Exception:
            pass  # Ignore cleanup errors
