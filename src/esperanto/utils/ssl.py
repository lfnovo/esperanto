"""SSL verification configuration utilities for Esperanto providers."""

import os
import warnings
from typing import Any, Union

# Environment variable names for SSL configuration
SSL_VERIFY_ENV_VAR = "ESPERANTO_SSL_VERIFY"
SSL_CA_BUNDLE_ENV_VAR = "ESPERANTO_SSL_CA_BUNDLE"


class SSLMixin:
    """Mixin providing SSL verification configuration functionality.

    This mixin provides a standardized way to configure SSL verification across all
    Esperanto providers. It implements a priority system:

    1. Config dict ssl_ca_bundle (highest priority)
    2. Config dict verify_ssl
    3. Environment variable ESPERANTO_SSL_CA_BUNDLE
    4. Environment variable ESPERANTO_SSL_VERIFY
    5. Default: True (SSL verification enabled)

    The mixin must be used with classes that have:
    - _config: Dict[str, Any] attribute

    Example:
        # Disable SSL verification (development only)
        model = AIFactory.create_language(
            "ollama",
            "llama3",
            config={"verify_ssl": False}
        )

        # Use custom CA bundle (recommended for self-signed certs)
        model = AIFactory.create_language(
            "ollama",
            "llama3",
            config={"ssl_ca_bundle": "/path/to/ca-bundle.pem"}
        )
    """

    def _get_ssl_verify(self) -> Union[bool, str]:
        """Get SSL verification setting using priority hierarchy.

        Priority order (highest to lowest):
        1. Config dict ssl_ca_bundle: config={"ssl_ca_bundle": "/path/to/ca.pem"}
        2. Config dict verify_ssl: config={"verify_ssl": False}
        3. Environment variable: ESPERANTO_SSL_CA_BUNDLE=/path/to/ca.pem
        4. Environment variable: ESPERANTO_SSL_VERIFY=false
        5. Default: True (SSL verification enabled)

        Returns:
            Union[bool, str]:
                - True: SSL verification enabled using default CA bundle
                - False: SSL verification disabled (insecure)
                - str: Path to custom CA bundle file

        Raises:
            ValueError: If CA bundle path does not exist or verify_ssl has invalid type
        """
        # 1. Config dict ssl_ca_bundle (highest priority)
        if hasattr(self, "_config"):
            ca_bundle = self._config.get("ssl_ca_bundle")
            if ca_bundle:
                self._validate_ca_bundle(ca_bundle)
                return ca_bundle

            # 2. Config dict verify_ssl
            verify_ssl = self._config.get("verify_ssl")
            if verify_ssl is not None:
                validated = self._validate_verify_ssl(verify_ssl)
                if not validated:
                    self._emit_ssl_warning()
                return validated

        # 3. Environment variable for CA bundle
        ca_bundle_env = os.getenv(SSL_CA_BUNDLE_ENV_VAR)
        if ca_bundle_env:
            self._validate_ca_bundle(ca_bundle_env)
            return ca_bundle_env

        # 4. Environment variable for verify_ssl
        verify_env = os.getenv(SSL_VERIFY_ENV_VAR, "").lower()
        if verify_env in ("false", "0", "no"):
            self._emit_ssl_warning()
            return False

        # 5. Default: SSL verification enabled
        return True

    def _validate_verify_ssl(self, value: Any) -> bool:
        """Validate and normalize verify_ssl configuration value.

        Accepts boolean values directly, and converts common string representations
        of boolean values. Raises ValueError for invalid types.

        Args:
            value: The verify_ssl configuration value to validate

        Returns:
            bool: Normalized boolean value

        Raises:
            ValueError: If value cannot be converted to boolean
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            if value.lower() in ("false", "0", "no"):
                return False
            if value.lower() in ("true", "1", "yes"):
                return True
        raise ValueError(
            f"verify_ssl must be a boolean, got {type(value).__name__}: {value!r}"
        )

    def _validate_ca_bundle(self, path: str) -> None:
        """Validate that CA bundle file exists.

        Args:
            path: Path to CA bundle file

        Raises:
            ValueError: If file does not exist
        """
        if not os.path.isfile(path):
            raise ValueError(f"CA bundle file not found: {path}")

    def _emit_ssl_warning(self) -> None:
        """Emit a security warning when SSL verification is disabled.

        This warning helps users identify when they're running in an insecure
        mode, which is particularly important if disabled configuration
        accidentally reaches production.
        """
        warnings.warn(
            "SSL verification is disabled. This is insecure and should only be used "
            "in development/testing environments. For production, use ssl_ca_bundle "
            "to specify a custom CA certificate instead.",
            UserWarning,
            stacklevel=4,
        )
