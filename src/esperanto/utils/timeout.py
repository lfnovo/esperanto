"""Timeout configuration utilities for Esperanto providers."""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict

# Default timeout values by provider type (in seconds)
DEFAULT_TIMEOUTS: Dict[str, float] = {
    "language": 60.0,        # LLM providers (increased from 30s)
    "embedding": 60.0,       # Embedding providers (increased from 30s)
    "reranker": 60.0,        # Reranker providers (increased from 30s)
    "speech_to_text": 300.0, # STT providers (10x increase for file processing)
    "text_to_speech": 300.0  # TTS providers (10x increase for file processing)
}

# Environment variable names for timeout configuration
TIMEOUT_ENV_VARS: Dict[str, str] = {
    "language": "ESPERANTO_LLM_TIMEOUT",
    "embedding": "ESPERANTO_EMBEDDING_TIMEOUT",
    "reranker": "ESPERANTO_RERANKER_TIMEOUT",
    "speech_to_text": "ESPERANTO_STT_TIMEOUT",
    "text_to_speech": "ESPERANTO_TTS_TIMEOUT"
}


class TimeoutMixin(ABC):
    """Mixin providing timeout configuration functionality.

    This mixin provides a standardized way to configure timeouts across all
    Esperanto providers. It implements a three-tier priority system:

    1. Config dict timeout (highest priority)
    2. Environment variable timeout
    3. Provider type default timeout (lowest priority)

    The mixin must be used with classes that have:
    - _config: Dict[str, Any] attribute
    - Provider-specific __post_init__() that calls super().__post_init__()
    """

    @abstractmethod
    def _get_provider_type(self) -> str:
        """Return the provider type for timeout defaults.

        Returns:
            str: One of "language", "embedding", "reranker", "speech_to_text", "text_to_speech"
        """
        pass

    def _get_timeout(self) -> float:
        """Get timeout value using priority hierarchy.

        Priority order (highest to lowest):
        1. Config dict: config={"timeout": 120}
        2. Environment variable: ESPERANTO_LLM_TIMEOUT=90
        3. Provider type default: 60.0 for LLM, 300.0 for STT/TTS

        Returns:
            float: Validated timeout value in seconds

        Raises:
            ValueError: If timeout value is invalid
        """
        # 1. Config dict (highest priority)
        if hasattr(self, '_config') and "timeout" in self._config:
            return self._validate_timeout(self._config["timeout"])

        # 2. Environment variable
        env_var = self._get_timeout_env_var()
        env_timeout = os.getenv(env_var)
        if env_timeout:
            try:
                return self._validate_timeout(float(env_timeout))
            except ValueError as e:
                raise ValueError(
                    f"Invalid timeout value in environment variable {env_var}={env_timeout}: {str(e)}"
                ) from e

        # 3. Provider type default (lowest priority)
        return self._get_default_timeout()

    def _validate_timeout(self, timeout: Any) -> float:
        """Validate timeout value and return as float.

        Args:
            timeout: Timeout value to validate

        Returns:
            float: Validated timeout value

        Raises:
            ValueError: If timeout is invalid
        """
        # Type validation
        if not isinstance(timeout, (int, float)):
            raise ValueError(f"Timeout must be a number, got {type(timeout).__name__}")

        # Convert to float
        timeout_float = float(timeout)

        # Range validation
        if timeout_float <= 0:
            raise ValueError(f"Timeout must be positive, got {timeout_float}")

        if timeout_float > 3600:  # 1 hour maximum
            raise ValueError(f"Timeout cannot exceed 3600 seconds (1 hour), got {timeout_float}")

        return timeout_float

    def _get_default_timeout(self) -> float:
        """Get the default timeout for this provider type.

        Returns:
            float: Default timeout in seconds

        Raises:
            ValueError: If provider type is not recognized
        """
        provider_type = self._get_provider_type()

        if provider_type not in DEFAULT_TIMEOUTS:
            raise ValueError(
                f"Unknown provider type '{provider_type}'. "
                f"Must be one of: {list(DEFAULT_TIMEOUTS.keys())}"
            )

        return DEFAULT_TIMEOUTS[provider_type]

    def _get_timeout_env_var(self) -> str:
        """Get the environment variable name for this provider type.

        Returns:
            str: Environment variable name (e.g., "ESPERANTO_LLM_TIMEOUT")

        Raises:
            ValueError: If provider type is not recognized
        """
        provider_type = self._get_provider_type()

        if provider_type not in TIMEOUT_ENV_VARS:
            raise ValueError(
                f"Unknown provider type '{provider_type}'. "
                f"Must be one of: {list(TIMEOUT_ENV_VARS.keys())}"
            )

        return TIMEOUT_ENV_VARS[provider_type]