"""Model cache with TTL for Esperanto."""

import threading
import time
from typing import Any, Dict, List, Optional

from esperanto.common_types import Model


class CachedResult:
    """Cached result with expiration timestamp."""

    def __init__(self, data: List[Model], ttl: float):
        """Initialize cached result.

        Args:
            data: The cached data
            ttl: Time-to-live in seconds
        """
        self.data = data
        self.expires_at = time.time() + ttl

    def is_expired(self) -> bool:
        """Check if this cached result has expired.

        Returns:
            True if expired, False otherwise
        """
        return time.time() > self.expires_at


class ModelCache:
    """Thread-safe cache for model lists with TTL expiration.

    This cache stores model lists per provider with time-based expiration.
    Used to reduce unnecessary API calls to provider model listing endpoints.
    """

    def __init__(self):
        """Initialize the model cache."""
        self._cache: Dict[str, CachedResult] = {}
        self._lock = threading.Lock()

    def get(self, cache_key: str) -> Optional[List[Model]]:
        """Get cached models for a provider.

        Args:
            cache_key: Unique cache key (typically provider:api_key_hash)

        Returns:
            Cached list of models if available and not expired, None otherwise
        """
        with self._lock:
            cached_result = self._cache.get(cache_key)

            if cached_result is None:
                return None

            if cached_result.is_expired():
                # Remove expired entry
                del self._cache[cache_key]
                return None

            return cached_result.data

    def set(self, cache_key: str, data: List[Model], ttl: float = 3600.0) -> None:
        """Store models in cache with TTL.

        Args:
            cache_key: Unique cache key (typically provider:api_key_hash)
            data: List of models to cache
            ttl: Time-to-live in seconds (default: 3600 = 1 hour)
        """
        with self._lock:
            self._cache[cache_key] = CachedResult(data, ttl)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    def invalidate(self, cache_key: str) -> None:
        """Invalidate a specific cache entry.

        Args:
            cache_key: Cache key to invalidate
        """
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
