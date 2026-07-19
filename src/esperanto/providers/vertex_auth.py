"""Shared Google Cloud (Vertex AI) authentication for the Vertex providers.

All three Vertex modalities (language, embedding, text-to-speech) authenticate
the same way against Google Cloud. This mixin is the single implementation:
resolve credentials once (`_load_credentials`), then mint an OAuth token per
request (`_get_access_token`).

Credential resolution priority:
1. An explicit service-account key file — from a ``credentials_file`` attribute,
   or a ``credentials_file`` / ``credentials_path`` config entry.
2. ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable.
3. Application Default Credentials (``google.auth.default``).
4. The ``gcloud`` CLI (last-resort fallback in ``_get_access_token``).
"""

import os
import subprocess
import time
from typing import Any, Optional

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class VertexAuthMixin:
    """Google Cloud auth shared across the Vertex provider modalities."""

    # Attributes the mixin reads/writes; providers may also initialize these.
    _credentials: Optional[Any] = None
    _access_token: Optional[str] = None
    _token_expiry: float = 0.0

    def _load_credentials(self) -> None:
        """Resolve service-account / ADC credentials into ``self._credentials``.

        Sets ``self._credentials`` to a google-auth Credentials object, or
        ``None`` to signal the ``gcloud`` CLI fallback in ``_get_access_token``.
        """
        self._credentials = None
        config = getattr(self, "_config", None) or {}
        creds_path = (
            getattr(self, "credentials_file", None)
            or config.get("credentials_file")
            or config.get("credentials_path")
            or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        )

        if creds_path:
            # Explicit credentials file — errors must propagate so the user
            # doesn't silently fall back to a different identity.
            try:
                from google.oauth2 import service_account
            except ImportError:
                raise ImportError(
                    "A Vertex credentials file requires the google-auth package. "
                    "Install with: uv add google-auth or pip install google-auth"
                )
            self._credentials = service_account.Credentials.from_service_account_file(
                creds_path, scopes=_SCOPES
            )
        else:
            # ADC — best-effort; fall back to the gcloud CLI if unavailable.
            try:
                import google.auth

                self._credentials, _ = google.auth.default(scopes=_SCOPES)
            except ImportError:
                self._credentials = None
            except Exception:
                self._credentials = None

    def _get_access_token(self) -> str:
        """Get an OAuth 2.0 access token for Google Cloud APIs."""
        # Prefer resolved google-auth credentials (service account or ADC).
        creds = getattr(self, "_credentials", None)
        if creds is not None:
            try:
                from google.auth.transport.requests import Request

                if not creds.valid:
                    creds.refresh(Request())
                return creds.token
            except Exception as e:
                raise RuntimeError(f"Failed to refresh Vertex credentials: {e}")

        current_time = time.time()

        # Cached gcloud token still valid (5-minute buffer)?
        cached = self._access_token
        if cached and current_time < (self._token_expiry - 300):
            return cached

        try:
            # Last-resort fallback: the gcloud CLI.
            result = subprocess.run(
                ["gcloud", "auth", "application-default", "print-access-token"],
                capture_output=True,
                text=True,
                check=True,
            )
            self._access_token = result.stdout.strip()
            self._token_expiry = current_time + 3600  # tokens last ~1 hour
            return self._access_token
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "Failed to get a Vertex AI access token. Provide a service account "
                "key via credentials_file / credentials_path or "
                "GOOGLE_APPLICATION_CREDENTIALS, install google-auth for ADC, or "
                f"authenticate with 'gcloud auth application-default login': {e}"
            )
