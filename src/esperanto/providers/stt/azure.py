"""Azure OpenAI speech-to-text provider."""

import os
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, List, Optional, Union

import httpx

from esperanto.common_types import (
    Model,
    TranscriptionResponse,
    TranscriptionSegment,
)
from esperanto.providers.stt.base import SpeechToTextModel, _guess_audio_content_type
from esperanto.providers.stt.openai import _WHISPER_SEGMENT_METADATA_KEYS


@dataclass
class AzureSpeechToTextModel(SpeechToTextModel):
    """Azure OpenAI Speech-to-Text implementation using direct HTTP."""

    def __post_init__(self):
        """Initialize with Azure-specific configuration."""
        # Call parent's post_init to handle config initialization
        super().__post_init__()

        # Resolve configuration with priority: config dict → modality env var → generic env var
        self.api_key = (
            self._config.get("api_key") or
            os.getenv("AZURE_OPENAI_API_KEY_STT") or
            os.getenv("AZURE_OPENAI_API_KEY")
        )

        self.azure_endpoint = (
            self._config.get("azure_endpoint") or
            os.getenv("AZURE_OPENAI_ENDPOINT_STT") or
            os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        if self.azure_endpoint:
            self.azure_endpoint = self.azure_endpoint.rstrip("/")

        self.api_version = (
            self._config.get("api_version") or
            os.getenv("AZURE_OPENAI_API_VERSION_STT") or
            os.getenv("AZURE_OPENAI_API_VERSION")
        )

        # deployment_name is model_name for Azure
        self.deployment_name = self.model_name or self._get_default_model()

        # Validate required parameters
        if not self.api_key:
            raise ValueError(
                "Azure OpenAI API key not found. Set AZURE_OPENAI_API_KEY_STT "
                "or AZURE_OPENAI_API_KEY environment variable, or provide in config."
            )
        if not self.azure_endpoint:
            raise ValueError(
                "Azure OpenAI endpoint not found. Set AZURE_OPENAI_ENDPOINT_STT "
                "or AZURE_OPENAI_ENDPOINT environment variable, or provide in config."
            )
        if not self.api_version:
            raise ValueError(
                "Azure OpenAI API version not found. Set AZURE_OPENAI_API_VERSION_STT "
                "or AZURE_OPENAI_API_VERSION environment variable, or provide in config."
            )

        # Initialize httpx clients with configurable timeout
        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Azure API requests."""
        return {
            "api-key": self.api_key or "",  # Azure uses api-key, not Bearer
        }

    def _build_url(self, path: str) -> str:
        """Build Azure OpenAI URL with deployment name."""
        # Remove trailing slash from endpoint
        endpoint = self.azure_endpoint.rstrip('/')
        # Azure URL pattern: {endpoint}/openai/deployments/{deployment}/{path}?api-version={version}
        return f"{endpoint}/openai/deployments/{self.deployment_name}/{path}?api-version={self.api_version}"

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Azure OpenAI API error: {error_message}")

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "whisper-1"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "azure"

    def _get_models(self) -> List[Model]:
        """List all available models for this provider.

        Note: Azure doesn't have a models API endpoint - it uses deployments.
        Returns an empty list since model discovery isn't available.
        """
        return []

    def get_model_name(self) -> str:
        """Get the model name (deployment name for Azure)."""
        return self.deployment_name

    def _get_api_kwargs(
        self, language: Optional[str] = None, prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get kwargs for the Azure OpenAI Whisper request.

        Always requests ``verbose_json`` so segments and duration are returned,
        matching the OpenAI provider's Hot-Swap-First Defaults behavior. Azure
        OpenAI Whisper accepts the same request shape as OpenAI Whisper.
        """
        data: Dict[str, Any] = {
            "model": self.deployment_name,
            "response_format": "verbose_json",
        }
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        return data

    def _build_response(
        self,
        response_data: Dict[str, Any],
        language: Optional[str] = None,
    ) -> TranscriptionResponse:
        """Build a TranscriptionResponse from an Azure OpenAI Whisper ``verbose_json`` payload."""
        raw_segments = response_data.get("segments") or []
        segments: Optional[List[TranscriptionSegment]] = None
        if raw_segments:
            segments = [
                TranscriptionSegment(
                    text=segment.get("text", ""),
                    start=float(segment.get("start", 0.0)),
                    end=float(segment.get("end", 0.0)),
                    metadata={
                        key: segment[key]
                        for key in _WHISPER_SEGMENT_METADATA_KEYS
                        if key in segment
                    }
                    or None,
                )
                for segment in raw_segments
            ]

        duration_raw = response_data.get("duration")
        duration = float(duration_raw) if duration_raw is not None else None

        return TranscriptionResponse(
            text=response_data["text"],
            language=response_data.get("language") or language,
            duration=duration,
            model=self.deployment_name,
            provider=self.provider,
            segments=segments,
        )

    def transcribe(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        """Transcribe audio using Azure OpenAI."""
        url = self._build_url("audio/transcriptions")

        # Prepare API kwargs (always requests verbose_json)
        data = self._get_api_kwargs(language=language, prompt=prompt)

        # Handle file input
        if isinstance(audio_file, str):
            # For file path, open and send as multipart form data
            with open(audio_file, "rb") as f:
                files: Dict[str, Any] = {"file": (audio_file, f, _guess_audio_content_type(audio_file))}
                response = self.client.post(
                    url,
                    headers=self._get_headers(),
                    files=files,
                    data=data
                )
        else:
            # For BinaryIO, send the file object directly
            filename = getattr(audio_file, 'name', 'audio.mp3')
            files = {"file": (filename, audio_file, _guess_audio_content_type(filename))}
            response = self.client.post(
                url,
                headers=self._get_headers(),
                files=files,
                data=data
            )

        self._handle_error(response)
        response_data = response.json()

        return self._build_response(response_data, language=language)

    async def atranscribe(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        """Async transcribe audio using Azure OpenAI."""
        url = self._build_url("audio/transcriptions")

        # Prepare API kwargs (always requests verbose_json)
        data = self._get_api_kwargs(language=language, prompt=prompt)

        # Handle file input
        if isinstance(audio_file, str):
            # For file path, open and send as multipart form data
            with open(audio_file, "rb") as f:
                files: Dict[str, Any] = {"file": (audio_file, f, _guess_audio_content_type(audio_file))}
                response = await self.async_client.post(
                    url,
                    headers=self._get_headers(),
                    files=files,
                    data=data
                )
        else:
            # For BinaryIO, send the file object directly
            filename = getattr(audio_file, 'name', 'audio.mp3')
            files = {"file": (filename, audio_file, _guess_audio_content_type(filename))}
            response = await self.async_client.post(
                url,
                headers=self._get_headers(),
                files=files,
                data=data
            )

        self._handle_error(response)
        response_data = response.json()

        return self._build_response(response_data, language=language)
