"""Google GenAI speech-to-text provider implementation."""

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import httpx

from esperanto.common_types import TranscriptionResponse
from esperanto.providers.stt.base import Model, SpeechToTextModel


@dataclass
class GoogleSpeechToTextModel(SpeechToTextModel):
    """Google GenAI speech-to-text model implementation.

    Uses the Gemini API for audio transcription via the generateContent endpoint.
    This is NOT Cloud Speech-to-Text API v2 Chirp 3, but Gemini's audio understanding.
    """

    def __post_init__(self):
        """Initialize HTTP clients."""
        # Call parent's post_init to handle config initialization
        super().__post_init__()

        # Get API key - check both GOOGLE_API_KEY and GEMINI_API_KEY
        self._api_key = (
            self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )
        if not self._api_key:
            raise ValueError(
                "Google API key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )

        # Set base URL - consistent with other Google providers
        base_host = os.getenv("GEMINI_API_BASE_URL") or "https://generativelanguage.googleapis.com"
        self.base_url = f"{base_host}/v1beta"

        # Initialize HTTP clients with configurable timeout
        self._create_http_clients()

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "gemini-2.5-flash"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "google"

    def _get_models(self) -> List[Model]:
        """List all available models for this provider.

        Returns hardcoded list of Gemini models that support audio transcription.
        """
        return [
            Model(
                id="gemini-2.5-flash",
                owned_by="Google",
                context_window=1000000,
            ),
            Model(
                id="gemini-2.0-flash",
                owned_by="Google",
                context_window=1000000,
            ),
        ]

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Google API requests."""
        return {
            "Content-Type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Google API error: {error_message}")

    def _get_mime_type(self, file_path: str) -> str:
        """Get MIME type from file extension.

        Args:
            file_path: Path to audio file

        Returns:
            MIME type string

        Raises:
            ValueError: If file extension is not supported
        """
        extension = Path(file_path).suffix.lower()
        mime_types = {
            ".mp3": "audio/mp3",
            ".wav": "audio/wav",
            ".aiff": "audio/aiff",
            ".aac": "audio/aac",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
        }

        if extension not in mime_types:
            raise ValueError(
                f"Unsupported audio format: {extension}. "
                f"Supported formats: {', '.join(mime_types.keys())}"
            )

        return mime_types[extension]

    def _encode_audio(self, audio_file: Union[str, BinaryIO]) -> Tuple[bytes, str]:
        """Read and base64 encode audio file.

        Args:
            audio_file: Path to audio file or file-like object

        Returns:
            Tuple of (audio_bytes, file_path_or_name)
        """
        if isinstance(audio_file, str):
            # File path
            with open(audio_file, "rb") as f:
                audio_bytes = f.read()
            return audio_bytes, audio_file
        else:
            # BinaryIO
            audio_bytes = audio_file.read()
            filename = getattr(audio_file, 'name', 'audio_input.mp3')
            return audio_bytes, filename

    def _build_prompt(
        self,
        language: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> str:
        """Build transcription prompt for Gemini API.

        Args:
            language: Optional language code to hint the transcription
            prompt: Optional user-provided prompt to guide transcription

        Returns:
            Complete prompt string
        """
        base_prompt = "Generate a transcript of the speech in this audio file."

        parts = [base_prompt]

        if language:
            parts.append(f"The audio is in {language} language.")

        if prompt:
            parts.append(prompt)

        return " ".join(parts)

    def _parse_response(self, response_data: Dict[str, Any]) -> str:
        """Parse transcription text from Gemini API response.

        Args:
            response_data: JSON response from Gemini API

        Returns:
            Transcribed text

        Raises:
            RuntimeError: If response structure is unexpected
        """
        try:
            text = response_data["candidates"][0]["content"]["parts"][0]["text"]
            return text
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"Failed to parse transcription from response: {e}. "
                f"Response structure: {list(response_data.keys())}"
            )

    def _prepare_request_payload(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """Prepare request payload for transcription.

        Args:
            audio_file: Path to audio file or file-like object
            language: Optional language code
            prompt: Optional text to guide transcription

        Returns:
            Tuple of (payload dict, model_name)
        """
        # Encode audio
        audio_bytes, file_identifier = self._encode_audio(audio_file)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Get MIME type
        mime_type = self._get_mime_type(file_identifier)

        # Build prompt
        text_prompt = self._build_prompt(language, prompt)

        # Construct request payload
        model_name = self.get_model_name()
        payload = {
            "contents": [{
                "parts": [
                    {"text": text_prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": audio_base64
                        }
                    }
                ]
            }]
        }

        return payload, model_name

    def transcribe(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        """Transcribe audio to text using Google Gemini API.

        Args:
            audio_file: Path to audio file or file-like object
            language: Optional language code (e.g., 'en', 'es', 'pt')
            prompt: Optional text to guide the transcription

        Returns:
            TranscriptionResponse containing the transcribed text and metadata
        """
        # Prepare request
        payload, model_name = self._prepare_request_payload(audio_file, language, prompt)

        # Make HTTP request
        response = self.client.post(
            f"{self.base_url}/models/{model_name}:generateContent?key={self._api_key}",
            headers=self._get_headers(),
            json=payload
        )

        # Handle errors
        self._handle_error(response)

        # Parse response
        response_data = response.json()
        text = self._parse_response(response_data)

        return TranscriptionResponse(
            text=text,
            language=language,  # Pass through from request, not detected
            model=model_name,
        )

    async def atranscribe(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        """Async transcribe audio to text using Google Gemini API.

        Args:
            audio_file: Path to audio file or file-like object
            language: Optional language code (e.g., 'en', 'es', 'pt')
            prompt: Optional text to guide the transcription

        Returns:
            TranscriptionResponse containing the transcribed text and metadata
        """
        # Prepare request
        payload, model_name = self._prepare_request_payload(audio_file, language, prompt)

        # Make async HTTP request
        response = await self.async_client.post(
            f"{self.base_url}/models/{model_name}:generateContent?key={self._api_key}",
            headers=self._get_headers(),
            json=payload
        )

        # Handle errors
        self._handle_error(response)

        # Parse response
        response_data = response.json()
        text = self._parse_response(response_data)

        return TranscriptionResponse(
            text=text,
            language=language,  # Pass through from request, not detected
            model=model_name,
        )
