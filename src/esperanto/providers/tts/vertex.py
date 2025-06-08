"""Google Vertex AI Text-to-Speech provider implementation."""
import base64
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

from .base import AudioResponse, Model, TextToSpeechModel, Voice


class VertexTextToSpeechModel(TextToSpeechModel):
    """Google Cloud Text-to-Speech provider implementation via Vertex AI.
    
    Uses the Cloud Text-to-Speech API for text-to-speech generation.
    """
    
    def __init__(
        self,
        model_name: str = "standard",
        base_url: Optional[str] = None,
        vertex_project: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initialize Google Cloud TTS model.

        Args:
            model_name: Model name to use
            base_url: Base URL for the API
            vertex_project: Google Cloud project ID
            **kwargs: Additional arguments passed to the provider
        """
        super().__init__(model_name=model_name, **kwargs)
        
    def __post_init__(self):
        """Initialize HTTP clients and authentication."""
        super().__post_init__()
        
        # Get project ID
        self.project_id = getattr(self, 'vertex_project', None) or os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        
        if not self.project_id:
            raise ValueError(
                "Google Cloud project ID not found. Please set VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT environment variable."
            )

        # Set base URL for Cloud Text-to-Speech API
        self.base_url = "https://texttospeech.googleapis.com/v1"

        # Initialize HTTP clients
        self.client = httpx.Client(timeout=30.0)
        self.async_client = httpx.AsyncClient(timeout=30.0)

        # Cache for access token
        self._access_token = None
        self._token_expiry = 0

    def _get_access_token(self) -> str:
        """Get OAuth 2.0 access token for Google Cloud APIs."""
        current_time = time.time()
        
        # Check if token is still valid (with 5-minute buffer)
        if self._access_token and current_time < (self._token_expiry - 300):
            return self._access_token
            
        try:
            # Use gcloud to get access token
            result = subprocess.run(
                ["gcloud", "auth", "application-default", "print-access-token"],
                capture_output=True,
                text=True,
                check=True
            )
            self._access_token = result.stdout.strip()
            # Tokens typically expire in 1 hour
            self._token_expiry = current_time + 3600
            return self._access_token
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to get access token. Make sure you're authenticated with 'gcloud auth application-default login': {e}"
            )

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Cloud Text-to-Speech API requests."""
        return {
            "Authorization": f"Bearer {self._get_access_token()}",
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
            raise RuntimeError(f"Cloud Text-to-Speech API error: {error_message}")

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "standard"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "vertex"

    @property
    def available_voices(self) -> Dict[str, Voice]:
        """Get available voices."""
        # Return some common voices - in practice, you could fetch from the API
        voices = {
            "en-US-Standard-A": Voice(
                name="en-US-Standard-A",
                id="en-US-Standard-A",
                gender="FEMALE",
                language_code="en-US",
                description="Standard English (US) Female Voice A"
            ),
            "en-US-Standard-B": Voice(
                name="en-US-Standard-B",
                id="en-US-Standard-B",
                gender="MALE",
                language_code="en-US",
                description="Standard English (US) Male Voice B"
            ),
            "en-US-Neural2-A": Voice(
                name="en-US-Neural2-A",
                id="en-US-Neural2-A",
                gender="FEMALE",
                language_code="en-US",
                description="Neural2 English (US) Female Voice A"
            ),
            "en-US-Neural2-B": Voice(
                name="en-US-Neural2-B",
                id="en-US-Neural2-B",
                gender="MALE",
                language_code="en-US",
                description="Neural2 English (US) Male Voice B"
            ),
            "en-US-Wavenet-A": Voice(
                name="en-US-Wavenet-A",
                id="en-US-Wavenet-A",
                gender="FEMALE",
                language_code="en-US",
                description="WaveNet English (US) Female Voice A"
            ),
            "en-US-Wavenet-B": Voice(
                name="en-US-Wavenet-B",
                id="en-US-Wavenet-B",
                gender="MALE",
                language_code="en-US",
                description="WaveNet English (US) Male Voice B"
            ),
        }
        return voices

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        return [
            Model(
                id="standard",
                owned_by="Google",
                context_window=None,
                type="text_to_speech",
            ),
            Model(
                id="wavenet",
                owned_by="Google",
                context_window=None,
                type="text_to_speech",
            ),
            Model(
                id="neural2",
                owned_by="Google",
                context_window=None,
                type="text_to_speech",
            ),
            Model(
                id="studio",
                owned_by="Google",
                context_window=None,
                type="text_to_speech",
            ),
        ]

    def generate_speech(
        self,
        text: str,
        voice: str,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech from text.

        Args:
            text: Text to convert to speech
            voice: Voice ID to use (e.g., "en-US-Standard-A")
            output_file: Optional path to save the audio file
            **kwargs: Additional arguments passed to the provider

        Returns:
            AudioResponse object containing the audio data and metadata
        """
        # Extract language code from voice (e.g., "en-US" from "en-US-Standard-A")
        language_code = voice.split("-")[0] + "-" + voice.split("-")[1]
        
        # Prepare request payload
        payload = {
            "input": {
                "text": text
            },
            "voice": {
                "languageCode": language_code,
                "name": voice
            },
            "audioConfig": {
                "audioEncoding": "MP3"
            }
        }

        # Make HTTP request
        response = self.client.post(
            f"{self.base_url}/text:synthesize",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)
        
        response_data = response.json()
        
        # Extract audio data from response
        audio_data_b64 = response_data["audioContent"]
        audio_data = base64.b64decode(audio_data_b64)

        response_audio = AudioResponse(
            audio_data=audio_data,
            content_type="audio/mp3",
            model=self.model_name or self._get_default_model(),
            voice=voice,
            provider="vertex"
        )

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_data)

        return response_audio

    async def agenerate_speech(
        self,
        text: str,
        voice: str,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech from text asynchronously.

        Args:
            text: Text to convert to speech
            voice: Voice ID to use (e.g., "en-US-Standard-A")
            output_file: Optional path to save the audio file
            **kwargs: Additional arguments passed to the provider

        Returns:
            AudioResponse object containing the audio data and metadata
        """
        # Extract language code from voice (e.g., "en-US" from "en-US-Standard-A")
        language_code = voice.split("-")[0] + "-" + voice.split("-")[1]
        
        # Prepare request payload
        payload = {
            "input": {
                "text": text
            },
            "voice": {
                "languageCode": language_code,
                "name": voice
            },
            "audioConfig": {
                "audioEncoding": "MP3"
            }
        }

        # Make async HTTP request
        response = await self.async_client.post(
            f"{self.base_url}/text:synthesize",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)
        
        response_data = response.json()
        
        # Extract audio data from response
        audio_data_b64 = response_data["audioContent"]
        audio_data = base64.b64decode(audio_data_b64)

        response_audio = AudioResponse(
            audio_data=audio_data,
            content_type="audio/mp3",
            model=self.model_name or self._get_default_model(),
            voice=voice,
            provider="vertex"
        )

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_data)

        return response_audio