"""OpenAI Text-to-Speech provider implementation."""
import os
import asyncio
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

import httpx

from .base import TextToSpeechModel, AudioResponse, Voice, Model


class OpenAITextToSpeechModel(TextToSpeechModel):
    """OpenAI Text-to-Speech provider implementation.
    
    Supports the TTS-1 model with multiple voice options.
    Available voices: alloy, echo, fable, onyx, nova, shimmer
    """
    
    DEFAULT_MODEL = "tts-1"
    DEFAULT_VOICE = "alloy"
    AVAILABLE_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    PROVIDER = "openai"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """Initialize OpenAI TTS provider.
        
        Args:
            model_name: Name of the model to use (default: tts-1)
            api_key: OpenAI API key. If not provided, will try to get from OPENAI_API_KEY env var
            base_url: Optional base URL for the API
            **kwargs: Additional configuration options
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            config=kwargs
        )
        
        # Set base URL
        self.base_url = self.base_url or "https://api.openai.com/v1"
        
        # Initialize HTTP clients
        self.client = httpx.Client(timeout=30.0)
        self.async_client = httpx.AsyncClient(timeout=30.0)

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for OpenAI API requests."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Organization is optional for TTS models
        if hasattr(self, 'organization') and self.organization:
            headers["OpenAI-Organization"] = self.organization
        return headers

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"OpenAI API error: {error_message}")

    @property
    def available_voices(self) -> Dict[str, Voice]:
        """Get available voices from OpenAI TTS.

        Returns:
            Dict[str, Voice]: Dictionary of available voices with their information
        """
        voices = {
            "alloy": Voice(
                name="alloy",
                id="alloy",
                gender="NEUTRAL",
                language_code="en-US",
                description="Neutral and balanced voice"
            ),
            "echo": Voice(
                name="echo",
                id="echo",
                gender="MALE",
                language_code="en-US",
                description="Mature and deep voice"
            ),
            "fable": Voice(
                name="fable",
                id="fable",
                gender="FEMALE",
                language_code="en-US",
                description="Warm and expressive voice"
            ),
            "onyx": Voice(
                name="onyx",
                id="onyx",
                gender="MALE",
                language_code="en-US",
                description="Smooth and authoritative voice"
            ),
            "nova": Voice(
                name="nova",
                id="nova",
                gender="FEMALE",
                language_code="en-US",
                description="Energetic and bright voice"
            ),
            "shimmer": Voice(
                name="shimmer",
                id="shimmer",
                gender="FEMALE",
                language_code="en-US",
                description="Clear and professional voice"
            ),
        }
        return voices

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        response = self.client.get(
            f"{self.base_url}/models",
            headers=self._get_headers()
        )
        self._handle_error(response)
        
        models_data = response.json()
        return [
            Model(
                id=model["id"],
                owned_by=model.get("owned_by", "openai"),
                context_window=None,  # Audio models don't have context windows
                type="text_to_speech"
            )
            for model in models_data["data"]
            if model["id"].startswith("tts")
        ]

    def generate_speech(
        self,
        text: str,
        voice: str = "alloy",
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech from text using OpenAI's Text-to-Speech API.

        Args:
            text: Text to convert to speech
            voice: Voice to use (default: "alloy")
            output_file: Optional path to save the audio file
            **kwargs: Additional parameters to pass to the OpenAI API

        Returns:
            AudioResponse containing the audio data and metadata

        Raises:
            RuntimeError: If speech generation fails
        """
        try:
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "voice": voice,
                "input": text,
                **kwargs
            }

            # Generate speech
            response = self.client.post(
                f"{self.base_url}/audio/speech",
                headers=self._get_headers(),
                json=payload
            )
            self._handle_error(response)

            # Get audio data (binary content)
            audio_data = response.content
            
            # Save to file if specified
            if output_file:
                output_file = Path(output_file)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_bytes(audio_data)

            return AudioResponse(
                audio_data=audio_data,
                content_type="audio/mp3",
                model=self.model_name,
                voice=voice,
                provider=self.PROVIDER,
                metadata={"text": text}
            )

        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {str(e)}") from e

    async def agenerate_speech(
        self,
        text: str,
        voice: str = "alloy",
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech from text using OpenAI's Text-to-Speech API asynchronously.

        Args:
            text: Text to convert to speech
            voice: Voice to use (default: "alloy")
            output_file: Optional path to save the audio file
            **kwargs: Additional parameters to pass to the OpenAI API

        Returns:
            AudioResponse containing the audio data and metadata

        Raises:
            RuntimeError: If speech generation fails
        """
        try:
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "voice": voice,
                "input": text,
                **kwargs
            }

            # Generate speech
            response = await self.async_client.post(
                f"{self.base_url}/audio/speech",
                headers=self._get_headers(),
                json=payload
            )
            self._handle_error(response)

            # Get audio data (binary content)
            audio_data = response.content
            
            # Save to file if specified
            if output_file:
                output_file = Path(output_file)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_bytes(audio_data)

            return AudioResponse(
                audio_data=audio_data,
                content_type="audio/mp3",
                model=self.model_name,
                voice=voice,
                provider=self.PROVIDER,
                metadata={"text": text}
            )

        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {str(e)}") from e
