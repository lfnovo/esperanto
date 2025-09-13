"""OpenAI-compatible language model implementation."""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from esperanto.common_types import Model
# from esperanto.providers.llm.openai import OpenAILanguageModel
from esperanto.utils.logging import logger
from esperanto.providers.tts.openai import OpenAITextToSpeechModel

# import httpx
from typing import Optional, Dict, Any, Union
from pathlib import Path
from .base import TextToSpeechModel, Voice, Model, AudioResponse


class OpenAICompatibleTextToSpeechModel(OpenAITextToSpeechModel):
    """OpenAI-compatible language model implementation for custom endpoints."""

    base_url: Optional[str] = None
    api_key: Optional[str] = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """Initialize OpenAI-Compatible TTS provider.
        
        Args:
            model_name: Name of the model to use (default: )
            api_key: OpenAI API key. If not provided, will try to get from OPENAI_API_KEY env var
            base_url: Optional base URL for the API
            **kwargs: Additional configuration options
        """
        
        super().__init__(
            model_name=model_name,
            api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY") or api_key,
            base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL") or base_url,
            config=kwargs
        )
        # super().__init__(
        #     model_name=model_name,
        #     api_key=self.api_key,
        #     base_url=self.base_url,
        #     **kwargs
        # )
        
        # Configuration precedence: Factory config > Environment variables > Default
        self.base_url = (
            base_url or 
            os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        )
        self.api_key = (
            api_key or 
            os.getenv("OPENAI_COMPATIBLE_API_KEY")
        )
        # logger.debug(f"api key added and url set: {self.base_url}")

    
    def _handle_error(self, response) -> None:
        """Handle HTTP error responses with graceful degradation."""
        if response.status_code >= 400:
            # Log original response for debugging
            logger.debug(f"OpenAI-compatible endpoint error: {response.text}")
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"OpenAI API error: {error_message}")

        
    @property
    def models(self) -> List[Model]:
        """List all available models for this provider.
        
        Note: This attempts to fetch models from the /models endpoint.
        If the endpoint doesn't support this, it will return an empty list.
        """
        try:
            response = self.client.get(
                f"{self.base_url}/models",
                headers=self._get_headers()
            )
            self._handle_error(response)
            
            models_data = response.json()
            return [
                Model(
                    id=model["id"],
                    owned_by=model.get("owned_by", "custom"),
                    context_window=None,  # Audio models don't have context windows
                    type="text_to_speech",
                )
                for model in models_data.get("data", [])
            ]
        except Exception as e:
            # Log the error but don't fail completely
            logger.debug(f"Could not fetch models from OpenAI-compatible endpoint: {e}")
            return []
    
    @property
    def available_voices(self) -> Dict[str, Voice]:
        """Get available voices from OpenAI Compatible TTS.

        Returns:
            Dict[str, Voice]: Dictionary of available voices with their information
        """
        try:
            response = self.client.get(
                f"{self.base_url}/audio/voices",
                headers=self._get_headers()
            )
            self._handle_error(response)

            voices_data = response.json()
            voices_rep = {}
            for voice_id in voices_data.get("voices", []):
                if (".onnx" in voice_id):  # Condition for piper-tts modesl
                    voices_rep[f"{voice_id}"] = Voice(
                        name=f"{voice_id.split('/')[2]}",
                        id=f"{voice_id.split('/')[2]}",
                        gender="NEUTRAL",
                        language_code=f"{voice_id.split('/')[1]}",
                        description=f"{voice_id.split('/')[3]}"
                    )
            return voices_rep
        
        except Exception as e:
            logger.debug(f"could not fetch voices from OpenAI-Compatible endpoints: {e}")
            return {}

    def generate_speech(
        self,
        text: str,
        voice: str,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech from text using OpenAI-Compatible's Text-to-Speech API.

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
                "text": text,
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
        voice: str,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech from text using OpenAI-Compatible's Text-to-Speech API asynchronously.

        Args:
            text: Text to convert to speech
            voice: Voice to use (default: )
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
                "text": text,
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