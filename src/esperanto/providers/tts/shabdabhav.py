"""
Shabdabhav Esperanto-compatible Text-to-Speech provider via HTTP API.
https://github.com/Hardik94/shabdabhav
For: POST http://localhost:8000/docs
"""

import httpx
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import os
from .base import TextToSpeechModel, Voice, Model

class ShabdabhavTextToSpeechModel(TextToSpeechModel):
    """
    Text-to-speech provider for a local/remote HTTP TTS API compatible with
    the following curl sample:

    Supports multiple model including
    - parler-tts/parler-tts-mini-v1
    - piper-tts
    """
    DEFAULT_URL = "http://localhost:8000/v1/audio/speech"
    DEFAULT_MODEL = "parler-tts/parler-tts-mini-v1"
    CONTENT_TYPE = "audio/wav"

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """Initialize Shabdabhav TTS provider.
        
        Args:
            model_name: Name of the model to use (e.g. parler-tts/parler-tts-mini-v1)
            api_key: Shabdabhav API key. If not provided, will try to get from SHABDABHAV_API_KEY env var
            base_url: Optional base URL for the API
            **kwargs: Additional configuration options including voice_settings
        """
        
        super().__init__(
            model_name=model_name or self.DEFAULT_MODEL,
            base_url=base_url or self.DEFAULT_URL,
            **kwargs
        )
        # self.model_name = model_name
        # self.base_url = base_url or self.DEFAULT_URL
        self.client = httpx.Client(timeout=60.0)
        self.async_client = httpx.AsyncClient(timeout=60.0)
        # self._available_voices = None

    def generate_speech(
        self,
        text: str,
        # model_name: str,
        voice: Optional[str] = "",
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> bytes:
        """
        Synchronously request TTS audio for the given text, model and description.
        Returns: audio bytes (WAV or MP3, as served by API)
        """
        payload: Dict[str, Any] = {
            "text": text,
            "model_id": self.model_name
        }
        if ("parler" in self.model_name):
            payload["description"] = kwargs.get('description', '')
        elif ("piper" in self.model_name):
            payload["voice"] = voice

        headers = {"Content-Type": "application/json"}

        response = self.client.post(
            self.base_url,
            json=payload,
            headers=headers
        )
        if response.status_code != 200:
            raise RuntimeError(f"TTS API error: {response.status_code} {response.text}")

        audio_bytes = response.content

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_bytes)

        return audio_bytes

    async def agenerate_speech(
        self,
        text: str,
        # model_name: str,
        # description: Optional[str] = "",
        voice: Optional[str] = "",
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> bytes:
        """
        Asynchronously request TTS audio for the given text, model and description.
        Returns: audio bytes (WAV or MP3, as served by API)
        """
        payload: Dict[str, Any] = {
            "text": text,
            "model_id": self.model_name
            # "description": description or ""
        }
        if ("parler" in self.model_name):
            payload["description"] = kwargs.get('description', '')
        elif ("piper" in self.model_name):
            payload["voice"] = voice
        headers = {"Content-Type": "application/json"}

        response = await self.async_client.post(
            self.base_url,
            json=payload,
            headers=headers
        )
        if (response.status_code != 200):
            raise RuntimeError(f"TTS API error: {response.status_code} {response.text}")

        audio_bytes = response.content

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_bytes)

        return audio_bytes

    @property
    def available_voices(self) -> Dict[str, Voice]:
        """Get available voices."""
        # response = self.client.get(
        #     f"{self.base_url}/v1/voices",
        #     headers=self._get_headers()
        # )
        # self._handle_error(response)
        
        # response_data = response.json()
        # voices = {}
        # for voice_data in response_data["voices"]:
        #     voices[voice_data["voice_id"]] = Voice(
        #         name=voice_data["name"],
        #         id=voice_data["voice_id"],
        #         gender=voice_data.get("labels", {}).get("gender", "unknown").upper(),
        #         language_code=voice_data.get("labels", {}).get("language", "en"),
        #         description=voice_data.get("description", ""),
        #         preview_url=voice_data.get("preview_url", "")
        #     )
        # return voices
        return {}

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        return []  # For now, return empty list as requested
