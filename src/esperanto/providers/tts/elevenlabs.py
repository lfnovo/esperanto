"""ElevenLabs Text-to-Speech provider implementation."""
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import httpx

from .base import AudioResponse, Model, TextToSpeechModel, Voice


class ElevenLabsTextToSpeechModel(TextToSpeechModel):
    """ElevenLabs Text-to-Speech provider implementation.
    
    Supports multiple models including:
    - eleven_multilingual_v2: Multilingual model
    - eleven_monolingual_v1: English-only model
    - eleven_turbo_v2: Faster, lower-quality model
    """
    
    DEFAULT_MODEL = "eleven_multilingual_v2"
    DEFAULT_VOICE = "Aria"  # One of their default voices
    PROVIDER = "elevenlabs"
    
    # Default voice settings
    DEFAULT_VOICE_SETTINGS = {
        "stability": 0.5,  # Range 0-1
        "similarity_boost": 0.75,  # Range 0-1
        "style": 0.0,  # Range 0-1
        "use_speaker_boost": True
    }
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """Initialize ElevenLabs TTS provider.
        
        Args:
            model_name: Name of the model to use
            api_key: ElevenLabs API key. If not provided, will try to get from ELEVENLABS_API_KEY env var
            base_url: Optional base URL for the API
            **kwargs: Additional configuration options including voice_settings
        """
        api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ElevenLabs API key not provided. Set ELEVENLABS_API_KEY environment variable or pass api_key parameter.")

        super().__init__(
            model_name=model_name or self.DEFAULT_MODEL,
            api_key=api_key,
            base_url=base_url or "https://api.elevenlabs.io",
            config=kwargs
        )
        
        self.voice_settings = {
            **self.DEFAULT_VOICE_SETTINGS,
            **(kwargs.get("voice_settings", {}) or {})
        }
        
        # Initialize HTTP clients
        self.client = httpx.Client(timeout=30.0)
        self.async_client = httpx.AsyncClient(timeout=30.0)
        
        # Cache available voices
        self._available_voices = None

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for ElevenLabs API requests."""
        return {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("detail", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"ElevenLabs API error: {error_message}")

    def generate_speech(self, text: str, voice: str, output_file: Optional[Union[str, Path]] = None) -> AudioResponse:
        """Generate speech synchronously."""
        # Prepare request payload
        payload = {
            "text": text,
            "model_id": self.model_name,
            "voice_settings": self.voice_settings
        }

        # Make HTTP request
        response = self.client.post(
            f"{self.base_url}/v1/text-to-speech/{voice}?output_format=mp3_44100_128",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        audio_bytes = response.content

        response_audio = AudioResponse(
            audio_data=audio_bytes,
            content_type="audio/mp3",
            model=self.model_name,
            voice=voice,
            provider="elevenlabs"
        )

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_bytes)

        return response_audio

    async def agenerate_speech(self, text: str, voice: str, output_file: Optional[Union[str, Path]] = None) -> AudioResponse:
        """Generate speech asynchronously."""
        # Prepare request payload
        payload = {
            "text": text,
            "model_id": self.model_name,
            "voice_settings": self.voice_settings
        }

        # Make async HTTP request
        response = await self.async_client.post(
            f"{self.base_url}/v1/text-to-speech/{voice}?output_format=mp3_44100_128",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        audio_bytes = response.content

        response_audio = AudioResponse(
            audio_data=audio_bytes,
            content_type="audio/mp3",
            model=self.model_name,
            voice=voice,
            provider="elevenlabs"
        )

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_bytes)

        return response_audio

    @property
    def available_voices(self) -> Dict[str, Voice]:
        """Get available voices."""
        response = self.client.get(
            f"{self.base_url}/v1/voices",
            headers=self._get_headers()
        )
        self._handle_error(response)
        
        response_data = response.json()
        voices = {}
        for voice_data in response_data["voices"]:
            voices[voice_data["voice_id"]] = Voice(
                name=voice_data["name"],
                id=voice_data["voice_id"],
                gender=voice_data.get("labels", {}).get("gender", "unknown").upper(),
                language_code=voice_data.get("labels", {}).get("language", "en"),
                description=voice_data.get("description", ""),
                preview_url=voice_data.get("preview_url", "")
            )
        return voices

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        return []  # For now, return empty list as requested

    def generate_multi_speaker_speech(
        self,
        text: str,
        speaker_configs: List[Dict[str, str]],
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech with multiple speakers using ElevenLabs text-to-dialogue.

        Args:
            text: Text containing the conversation with speaker names
            speaker_configs: List of dicts with 'speaker' and 'voice' keys
            output_file: Optional path to save the audio file
            **kwargs: Additional arguments passed to the provider

        Returns:
            AudioResponse object containing the audio data and metadata
            
        Example:
            speaker_configs = [
                {"speaker": "Joe", "voice": "JBFqnCBsd6RMkjVDRZzb"},
                {"speaker": "Jane", "voice": "Aw4FAjKCGjjNkVhN1Xmq"}
            ]
        """
        # Create speaker mapping
        speaker_to_voice = {config["speaker"]: config["voice"] for config in speaker_configs}
        
        # Parse the text to extract dialogue lines
        dialogue_inputs = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                speaker, dialogue = line.split(':', 1)
                speaker = speaker.strip()
                dialogue = dialogue.strip()
                
                if speaker in speaker_to_voice:
                    dialogue_inputs.append({
                        "text": dialogue,
                        "voice_id": speaker_to_voice[speaker]
                    })
        
        if not dialogue_inputs:
            raise ValueError("No valid dialogue found. Text should contain speaker names followed by ':' and their dialogue.")
        
        # Prepare request payload
        payload = {
            "inputs": dialogue_inputs,
            "model_id": kwargs.get("model_id", "eleven_v3"),
            "output_format": kwargs.get("output_format", "mp3_44100_128")
        }
        
        # Add optional settings if provided
        if "settings" in kwargs:
            payload["settings"] = kwargs["settings"]
        if "seed" in kwargs:
            payload["seed"] = kwargs["seed"]
        if "pronunciation_dictionary_locators" in kwargs:
            payload["pronunciation_dictionary_locators"] = kwargs["pronunciation_dictionary_locators"]

        # Make HTTP request
        response = self.client.post(
            f"{self.base_url}/v1/text-to-dialogue",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        audio_bytes = response.content

        response_audio = AudioResponse(
            audio_data=audio_bytes,
            content_type="audio/mp3",
            model=payload["model_id"],
            voice="multi-speaker",
            provider="elevenlabs"
        )

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_bytes)

        return response_audio

    async def agenerate_multi_speaker_speech(
        self,
        text: str,
        speaker_configs: List[Dict[str, str]],
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech with multiple speakers asynchronously using ElevenLabs text-to-dialogue.

        Args:
            text: Text containing the conversation with speaker names
            speaker_configs: List of dicts with 'speaker' and 'voice' keys
            output_file: Optional path to save the audio file
            **kwargs: Additional arguments passed to the provider

        Returns:
            AudioResponse object containing the audio data and metadata
        """
        # Create speaker mapping
        speaker_to_voice = {config["speaker"]: config["voice"] for config in speaker_configs}
        
        # Parse the text to extract dialogue lines
        dialogue_inputs = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                speaker, dialogue = line.split(':', 1)
                speaker = speaker.strip()
                dialogue = dialogue.strip()
                
                if speaker in speaker_to_voice:
                    dialogue_inputs.append({
                        "text": dialogue,
                        "voice_id": speaker_to_voice[speaker]
                    })
        
        if not dialogue_inputs:
            raise ValueError("No valid dialogue found. Text should contain speaker names followed by ':' and their dialogue.")
        
        # Prepare request payload
        payload = {
            "inputs": dialogue_inputs,
            "model_id": kwargs.get("model_id", "eleven_v3"),
            "output_format": kwargs.get("output_format", "mp3_44100_128")
        }
        
        # Add optional settings if provided
        if "settings" in kwargs:
            payload["settings"] = kwargs["settings"]
        if "seed" in kwargs:
            payload["seed"] = kwargs["seed"]
        if "pronunciation_dictionary_locators" in kwargs:
            payload["pronunciation_dictionary_locators"] = kwargs["pronunciation_dictionary_locators"]

        # Make async HTTP request
        response = await self.async_client.post(
            f"{self.base_url}/v1/text-to-dialogue",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        audio_bytes = response.content

        response_audio = AudioResponse(
            audio_data=audio_bytes,
            content_type="audio/mp3",
            model=payload["model_id"],
            voice="multi-speaker",
            provider="elevenlabs"
        )

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_bytes)

        return response_audio

    def get_supported_tags(self) -> List[str]:
        """Get list of supported SSML tags.
        
        ElevenLabs has limited SSML support compared to other providers.
        """
        return ["speak", "break", "emphasis", "prosody"]
