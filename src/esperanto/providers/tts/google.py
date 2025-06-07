"""Google GenAI Text-to-Speech provider implementation."""
import base64
import io
import os
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

from .base import AudioResponse, Model, TextToSpeechModel, Voice


class GoogleTextToSpeechModel(TextToSpeechModel):
    """Google GenAI Text-to-Speech provider implementation.
    
    Uses the Gemini API for text-to-speech generation.
    Supports single voice and multi-speaker conversations.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-preview-tts",
        base_url: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initialize Google GenAI TTS model.

        Args:
            model_name: Model name to use
            base_url: Base URL for the API
            **kwargs: Additional arguments passed to the provider
        """
        super().__init__(model_name=model_name, **kwargs)
        
    def __post_init__(self):
        """Initialize HTTP clients."""
        super().__post_init__()
        
        # Get API key
        self.api_key = (
            self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Google API key not found. Please set GOOGLE_API_KEY environment variable."
            )

        # Set base URL
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

        # Initialize HTTP clients
        self.client = httpx.Client(timeout=30.0)
        self.async_client = httpx.AsyncClient(timeout=30.0)

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

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "gemini-2.5-flash-preview-tts"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "google"

    def _convert_pcm_to_wav(self, pcm_data: bytes) -> bytes:
        """Convert PCM audio data to WAV format.
        
        Google returns 16-bit PCM at 24kHz sample rate.
        """
        # Create a WAV file in memory
        wav_buffer = io.BytesIO()
        
        # PCM format parameters from Google's API
        sample_rate = 24000  # 24kHz
        sample_width = 2     # 16-bit = 2 bytes
        channels = 1         # Mono
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        
        wav_buffer.seek(0)
        return wav_buffer.read()

    @property
    def available_voices(self) -> Dict[str, Voice]:
        """Get available voices."""
        # Google GenAI TTS has predefined voices
        voices = {
            "achernar": Voice(
                name="UpbeatAchernar",
                id="achernar",
                gender="FEMALE",
                description="UpbeatAchernar"
            ),
            "achird": Voice(
                name="ForwardAchird",
                id="achird",
                gender="NEUTRAL",
                description="ForwardAchird"
            ),
            "algenib": Voice(
                name="ClearAlgenib",
                id="algenib",
                gender="MALE",
                description="ClearAlgenib"
            ),
            "algieba": Voice(
                name="Easy-goingAlgieba",
                id="algieba",
                gender="MALE",
                description="Easy-goingAlgieba"
            ),
            "alnilam": Voice(
                name="SoftAlnilam",
                id="alnilam",
                gender="MALE",
                description="SoftAlnilam"
            ),
            "aoede": Voice(
                name="FirmAoede",
                id="aoede",
                gender="FEMALE",
                description="FirmAoede"
            ),
            "autonoe": Voice(
                name="Easy-goingAutonoe",
                id="autonoe",
                gender="FEMALE",
                description="Easy-goingAutonoe"
            ),
            "callirrhoe": Voice(
                name="BreezyCallirrhoe",
                id="callirrhoe",
                gender="FEMALE",
                description="BreezyCallirrhoe"
            ),
            "charon": Voice(
                name="UpbeatCharon",
                id="charon",
                gender="MALE",
                description="UpbeatCharon"
            ),
            "despina": Voice(
                name="SmoothDespina",
                id="despina",
                gender="FEMALE",
                description="SmoothDespina"
            ),
            "enceladus": Voice(
                name="BrightEnceladus",
                id="enceladus",
                gender="MALE",
                description="BrightEnceladus"
            ),
            "erinome": Voice(
                name="SmoothErinome",
                id="erinome",
                gender="FEMALE",
                description="SmoothErinome"
            ),
            "fenrir": Voice(
                name="FirmFenrir",
                id="fenrir",
                gender="MALE",
                description="FirmFenrir"
            ),
            "gacrux": Voice(
                name="EvenGacrux",
                id="gacrux",
                gender="FEMALE",
                description="EvenGacrux"
            ),
            "iapetus": Voice(
                name="BreathyIapetus",
                id="iapetus",
                gender="MALE",
                description="BreathyIapetus"
            ),
            "kore": Voice(
                name="InformativeKore",
                id="kore",
                gender="FEMALE",
                description="InformativeKore"
            ),
            "laomedeia": Voice(
                name="InformativeLaomedeia",
                id="laomedeia",
                gender="FEMALE",
                description="InformativeLaomedeia"
            ),
            "leda": Voice(
                name="ExcitableLeda",
                id="leda",
                gender="FEMALE",
                description="ExcitableLeda"
            ),
            "orus": Voice(
                name="YouthfulOrus",
                id="orus",
                gender="MALE",
                description="YouthfulOrus"
            ),
            "puck": Voice(
                name="BrightPuck",
                id="puck",
                gender="MALE",
                description="BrightPuck"
            ),
            "pulcherrima": Voice(
                name="MaturePulcherrima",
                id="pulcherrima",
                gender="NEUTRAL",
                description="MaturePulcherrima"
            ),
            "rasalgethi": Voice(
                name="GravellyRasalgethi",
                id="rasalgethi",
                gender="MALE",
                description="GravellyRasalgethi"
            ),
            "sadachbia": Voice(
                name="GentleSadachbia",
                id="sadachbia",
                gender="MALE",
                description="GentleSadachbia"
            ),
            "sadaltager": Voice(
                name="LivelySadaltager",
                id="sadaltager",
                gender="NEUTRAL",
                description="LivelySadaltager"
            ),
            "schedar": Voice(
                name="FirmSchedar",
                id="schedar",
                gender="MALE",
                description="FirmSchedar"
            ),
            "sulafat": Voice(
                name="KnowledgeableSulafat",
                id="sulafat",
                gender="FEMALE",
                description="KnowledgeableSulafat"
            ),
            "umbriel": Voice(
                name="ClearUmbriel",
                id="umbriel",
                gender="MALE",
                description="ClearUmbriel"
            ),
            "vindemiatrix": Voice(
                name="CasualVindemiatrix",
                id="vindemiatrix",
                gender="FEMALE",
                description="CasualVindemiatrix"
            ),
            "zephyr": Voice(
                name="Zephyr",
                id="zephyr",
                gender="FEMALE",
                description="Zephyr"
            ),
            "zubenelgenubi": Voice(
                name="FriendlyZubenelgenubi",
                id="zubenelgenubi",
                gender="MALE",
                description="FriendlyZubenelgenubi"
            ),
        }
        return voices

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        return [
            Model(
                id="gemini-2.5-flash-preview-tts",
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
            voice: Voice ID to use
            output_file: Optional path to save the audio file
            **kwargs: Additional arguments passed to the provider

        Returns:
            AudioResponse object containing the audio data and metadata
        """
        model_name = self.model_name or self._get_default_model()
        
        # Prepare request payload
        payload = {
            "contents": [{
                "parts": [{
                    "text": text
                }]
            }],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": voice
                        }
                    }
                }
            },
            "model": model_name,
        }

        # Make HTTP request
        response = self.client.post(
            f"{self.base_url}/models/{model_name}:generateContent?key={self.api_key}",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)
        
        response_data = response.json()
        
        # Extract audio data from response
        audio_data_b64 = response_data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        pcm_data = base64.b64decode(audio_data_b64)
        
        # Convert PCM to WAV format
        audio_data = self._convert_pcm_to_wav(pcm_data)

        response_audio = AudioResponse(
            audio_data=audio_data,
            content_type="audio/wav",  # Converted to WAV format
            model=model_name,
            voice=voice,
            provider="google"
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
            voice: Voice ID to use
            output_file: Optional path to save the audio file
            **kwargs: Additional arguments passed to the provider

        Returns:
            AudioResponse object containing the audio data and metadata
        """
        model_name = self.model_name or self._get_default_model()
        
        # Prepare request payload
        payload = {
            "contents": [{
                "parts": [{
                    "text": text
                }]
            }],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": voice
                        }
                    }
                }
            },
            "model": model_name,
        }

        # Make async HTTP request
        response = await self.async_client.post(
            f"{self.base_url}/models/{model_name}:generateContent?key={self.api_key}",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)
        
        response_data = response.json()
        
        # Extract audio data from response
        audio_data_b64 = response_data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        pcm_data = base64.b64decode(audio_data_b64)
        
        # Convert PCM to WAV format
        audio_data = self._convert_pcm_to_wav(pcm_data)

        response_audio = AudioResponse(
            audio_data=audio_data,
            content_type="audio/wav",  # Converted to WAV format
            model=model_name,
            voice=voice,
            provider="google"
        )

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_data)

        return response_audio

    def generate_multi_speaker_speech(
        self,
        text: str,
        speaker_configs: List[Dict[str, str]],
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech with multiple speakers (Google-specific feature).

        Args:
            text: Text containing the conversation with speaker names
            speaker_configs: List of dicts with 'speaker' and 'voice' keys
            output_file: Optional path to save the audio file
            **kwargs: Additional arguments passed to the provider

        Returns:
            AudioResponse object containing the audio data and metadata
            
        Example:
            speaker_configs = [
                {"speaker": "Joe", "voice": "Kore"},
                {"speaker": "Jane", "voice": "Puck"}
            ]
        """
        model_name = self.model_name or self._get_default_model()
        
        # Build speaker voice configs
        speaker_voice_configs = []
        for config in speaker_configs:
            speaker_voice_configs.append({
                "speaker": config["speaker"],
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": config["voice"]
                    }
                }
            })
        
        # Prepare request payload for multi-speaker
        payload = {
            "contents": [{
                "parts": [{
                    "text": text
                }]
            }],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "multiSpeakerVoiceConfig": {
                        "speakerVoiceConfigs": speaker_voice_configs
                    }
                }
            },
            "model": model_name,
        }

        # Make HTTP request
        response = self.client.post(
            f"{self.base_url}/models/{model_name}:generateContent?key={self.api_key}",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)
        
        response_data = response.json()
        
        # Extract audio data from response
        audio_data_b64 = response_data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        pcm_data = base64.b64decode(audio_data_b64)
        
        # Convert PCM to WAV format
        audio_data = self._convert_pcm_to_wav(pcm_data)

        response_audio = AudioResponse(
            audio_data=audio_data,
            content_type="audio/wav",  # Converted to WAV format
            model=model_name,
            voice="multi-speaker",
            provider="google"
        )

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_data)

        return response_audio
    
    async def agenerate_multi_speaker_speech(
        self,
        text: str,
        speaker_configs: List[Dict[str, str]],
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech with multiple speakers asynchronously (Google-specific feature).

        Args:
            text: Text containing the conversation with speaker names
            speaker_configs: List of dicts with 'speaker' and 'voice' keys
            output_file: Optional path to save the audio file
            **kwargs: Additional arguments passed to the provider

        Returns:
            AudioResponse object containing the audio data and metadata
        """
        model_name = self.model_name or self._get_default_model()
        
        # Build speaker voice configs
        speaker_voice_configs = []
        for config in speaker_configs:
            speaker_voice_configs.append({
                "speaker": config["speaker"],
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": config["voice"]
                    }
                }
            })
        
        # Prepare request payload for multi-speaker
        payload = {
            "contents": [{
                "parts": [{
                    "text": text
                }]
            }],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "multiSpeakerVoiceConfig": {
                        "speakerVoiceConfigs": speaker_voice_configs
                    }
                }
            },
            "model": model_name,
        }

        # Make async HTTP request
        response = await self.async_client.post(
            f"{self.base_url}/models/{model_name}:generateContent?key={self.api_key}",
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)
        
        response_data = response.json()
        
        # Extract audio data from response
        audio_data_b64 = response_data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        pcm_data = base64.b64decode(audio_data_b64)
        
        # Convert PCM to WAV format
        audio_data = self._convert_pcm_to_wav(pcm_data)

        response_audio = AudioResponse(
            audio_data=audio_data,
            content_type="audio/wav",  # Converted to WAV format
            model=model_name,
            voice="multi-speaker",
            provider="google"
        )

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_data)

        return response_audio
