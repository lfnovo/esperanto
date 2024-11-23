"""Google Cloud text-to-speech model implementation."""

from typing import Any, Dict, Optional

from google.cloud import texttospeech_v1

from esperanto.base.types import TextToSpeechModel


class GeminiTextToSpeechModel(TextToSpeechModel):
    """Google Cloud text-to-speech model implementation."""

    def __init__(
        self,
        model_name: str = "en-US-Standard-A",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_name, config)
        self._client = None
        self.model = model_name
        self.language_code = self.config.get("language_code", "en-US")
        self.voice_name = self.config.get("voice_name", "en-US-Standard-A")
        self.speaking_rate = self.config.get("speaking_rate", 1.0)
        self.pitch = self.config.get("pitch", 0.0)
        self.volume_gain_db = self.config.get("volume_gain_db", 0.0)
        self.audio_encoding = self.config.get("audio_encoding", "MP3")
        self.sample_rate_hertz = self.config.get("sample_rate_hertz", 16000)

    @property
    def client(self) -> texttospeech_v1.TextToSpeechAsyncClient:
        """Get the Google Cloud Text-to-Speech client."""
        if self._client is None:
            self._client = texttospeech_v1.TextToSpeechAsyncClient()
        return self._client

    @client.setter
    def client(self, value: texttospeech_v1.TextToSpeechAsyncClient) -> None:
        """Set the Google Cloud Text-to-Speech client."""
        self._client = value

    @client.deleter
    def client(self) -> None:
        """Delete the Google Cloud Text-to-Speech client."""
        self._client = None

    @property
    def provider(self) -> str:
        """Get provider name."""
        return "gemini"

    def validate_config(self) -> None:
        """Validate Google Cloud Text-to-Speech configuration."""
        if not self.model:
            raise ValueError("model must be specified for Google Cloud text-to-speech model")
        if not self.language_code:
            raise ValueError("language_code must be specified")
        if not self.voice_name:
            raise ValueError("voice_name must be specified")
        if not 0.25 <= self.speaking_rate <= 4.0:
            raise ValueError("speaking_rate must be between 0.25 and 4.0")
        if not -20.0 <= self.pitch <= 20.0:
            raise ValueError("pitch must be between -20.0 and 20.0")
        if not -96.0 <= self.volume_gain_db <= 16.0:
            raise ValueError("volume_gain_db must be between -96.0 and 16.0")
        if not 8000 <= self.sample_rate_hertz <= 48000:
            raise ValueError("sample_rate_hertz must be between 8000 and 48000")

    async def synthesize(self, text: str, output_file: str, **kwargs: Any) -> str:
        """Synthesize text to speech and save to file.
        
        Args:
            text: Text to synthesize
            output_file: Path to save the audio file
            **kwargs: Additional arguments for synthesis
            
        Returns:
            Path to the output audio file
        """
        self.validate_config()

        # Create input
        synthesis_input = texttospeech_v1.SynthesisInput(text=text)

        # Create voice config
        voice = texttospeech_v1.VoiceSelectionParams(
            language_code=self.language_code,
            name=self.voice_name,
        )

        # Create audio config
        audio_config = texttospeech_v1.AudioConfig(
            audio_encoding=getattr(texttospeech_v1.AudioEncoding, self.audio_encoding),
            speaking_rate=kwargs.get("speaking_rate", self.speaking_rate),
            pitch=kwargs.get("pitch", self.pitch),
            volume_gain_db=kwargs.get("volume_gain_db", self.volume_gain_db),
            sample_rate_hertz=kwargs.get("sample_rate_hertz", self.sample_rate_hertz),
        )

        # Generate audio
        response = await self.client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )

        # Save to file
        with open(output_file, "wb") as f:
            f.write(response.audio_content)

        return output_file
