"""OpenAI Text-to-Speech model implementation."""

import os
from typing import Any, ClassVar, Dict, Optional

import openai
from pydantic import SecretStr

from esperanto.base.types import TextToSpeechModel


class OpenAITextToSpeechModel(TextToSpeechModel):
    """OpenAI Text-to-Speech model implementation."""

    DEFAULT_MODEL: ClassVar[str] = "tts-1"
    VALID_MODELS: ClassVar[tuple] = ("tts-1", "tts-1-hd")
    VALID_VOICES: ClassVar[tuple] = ("alloy", "echo", "fable", "onyx", "nova", "shimmer")
    DEFAULT_VOICE: ClassVar[str] = "alloy"
    VALID_FORMATS: ClassVar[tuple] = ("mp3", "opus", "aac", "flac")
    DEFAULT_FORMAT: ClassVar[str] = "mp3"

    def __init__(
        self,
        model_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize OpenAI Text-to-Speech model.
        
        Args:
            model_name: Model name (default: tts-1)
            config: Configuration dictionary. Can include:
                - api_key: OpenAI API key
                - model: Model name (default: tts-1)
                - voice: Voice ID (default: alloy)
                - response_format: Audio format (default: mp3)
                - speed: Speech speed (default: 1.0)
        """
        config = config or {}
        if model_name:
            config["model"] = model_name
        super().__init__(config)
        self._client = None
        
        # API configuration
        api_key = self.config.get("api_key")
        if not api_key and "OPENAI_API_KEY" in os.environ:
            api_key = os.environ["OPENAI_API_KEY"]
        self._api_key = SecretStr(api_key) if api_key else None
        
        # Model configuration
        self._model = model_name or config.get("model") or None
        self._voice = config.get("voice", self.DEFAULT_VOICE)
        self._response_format = config.get("response_format", self.DEFAULT_FORMAT)
        self._speed = float(config.get("speed", 1.0))

    @property
    def client(self) -> openai.Client:
        """Get OpenAI client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("OpenAI API key must be provided in config or OPENAI_API_KEY environment variable")
            self._client = openai.Client(api_key=self.api_key.get_secret_value())
        return self._client

    @client.setter
    def client(self, value: openai.Client) -> None:
        """Set OpenAI client."""
        self._client = value

    @property
    def api_key(self) -> Optional[SecretStr]:
        """Get API key."""
        if self._api_key:
            return self._api_key
        if openai.api_key:
            return SecretStr(openai.api_key)
        if "OPENAI_API_KEY" in os.environ:
            return SecretStr(os.environ["OPENAI_API_KEY"])
        return None

    @property
    def model(self) -> str:
        """Get model name."""
        return self._model

    @property
    def voice(self) -> str:
        """Get voice ID."""
        return self._voice

    @property
    def response_format(self) -> str:
        """Get response format."""
        return self._response_format

    @property
    def speed(self) -> float:
        """Get speech speed."""
        return self._speed

    @property
    def provider(self) -> str:
        """Get provider name."""
        return "openai"

    def validate_config(self) -> None:
        """Validate configuration."""
        # Check API key
        if not self.api_key:
            raise ValueError("api_key must be specified")
        
        # Check model name
        if not self._model:
            raise ValueError("model_name must be specified")
        if self._model not in self.VALID_MODELS:
            raise ValueError(f"Invalid model. Must be one of: {', '.join(self.VALID_MODELS)}")
        
        # Check voice
        if not self._voice:
            raise ValueError("voice must be specified")
        if self._voice not in self.VALID_VOICES:
            raise ValueError(f"Invalid voice. Must be one of: {', '.join(self.VALID_VOICES)}")
        
        # Check response format
        if not self._response_format:
            raise ValueError("response_format must be specified")
        if self._response_format not in self.VALID_FORMATS:
            raise ValueError(f"Invalid response_format. Must be one of: {', '.join(self.VALID_FORMATS)}")
        
        # Check speed
        if not isinstance(self._speed, (int, float)):
            raise ValueError("speed must be a number")
        if not 0.25 <= self._speed <= 4.0:
            raise ValueError("Speed must be between 0.25 and 4.0")

    async def synthesize(self, text: str, output_file: str, **kwargs: Any) -> str:
        """Synthesize text to speech using OpenAI's API.
        
        Args:
            text: Text to synthesize
            output_file: Path to save the audio file
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            Path to the output audio file
            
        Raises:
            ValueError: If configuration is invalid or synthesis fails
        """
        try:
            self.validate_config()

            # Extract known parameters from kwargs
            voice = kwargs.pop("voice", self.voice)
            speed = kwargs.pop("speed", self.speed)

            # Create synthesis request
            response = self.client.audio.speech.create(  # OpenAI's create is synchronous
                model=self.model,
                voice=voice,
                input=text,
                response_format=self.response_format,
                speed=speed,
                **kwargs,  # Pass remaining kwargs
            )

            # Save to file
            with open(output_file, "wb") as f:
                for chunk in response.iter_bytes():  # Handle streaming response
                    f.write(chunk)

            return output_file

        except Exception as e:
            raise ValueError(f"Failed to synthesize speech: {str(e)}") from e
