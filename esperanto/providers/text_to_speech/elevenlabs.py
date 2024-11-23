"""ElevenLabs text-to-speech provider."""

from typing import Any, Optional

from elevenlabs import AsyncElevenLabs
from pydantic import SecretStr

from esperanto.base import TextToSpeechModel


class ElevenLabsTextToSpeechModel(TextToSpeechModel):
    """ElevenLabs text-to-speech model."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize ElevenLabs text-to-speech model.

        Args:
            model_name: Model name
            config: Configuration dictionary
        """
        super().__init__(model_name=model_name, config=config)
        self._client: Optional[AsyncElevenLabs] = None
        self._model = self.model_name  # Set _model from model_name
        self._api_key = None  # Initialize _api_key as None
        self._voice = None  # Initialize _voice as None
        
        # Set API key from config if provided
        if config and "api_key" in config:
            self._api_key = SecretStr(config["api_key"])
        else:
            # Try to get from environment variable
            import os
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if api_key:
                self._api_key = SecretStr(api_key)
                
        # Set voice from config if provided
        if config and "voice" in config:
            self._voice = config["voice"]
        else:
            # Set default voice if not provided
            self._voice = "Adam"  # Using a default voice from ElevenLabs

    @property
    def model(self) -> str:
        """Get model name.

        Returns:
            Model name
        """
        return self._model

    @property
    def provider(self) -> str:
        """Get provider name.

        Returns:
            Provider name
        """
        return "elevenlabs"

    def validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate model name
        if not self._model:
            raise ValueError("model_name must be specified")

        # Validate API key
        if not self._api_key:
            raise ValueError("api_key must be specified")

        # Validate voice
        if not self._voice:
            raise ValueError("voice must be specified")

        # Validate stability (if provided)
        stability = self.config.get("stability")
        if stability is not None:
            if (
                not isinstance(stability, (int, float))
                or stability < 0
                or stability > 1
            ):
                raise ValueError("Stability must be between 0 and 1")

        # Validate similarity boost (if provided)
        similarity_boost = self.config.get("similarity_boost")
        if similarity_boost is not None:
            if (
                not isinstance(similarity_boost, (int, float))
                or similarity_boost < 0
                or similarity_boost > 1
            ):
                raise ValueError("Similarity boost must be between 0 and 1")

        # Validate style (if provided)
        style = self.config.get("style")
        if style is not None:
            if not isinstance(style, (int, float)) or style < 0 or style > 1:
                raise ValueError("Style must be between 0 and 1")

    @property
    def client(self) -> AsyncElevenLabs:
        """Get ElevenLabs client.

        Returns:
            ElevenLabs client

        Raises:
            ValueError: If API key is not set
        """
        if self._client is None:
            if not self._api_key:
                raise ValueError("API key not set")
            self._client = AsyncElevenLabs(api_key=self._api_key.get_secret_value())
        return self._client

    @client.setter
    def client(self, client: AsyncElevenLabs) -> None:
        """Set ElevenLabs client.

        Args:
            client: ElevenLabs client
        """
        self._client = client

    @client.deleter
    def client(self) -> None:
        """Delete ElevenLabs client."""
        self._client = None

    @property
    def voice(self) -> str:
        """Get voice.

        Returns:
            Voice
        """
        return self._voice

    async def synthesize(self, text: str, output_file: str, **kwargs: Any) -> str:
        """Synthesize text to speech using ElevenLabs API.

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

            # Get optional parameters from config or kwargs
            stability = kwargs.pop("stability", self.config.get("stability"))
            similarity_boost = kwargs.pop(
                "similarity_boost", self.config.get("similarity_boost")
            )
            style = kwargs.pop("style", self.config.get("style"))

            # Create voice settings dictionary
            voice_settings = {}
            if stability is not None:
                voice_settings["stability"] = stability
            if similarity_boost is not None:
                voice_settings["similarity_boost"] = similarity_boost
            if style is not None:
                voice_settings["style"] = style

            # Create synthesis request
            audio_stream = await self.client.generate(
                text=text,
                voice=self.voice,
                model=self.model,
                voice_settings=voice_settings if voice_settings else None,
                **kwargs,  # Pass remaining kwargs
            )

            # Save to file
            with open(output_file, "wb") as f:
                async for chunk in audio_stream:
                    f.write(chunk)

            return output_file

        except Exception as e:
            raise ValueError(f"Failed to synthesize speech: {str(e)}") from e
