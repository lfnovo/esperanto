"""OpenRouter Text-to-Speech provider implementation."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from .base import AudioResponse, Model, Voice
from .openai import OpenAITextToSpeechModel


class OpenRouterTextToSpeechModel(OpenAITextToSpeechModel):
    """OpenRouter Text-to-Speech provider.

    OpenRouter exposes an OpenAI-compatible audio speech endpoint
    (``POST /api/v1/audio/speech``) that accepts ``model``, ``input``, ``voice``
    and ``response_format`` (``mp3`` or ``pcm`` only). This provider reuses the
    OpenAI TTS request logic and overrides credential resolution, the required
    OpenRouter HTTP headers, provider identity, and the model-specific defaults.

    Model names follow OpenRouter's ``vendor/model`` convention. Note that
    voices are model-specific — the default ``microsoft/mai-voice-2`` uses
    Microsoft neural voice names (e.g. ``en-US-AvaNeural``), not OpenAI's
    ``alloy``/``nova`` set. When selecting a different model, pass a voice that
    the model supports (see its page on https://openrouter.ai/models).
    """

    DEFAULT_MODEL = "microsoft/mai-voice-2"
    DEFAULT_VOICE = "en-US-AvaNeural"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    PROVIDER = "openrouter"

    # Verified working Microsoft neural voices for the default model.
    _DEFAULT_MODEL_VOICES = {
        "en-US-AvaNeural": ("FEMALE", "en-US"),
        "en-US-EmmaNeural": ("FEMALE", "en-US"),
        "en-US-AriaNeural": ("FEMALE", "en-US"),
        "en-US-JennyNeural": ("FEMALE", "en-US"),
        "en-US-AndrewNeural": ("MALE", "en-US"),
        "en-US-BrianNeural": ("MALE", "en-US"),
        "en-GB-SoniaNeural": ("FEMALE", "en-GB"),
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the OpenRouter TTS provider.

        Args:
            model_name: OpenRouter TTS model id (default: ``microsoft/mai-voice-2``).
            api_key: OpenRouter API key. Falls back to ``OPENROUTER_API_KEY``.
            base_url: API base URL. Falls back to ``OPENROUTER_BASE_URL`` then the
                OpenRouter default (``https://openrouter.ai/api/v1``).
            **kwargs: Additional configuration options.
        """
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key not found. Set the OPENROUTER_API_KEY environment variable."
            )

        base_url = base_url or os.getenv("OPENROUTER_BASE_URL") or self.DEFAULT_BASE_URL

        super().__init__(
            model_name=model_name or self.DEFAULT_MODEL,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers, adding OpenRouter's required attribution headers."""
        headers = super()._get_headers()
        headers.update(
            {
                "HTTP-Referer": "https://github.com/lfnovo/esperanto",
                "X-Title": "Esperanto",
            }
        )
        return headers

    def _handle_error(self, response) -> None:
        """Handle HTTP error responses with OpenRouter-specific messaging."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get(
                    "message", f"HTTP {response.status_code}"
                )
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"OpenRouter API error: {error_message}")

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return self.PROVIDER

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return self.DEFAULT_MODEL

    def _get_models(self) -> List[Model]:
        """List OpenRouter models that output speech (dedicated TTS models).

        Uses OpenRouter's ``output_modalities=speech`` filter, which surfaces the
        dedicated text-to-speech models (these do not appear in the unfiltered
        ``/models`` listing). Returns an empty list if the endpoint is
        unreachable.
        """
        try:
            response = self.client.get(
                f"{self.base_url}/models",
                headers=self._get_headers(),
                params={"output_modalities": "speech"},
            )
            self._handle_error(response)
            return [
                Model(
                    id=model["id"],
                    owned_by=model["id"].split("/")[0] if "/" in model["id"] else "OpenRouter",
                    context_window=None,  # Audio models don't have context windows
                )
                for model in response.json().get("data", [])
            ]
        except Exception:
            return []

    @property
    def available_voices(self) -> Dict[str, Voice]:
        """Return the voice catalog for the default model.

        Voices on OpenRouter are model-specific; this catalog covers the
        default ``microsoft/mai-voice-2`` model. For other models, supply a
        voice listed on the model's page rather than relying on this catalog.
        """
        return {
            voice_id: Voice(
                name=voice_id,
                id=voice_id,
                gender=gender,
                language_code=language_code,
                description=f"Microsoft neural voice ({voice_id})",
            )
            for voice_id, (gender, language_code) in self._DEFAULT_MODEL_VOICES.items()
        }

    def generate_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> AudioResponse:
        """Generate speech, defaulting to a voice valid for the default model."""
        return super().generate_speech(
            text, voice or self.DEFAULT_VOICE, output_file, **kwargs
        )

    async def agenerate_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> AudioResponse:
        """Async generate speech, defaulting to a voice valid for the default model."""
        return await super().agenerate_speech(
            text, voice or self.DEFAULT_VOICE, output_file, **kwargs
        )
