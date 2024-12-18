"""Text-to-speech providers package."""

from .base import TextToSpeechModel

# Import base models directly since they don't have external dependencies
__all__ = ["TextToSpeechModel"]

# Add provider classes to __all__ but don't import them directly
# They will be imported dynamically by AIFactory when needed
__all__ += [
    "OpenAITextToSpeechModel",
    "ElevenLabsTextToSpeechModel",
    "GoogleTextToSpeechModel"
]
