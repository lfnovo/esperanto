"""Speech-to-text model implementations."""

from esperanto.providers.speech_to_text.groq import GroqSpeechToTextModel
from esperanto.providers.speech_to_text.openai import OpenAISpeechToTextModel

__all__ = [
    "OpenAISpeechToTextModel",
    "GroqSpeechToTextModel",
]
