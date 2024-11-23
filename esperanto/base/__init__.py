"""Base package for esperanto."""

from .model import BaseModel
from .types import (
    LanguageModel,
    EmbeddingModel,
    SpeechToTextModel,
    TextToSpeechModel,
    Message,
    ChatCompletion,
    Embedding,
    AudioTranscription,
    AudioSynthesis,
)