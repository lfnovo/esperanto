"""Base types for the esperanto package."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from esperanto.base.model import BaseModel


@dataclass
class Message:
    """A message in a conversation."""
    role: str
    content: str


@dataclass
class ChatCompletion:
    """A chat completion response."""
    messages: List[Message]
    model: str
    usage: Dict[str, int]
    finish_reason: Optional[str] = None


@dataclass
class Embedding:
    """An embedding vector."""
    vector: List[float]
    text: str
    model: str


@dataclass
class AudioTranscription:
    """A transcription of audio to text."""
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class AudioSynthesis:
    """A synthesis of text to audio."""
    audio: bytes
    format: str
    model: str


class LanguageModel(BaseModel, ABC):
    """Base class for language models."""

    @abstractmethod
    def to_langchain(self) -> BaseChatModel:
        """Convert to a LangChain chat model."""
        pass


class EmbeddingModel(BaseModel, ABC):
    """Base class for embedding models."""

    @abstractmethod
    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Embed texts into vectors."""
        pass


class SpeechToTextModel(BaseModel, ABC):
    """Base class for speech-to-text models."""

    @abstractmethod
    async def transcribe(self, audio_file: str, **kwargs) -> str:
        """Transcribe audio to text."""
        pass


class TextToSpeechModel(BaseModel, ABC):
    """Base class for text-to-speech models."""

    @abstractmethod
    def synthesize(
        self, text: str, voice: Optional[str] = None, **kwargs
    ) -> bytes:
        """Synthesize text to speech."""
        pass
