"""Response types for Esperanto."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Usage:
    """Usage statistics for a completion."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class Message:
    """A message in a chat completion."""

    content: Optional[str] = None
    role: Optional[str] = None


@dataclass
class ChatCompletionMessage(Message):
    """A message in a chat completion."""

    pass


@dataclass
class DeltaMessage(Message):
    """A delta message in a streaming chat completion."""

    pass


@dataclass
class Choice:
    """A single choice in a chat completion."""

    index: int
    message: Message
    finish_reason: Optional[str] = None


@dataclass
class ChatCompletionChoice(Choice):
    """A single choice in a chat completion."""

    message: ChatCompletionMessage


@dataclass
class StreamChoice:
    """A single choice in a streaming chat completion."""

    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


@dataclass
class ChatCompletion:
    """A chat completion response."""

    id: str
    choices: List[ChatCompletionChoice]
    model: str
    provider: str
    created: Optional[int] = None
    usage: Optional[Usage] = None
    object: str = "chat.completion"

    @property
    def content(self) -> str:
        """Get the content of the first choice's message."""
        if not self.choices or not self.choices[0].message:
            return ""
        return self.choices[0].message.content or ""


@dataclass
class ChatCompletionChunk:
    """A chunk of a streaming chat completion."""

    id: str
    choices: List[StreamChoice]
    model: str
    created: int
    object: str = "chat.completion.chunk"
