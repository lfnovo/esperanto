"""Model type definitions for Esperanto."""
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class Model:
    """Model information from providers."""
    id: str
    owned_by: str
    context_window: Optional[int] = None
    type: Literal["language", "embedding", "text_to_speech", "speech_to_text"] = "language"
