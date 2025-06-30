"""Model type definitions for Esperanto."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Model(BaseModel):
    """Model information from providers."""

    id: str = Field(description="The unique identifier of the model")
    owned_by: str = Field(description="The organization that owns the model")
    context_window: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens the model can process in a single request",
        ge=0,  # context window must be non-negative
    )
    type: Literal["language", "embedding", "text_to_speech", "speech_to_text", "reranker"] = Field(
        default="language",
        description="The type of model (language model, embedding model, reranker, etc.)",
    )

    model_config = ConfigDict(frozen=True)  # Make models immutable
