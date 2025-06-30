"""Reranker types for Esperanto."""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RerankResult(BaseModel):
    """Individual reranking result for a document."""

    index: int = Field(description="Original document index", ge=0)
    document: str = Field(description="Original document text")
    relevance_score: float = Field(
        description="Normalized 0-1 relevance score", ge=0.0, le=1.0
    )

    model_config = ConfigDict(frozen=True)


class RerankResponse(BaseModel):
    """Standardized reranking response across all providers."""

    results: List[RerankResult] = Field(
        description="Ranked results (highest score first)"
    )
    model: str = Field(description="Model used for reranking")
    usage: Optional["Usage"] = Field(
        default=None, description="Token/request usage if available"
    )

    @property
    def top_result(self) -> Optional[RerankResult]:
        """Get the highest ranked result."""
        return self.results[0] if self.results else None

    def get_top_k(self, k: int) -> List[RerankResult]:
        """Get top k results."""
        return self.results[:k]

    model_config = ConfigDict(frozen=True)


# Import Usage after defining our classes to avoid circular imports
from .response import Usage

# Update forward reference
RerankResponse.model_rebuild()