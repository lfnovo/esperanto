"""Task type enum for embedding models."""

import json
from enum import Enum


class EmbeddingTaskType(Enum):
    """Universal task types for embedding optimization.
    
    All embedding providers in Esperanto support these task types,
    either through native API support or emulation.
    """
    
    # Retrieval tasks
    RETRIEVAL_QUERY = "retrieval.query"          # Optimized for search queries
    RETRIEVAL_DOCUMENT = "retrieval.document"    # Optimized for document storage
    
    # Similarity tasks  
    SIMILARITY = "similarity"                     # General text similarity
    CLASSIFICATION = "classification"             # Text classification
    CLUSTERING = "clustering"                     # Document clustering
    
    # Code tasks
    CODE_RETRIEVAL = "code.retrieval"            # Code search optimization
    
    # Question answering and fact verification
    QUESTION_ANSWERING = "question_answering"    # Optimized for Q&A tasks
    FACT_VERIFICATION = "fact_verification"      # Optimized for fact checking
    
    # Default/Generic
    DEFAULT = "default"                          # No specific optimization
    
    def __str__(self) -> str:
        """Return string representation of the task type."""
        return self.value
    
    def __repr__(self) -> str:
        """Return detailed representation of the task type."""
        return f"EmbeddingTaskType.{self.name}"