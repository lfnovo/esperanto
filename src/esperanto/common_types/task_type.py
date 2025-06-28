"""Task type enum for embedding models."""

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
    
    # Default/Generic
    DEFAULT = "default"                          # No specific optimization