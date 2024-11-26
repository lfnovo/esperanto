"""Embedding providers module."""
from esperanto.providers.embedding.openai import OpenAIEmbeddingModel
from esperanto.providers.embedding.ollama import OllamaEmbeddingModel
from esperanto.providers.embedding.gemini import GeminiEmbeddingModel
from esperanto.providers.embedding.vertex import VertexEmbeddingModel
from esperanto.providers.embedding.base import EmbeddingModel

__all__ = [
    "OpenAIEmbeddingModel",
    "OllamaEmbeddingModel",
    "GeminiEmbeddingModel",
    "VertexEmbeddingModel",
    "EmbeddingModel"
]
