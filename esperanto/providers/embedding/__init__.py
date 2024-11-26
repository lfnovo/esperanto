"""Embedding providers module."""
from esperanto.providers.embedding.base import EmbeddingModel
from esperanto.providers.embedding.gemini import GeminiEmbeddingModel
from esperanto.providers.embedding.ollama import OllamaEmbeddingModel
from esperanto.providers.embedding.openai import OpenAIEmbeddingModel
from esperanto.providers.embedding.vertex import VertexEmbeddingModel

__all__ = [
    "OpenAIEmbeddingModel",
    "OllamaEmbeddingModel",
    "GeminiEmbeddingModel",
    "VertexEmbeddingModel",
    "EmbeddingModel"
]
