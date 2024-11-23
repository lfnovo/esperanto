"""
Esperanto - A unified interface for various AI model providers
"""

__version__ = "0.1.0"

from .base import (
    AudioSynthesis,
    AudioTranscription,
    BaseModel,
    ChatCompletion,
    Embedding,
    EmbeddingModel,
    LanguageModel,
    Message,
    SpeechToTextModel,
    TextToSpeechModel,
)
from .providers.embedding import (
    GeminiEmbeddingModel,
    OllamaEmbeddingModel,
    OpenAIEmbeddingModel,
    VertexEmbeddingModel,
)
from .providers.llm import (
    AnthropicLanguageModel,
    GeminiLanguageModel,
    GroqLanguageModel,
    LiteLLMLanguageModel,
    OllamaLanguageModel,
    OpenAILanguageModel,
    OpenRouterLanguageModel,
    VertexAILanguageModel,
    VertexAnthropicLanguageModel,
    XAILanguageModel,
)
from .providers.speech_to_text import (
    GroqSpeechToTextModel,
    OpenAISpeechToTextModel,
)
from .providers.text_to_speech import (
    ElevenLabsTextToSpeechModel,
    GeminiTextToSpeechModel,
    OpenAITextToSpeechModel,
)

__all__ = [
    # Base classes
    "BaseModel",
    "LanguageModel",
    "EmbeddingModel",
    "SpeechToTextModel",
    "TextToSpeechModel",
    # Data classes
    "Message",
    "ChatCompletion",
    "Embedding",
    "AudioTranscription",
    "AudioSynthesis",
    # Embedding models
    "GeminiEmbeddingModel",
    "OllamaEmbeddingModel",
    "OpenAIEmbeddingModel",
    "VertexEmbeddingModel",
    # Language models
    "AnthropicLanguageModel",
    "GeminiLanguageModel",
    "GroqLanguageModel",
    "LiteLLMLanguageModel",
    "OllamaLanguageModel",
    "OpenAILanguageModel",
    "OpenRouterLanguageModel",
    "VertexAILanguageModel",
    "VertexAnthropicLanguageModel",
    "XAILanguageModel",
    # Speech-to-text models
    "OpenAISpeechToTextModel",
    "GroqSpeechToTextModel",
    # Text-to-speech models
    "ElevenLabsTextToSpeechModel",
    "GeminiTextToSpeechModel",
    "OpenAITextToSpeechModel",
]
