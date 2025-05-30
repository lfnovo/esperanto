"""
Esperanto: A unified interface for language models.
This module exports all public components of the library.
"""

from esperanto.providers.embedding.base import EmbeddingModel
from esperanto.providers.llm.base import LanguageModel
from esperanto.providers.stt.base import SpeechToTextModel
from esperanto.providers.tts.base import TextToSpeechModel

# Import providers conditionally to handle optional dependencies
try:
    from esperanto.providers.llm.anthropic import AnthropicLanguageModel
except ImportError:
    AnthropicLanguageModel = None

try:
    from esperanto.providers.llm.google import GoogleLanguageModel
except ImportError:
    GoogleLanguageModel = None

try:
    from esperanto.providers.llm.ollama import OllamaLanguageModel
except ImportError:
    OllamaLanguageModel = None


try:
    from esperanto.providers.llm.openai import OpenAILanguageModel
except ImportError:
    OpenAILanguageModel = None

try:
    from esperanto.providers.llm.openrouter import OpenRouterLanguageModel
except ImportError:
    OpenRouterLanguageModel = None

try:
    from esperanto.providers.llm.xai import XAILanguageModel
except ImportError:
    XAILanguageModel = None

try:
    from esperanto.providers.embedding.openai import OpenAIEmbeddingModel
except ImportError:
    OpenAIEmbeddingModel = None

try:
    from esperanto.providers.embedding.google import GoogleEmbeddingModel
except ImportError:
    GoogleEmbeddingModel = None

try:
    from esperanto.providers.embedding.ollama import OllamaEmbeddingModel
except ImportError:
    OllamaEmbeddingModel = None

try:
    from esperanto.providers.llm.azure import AzureLanguageModel
except ImportError:
    AzureLanguageModel = None

# Store all provider classes
__provider_classes = {
    'AnthropicLanguageModel': AnthropicLanguageModel,
    'GoogleLanguageModel': GoogleLanguageModel,
    'OpenAILanguageModel': OpenAILanguageModel,
    'OpenRouterLanguageModel': OpenRouterLanguageModel,
    'XAILanguageModel': XAILanguageModel,
    'OpenAIEmbeddingModel': OpenAIEmbeddingModel,
    'GoogleEmbeddingModel': GoogleEmbeddingModel,
    "OllamaEmbeddingModel": OllamaEmbeddingModel,
    "OllamaLanguageModel": OllamaLanguageModel,
    "AzureLanguageModel": AzureLanguageModel
}

# Get list of available provider classes (excluding None values)
provider_classes = [name for name, cls in __provider_classes.items() if cls is not None]

# Import factory after defining providers
from esperanto.factory import AIFactory

__all__ = ["AIFactory", "LanguageModel", "EmbeddingModel", "SpeechToTextModel", "TextToSpeechModel"] + provider_classes

# Make provider classes available at module level
globals().update({k: v for k, v in __provider_classes.items() if v is not None})