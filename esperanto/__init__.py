"""
Esperanto: A unified interface for language models.
This module exports all public components of the library.
"""

from typing import Dict, List, Optional, Type

from esperanto.providers.embedding.base import EmbeddingModel
from esperanto.providers.llm.base import LanguageModel

# Import providers conditionally to handle optional dependencies
try:
    from esperanto.providers.llm.anthropic import AnthropicLanguageModel
except ImportError:
    AnthropicLanguageModel = None

try:
    from esperanto.providers.llm.gemini import GeminiLanguageModel
except ImportError:
    GeminiLanguageModel = None

try:
    from esperanto.providers.llm.openai import OpenAILanguageModel
except ImportError:
    OpenAILanguageModel = None

try:
    from esperanto.providers.llm.openrouter import OpenRouterLanguageModel
except ImportError:
    OpenRouterLanguageModel = None

try:
    from esperanto.providers.llm.mistral import MistralLanguageModel
except ImportError:
    MistralLanguageModel = None

try:
    from esperanto.providers.llm.xai import XAILanguageModel
except ImportError:
    XAILanguageModel = None

try:
    from esperanto.providers.embedding.openai import OpenAIEmbeddingModel
except ImportError:
    OpenAIEmbeddingModel = None

try:
    from esperanto.providers.embedding.gemini import GeminiEmbeddingModel
except ImportError:
    GeminiEmbeddingModel = None

try:
    from esperanto.providers.embedding.xai import XAIEmbeddingModel
except ImportError:
    XAIEmbeddingModel = None

# Store all provider classes
__provider_classes = {
    'AnthropicLanguageModel': AnthropicLanguageModel,
    'GeminiLanguageModel': GeminiLanguageModel,
    'OpenAILanguageModel': OpenAILanguageModel,
    'OpenRouterLanguageModel': OpenRouterLanguageModel,
    'MistralLanguageModel': MistralLanguageModel,
    'XAILanguageModel': XAILanguageModel,
    'OpenAIEmbeddingModel': OpenAIEmbeddingModel,
    'GeminiEmbeddingModel': GeminiEmbeddingModel,
    'XAIEmbeddingModel': XAIEmbeddingModel,
}

# Get list of available provider classes (excluding None values)
provider_classes = [name for name, cls in __provider_classes.items() if cls is not None]

# Import factory after defining providers
from esperanto.factory import AIFactory

__all__ = ["AIFactory", "LanguageModel", "EmbeddingModel"] + provider_classes

# Make provider classes available at module level
globals().update({k: v for k, v in __provider_classes.items() if v is not None})