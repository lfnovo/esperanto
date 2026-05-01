"""
Esperanto: A unified interface for language models.
This module exports all public components of the library.
"""

from esperanto.common_types import (
    FunctionCall,
    Tool,
    ToolCall,
    ToolCallValidationError,
    ToolFunction,
    find_tool_by_name,
    validate_tool_call,
    validate_tool_calls,
)
from esperanto.providers.embedding.base import EmbeddingModel
from esperanto.providers.llm.base import LanguageModel
from esperanto.providers.llm.profiles import OpenAICompatibleProfile
from esperanto.providers.stt.base import SpeechToTextModel
from esperanto.providers.tts.base import TextToSpeechModel

# Import providers conditionally to handle optional dependencies
try:
    from esperanto.providers.llm.anthropic import AnthropicLanguageModel
except ImportError:
    AnthropicLanguageModel = None  # type: ignore[assignment,misc]

try:
    from esperanto.providers.llm.google import GoogleLanguageModel
except ImportError:
    GoogleLanguageModel = None  # type: ignore[assignment,misc]

try:
    from esperanto.providers.llm.ollama import OllamaLanguageModel
except ImportError:
    OllamaLanguageModel = None  # type: ignore[assignment,misc]


try:
    from esperanto.providers.llm.openai import OpenAILanguageModel
except ImportError:
    OpenAILanguageModel = None  # type: ignore[assignment,misc]

try:
    from esperanto.providers.llm.openai_compatible import OpenAICompatibleLanguageModel
except ImportError:
    OpenAICompatibleLanguageModel = None  # type: ignore[assignment,misc]

try:
    from esperanto.providers.llm.openrouter import OpenRouterLanguageModel
except ImportError:
    OpenRouterLanguageModel = None  # type: ignore[assignment,misc]

try:
    from esperanto.providers.llm.xai import XAILanguageModel
except ImportError:
    XAILanguageModel = None  # type: ignore[assignment,misc]

try:
    from esperanto.providers.embedding.openai import OpenAIEmbeddingModel
except ImportError:
    OpenAIEmbeddingModel = None  # type: ignore[assignment,misc]

try:
    from esperanto.providers.embedding.google import GoogleEmbeddingModel
except ImportError:
    GoogleEmbeddingModel = None  # type: ignore[assignment,misc]

try:
    from esperanto.providers.embedding.ollama import OllamaEmbeddingModel
except ImportError:
    OllamaEmbeddingModel = None  # type: ignore[assignment,misc]

try:
    from esperanto.providers.embedding.azure import AzureEmbeddingModel
except ImportError:
    AzureEmbeddingModel = None  # type: ignore[assignment,misc]

try:
    from esperanto.providers.llm.azure import AzureLanguageModel
except ImportError:
    AzureLanguageModel = None  # type: ignore[assignment,misc]

try:
    from esperanto.providers.llm.mistral import MistralLanguageModel
except ImportError:
    MistralLanguageModel = None  # type: ignore[assignment,misc]

try:
    from esperanto.providers.llm.deepseek import DeepSeekLanguageModel
except ImportError:
    DeepSeekLanguageModel = None  # type: ignore[assignment,misc]

try:
    from esperanto.providers.llm.groq import GroqLanguageModel
except ImportError:
    GroqLanguageModel = None  # type: ignore[assignment,misc]

try:
    from esperanto.providers.embedding.vertex import VertexEmbeddingModel
except ImportError:
    VertexEmbeddingModel = None  # type: ignore[assignment,misc]

try:
    from esperanto.providers.llm.vertex import VertexLanguageModel
except ImportError:
    VertexLanguageModel = None  # type: ignore[assignment,misc]

try:
    from esperanto.providers.tts.vertex import VertexTextToSpeechModel
except ImportError:
    VertexTextToSpeechModel = None  # type: ignore[assignment,misc]

# Store all provider classes
__provider_classes = {
    'AnthropicLanguageModel': AnthropicLanguageModel,
    'GoogleLanguageModel': GoogleLanguageModel,
    'OpenAILanguageModel': OpenAILanguageModel,
    'OpenAICompatibleLanguageModel': OpenAICompatibleLanguageModel,
    'OpenRouterLanguageModel': OpenRouterLanguageModel,
    'XAILanguageModel': XAILanguageModel,
    'OpenAIEmbeddingModel': OpenAIEmbeddingModel,
    'GoogleEmbeddingModel': GoogleEmbeddingModel,
    "AzureEmbeddingModel": AzureEmbeddingModel,
    "OllamaEmbeddingModel": OllamaEmbeddingModel,
    "OllamaLanguageModel": OllamaLanguageModel,
    "AzureLanguageModel": AzureLanguageModel,
    "MistralLanguageModel": MistralLanguageModel,
    "DeepSeekLanguageModel": DeepSeekLanguageModel,
    "GroqLanguageModel": GroqLanguageModel,
    "VertexLanguageModel": VertexLanguageModel,
    "VertexEmbeddingModel": VertexEmbeddingModel,
    "VertexTextToSpeechModel": VertexTextToSpeechModel
}

# Get list of available provider classes (excluding None values)
provider_classes = [name for name, cls in __provider_classes.items() if cls is not None]

# Import factory after defining providers
from esperanto.factory import AIFactory  # noqa: E402

__all__ = [
    # Factory
    "AIFactory",
    # Base classes
    "LanguageModel",
    "EmbeddingModel",
    "SpeechToTextModel",
    "TextToSpeechModel",
    # Tool types
    "Tool",
    "ToolFunction",
    "ToolCall",
    "FunctionCall",
    "ToolCallValidationError",
    "validate_tool_call",
    "validate_tool_calls",
    "find_tool_by_name",
    # Profiles
    "OpenAICompatibleProfile",
] + provider_classes

# Make provider classes available at module level
globals().update({k: v for k, v in __provider_classes.items() if v is not None})