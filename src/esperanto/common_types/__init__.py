"""Types module for Esperanto."""

from .exceptions import ToolCallValidationError
from .model import Model
from .reranker import RerankResponse, RerankResult
from .response import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    DeltaMessage,
    FunctionCall,
    Message,
    StreamChoice,
    Tool,
    ToolCall,
    ToolFunction,
    Usage,
)
from .stt import TranscriptionResponse
from .task_type import EmbeddingTaskType
from .tts import AudioResponse
from .validation import find_tool_by_name, validate_tool_call, validate_tool_calls

__all__ = [
    # Response types
    "Usage",
    "Message",
    "DeltaMessage",
    "Choice",
    "StreamChoice",
    "ChatCompletion",
    "ChatCompletionChunk",
    # Tool types
    "Tool",
    "ToolFunction",
    "ToolCall",
    "FunctionCall",
    # Validation
    "ToolCallValidationError",
    "validate_tool_call",
    "validate_tool_calls",
    "find_tool_by_name",
    # Other types
    "TranscriptionResponse",
    "AudioResponse",
    "Model",
    "EmbeddingTaskType",
    "RerankResponse",
    "RerankResult",
]
