"""Response types for Esperanto."""

import re
import warnings
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Regex pattern to match <think>...</think> blocks (including multiline)
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


# =============================================================================
# Tool-related types
# =============================================================================


class FunctionCall(BaseModel):
    """A function call within a tool call."""

    name: str = Field(default="", description="Name of the function to call")
    arguments: str = Field(default="", description="JSON-encoded arguments for the function")

    model_config = ConfigDict(frozen=True)


class ToolCall(BaseModel):
    """A tool call from the model."""

    id: str = Field(default="", description="Unique identifier for this tool call")
    type: str = Field(default="function", description="Type of tool call")
    function: FunctionCall = Field(description="The function call details")
    index: Optional[int] = Field(
        default=None, description="Index for streaming tool calls"
    )

    model_config = ConfigDict(frozen=True)


class ToolFunction(BaseModel):
    """Definition of a function that can be called as a tool."""

    name: str = Field(description="Name of the function")
    description: str = Field(description="Description of what the function does")
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema for function parameters",
    )
    strict: Optional[bool] = Field(
        default=None, description="Enable strict mode (OpenAI-specific)"
    )

    model_config = ConfigDict(frozen=True)


class Tool(BaseModel):
    """A tool definition to send to the model."""

    type: str = Field(default="function", description="Type of tool")
    function: ToolFunction = Field(description="The function definition")

    model_config = ConfigDict(frozen=True)


# =============================================================================
# Helper functions
# =============================================================================


def to_dict(obj: Any) -> Dict[str, Any]:
    """Convert an object to a dictionary."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    elif isinstance(obj, dict):
        return obj
    return {"content": str(obj)}


class Usage(BaseModel):
    """Usage statistics for a completion."""

    prompt_tokens: int = Field(description="Number of tokens in the prompt", ge=0)
    completion_tokens: int = Field(
        description="Number of tokens in the completion", ge=0
    )
    total_tokens: int = Field(description="Total number of tokens used", ge=0)

    model_config = ConfigDict(frozen=True)


class Message(BaseModel):
    """A message in a chat completion."""

    content: Optional[str] = Field(
        default=None, description="The content of the message"
    )
    role: Optional[str] = Field(
        default=None,
        description="The role of the message sender (e.g., 'system', 'user', 'assistant')",
    )
    function_call: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Deprecated: Use tool_calls instead. Will be removed in v3.0.",
    )
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None, description="Tool calls if the message contains tool invocations"
    )

    def __getitem__(self, key: str) -> Any:
        """Enable dict-like access for backward compatibility."""
        return getattr(self, key)

    @property
    def thinking(self) -> Optional[str]:
        """Extract content inside <think> tags (reasoning trace).

        Returns the concatenated content of all <think>...</think> blocks
        in the message. Returns None if no thinking tags are present or
        if all thinking blocks are empty.

        This is useful for models like Qwen3, DeepSeek R1, and others that
        include chain-of-thought reasoning in their responses.
        """
        if not self.content:
            return None
        matches = _THINK_PATTERN.findall(self.content)
        if not matches:
            return None
        # Concatenate all thinking blocks, stripping whitespace
        non_empty = [match.strip() for match in matches if match.strip()]
        if not non_empty:
            return None
        return "\n\n".join(non_empty)

    @property
    def cleaned_content(self) -> str:
        """Get content with <think> tags removed (actual response).

        Returns the message content with all <think>...</think> blocks
        removed. If there are no thinking tags, returns the original content.

        This is useful for getting the actual response from models that
        include chain-of-thought reasoning in their responses.
        """
        if not self.content:
            return ""
        # Remove all <think>...</think> blocks and clean up whitespace
        cleaned = _THINK_PATTERN.sub("", self.content)
        # Clean up extra whitespace/newlines left behind
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @model_validator(mode="before")
    @classmethod
    def normalize_message_data(cls, data: Any) -> Any:
        """Normalize message data including tool_calls and content."""
        if not isinstance(data, dict):
            data = to_dict(data)

        # Convert mock objects to strings for content field
        if "content" in data and data["content"] is not None:
            try:
                data["content"] = str(data["content"])
            except Exception:
                pass

        # Convert dict tool_calls to ToolCall objects for backward compatibility
        if "tool_calls" in data and data["tool_calls"]:
            normalized = []
            for tc in data["tool_calls"]:
                if isinstance(tc, ToolCall):
                    normalized.append(tc)
                elif isinstance(tc, dict):
                    # Handle nested function dict
                    if "function" in tc and isinstance(tc["function"], dict):
                        func_data = tc["function"]
                        tc = {
                            **tc,
                            "function": FunctionCall(
                                name=func_data.get("name", ""),
                                arguments=func_data.get("arguments", "{}"),
                            ),
                        }
                    normalized.append(ToolCall(**tc))
            data["tool_calls"] = normalized

        return data

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        from_attributes=True,
    )


class DeltaMessage(Message):
    """A delta message in a streaming chat completion."""

    pass


class Choice(BaseModel):
    """A single choice in a chat completion."""

    index: int = Field(description="Index of this choice", ge=0)
    message: Message = Field(description="The message content for this choice")
    finish_reason: Optional[str] = Field(
        default=None,
        description="Reason why the model stopped generating (e.g., 'stop', 'length')",
    )

    @model_validator(mode="before")
    @classmethod
    def ensure_message_type(cls, data: Any) -> Any:
        """Ensure message is the correct type."""
        if not isinstance(data, dict):
            data = to_dict(data)
        if "message" in data:
            if not isinstance(data["message"], Message):
                data["message"] = Message(**to_dict(data["message"]))
        if "finish_reason" in data:
            try:
                data["finish_reason"] = str(data["finish_reason"])
            except Exception:
                data["finish_reason"] = "stop"
        return data

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        from_attributes=True,
    )


class StreamChoice(BaseModel):
    """A single choice in a streaming chat completion."""

    index: int = Field(description="Index of this choice", ge=0)
    delta: DeltaMessage = Field(description="The delta content for this choice")
    finish_reason: Optional[str] = Field(
        default=None,
        description="Reason why the model stopped generating (e.g., 'stop', 'length')",
    )

    @model_validator(mode="before")
    @classmethod
    def ensure_delta_type(cls, data: Any) -> Any:
        """Ensure delta is the correct type."""
        if not isinstance(data, dict):
            data = to_dict(data)
        if "delta" in data:
            if not isinstance(data["delta"], DeltaMessage):
                data["delta"] = DeltaMessage(**to_dict(data["delta"]))
        if "finish_reason" in data:
            try:
                data["finish_reason"] = str(data["finish_reason"])
            except Exception:
                data["finish_reason"] = None
        return data

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        from_attributes=True,
    )


class ChatCompletion(BaseModel):
    """A chat completion response."""

    id: str = Field(description="Unique identifier for this chat completion")
    choices: List[Choice] = Field(description="List of completion choices")
    model: str = Field(description="The model used for completion")
    provider: str = Field(description="The provider of the model")
    created: Optional[int] = Field(
        default=None,
        description="Unix timestamp of when this completion was created",
        ge=0,
    )
    usage: Optional[Usage] = Field(
        default=None, description="Usage statistics for this completion"
    )
    object: str = Field(
        default="chat.completion", description="Object type, always 'chat.completion'"
    )

    @property
    def content(self) -> str:
        """Get the content of the first choice's message."""
        if not self.choices or not self.choices[0].message:
            return ""
        return self.choices[0].message.content or ""

    @model_validator(mode="before")
    @classmethod
    def ensure_choice_types(cls, data: Any) -> Any:
        """Ensure choices are the correct type."""
        if not isinstance(data, dict):
            data = to_dict(data)
        if "choices" in data:
            data["choices"] = [
                (
                    Choice(**to_dict(choice))
                    if not isinstance(choice, Choice)
                    else choice
                )
                for choice in data["choices"]
            ]
        return data

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        from_attributes=True,
    )


class ChatCompletionChunk(BaseModel):
    """A chunk of a streaming chat completion."""

    id: str = Field(description="Unique identifier for this chat completion chunk")
    choices: List[StreamChoice] = Field(
        description="List of completion choices in this chunk"
    )
    model: str = Field(description="The model used for completion")
    created: int = Field(
        description="Unix timestamp of when this chunk was created", ge=0
    )
    object: str = Field(
        default="chat.completion.chunk",
        description="Object type, always 'chat.completion.chunk'",
    )

    @model_validator(mode="before")
    @classmethod
    def ensure_choice_types(cls, data: Any) -> Any:
        """Ensure choices are the correct type."""
        if not isinstance(data, dict):
            data = to_dict(data)
        if "choices" in data:
            data["choices"] = [
                (
                    StreamChoice(**to_dict(choice))
                    if not isinstance(choice, StreamChoice)
                    else choice
                )
                for choice in data["choices"]
            ]
        return data

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        from_attributes=True,
    )
