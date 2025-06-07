"""Response types for Esperanto."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
        description="Function call details if the message is a function call",
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Tool calls if the message contains tool invocations"
    )

    def __getitem__(self, key: str) -> Any:
        """Enable dict-like access for backward compatibility."""
        return getattr(self, key)

    @model_validator(mode="before")
    @classmethod
    def convert_mock_content(cls, data: Any) -> Any:
        """Convert mock objects to strings for content field."""
        if not isinstance(data, dict):
            data = to_dict(data)
        if "content" in data and data["content"] is not None:
            try:
                data["content"] = str(data["content"])
            except Exception:
                pass
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
