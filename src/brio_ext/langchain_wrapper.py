"""LangChain wrapper for brio_ext models that preserves the rendering pipeline."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from esperanto.common_types import ChatCompletion
from esperanto.providers.llm.base import LanguageModel


class BrioLangChainWrapper:
    """
    LangChain-compatible wrapper for brio_ext models.

    This wrapper allows brio_ext models to be used with LangGraph and other
    LangChain tools while preserving brio_ext's chat template rendering and
    response cleaning pipeline.

    Unlike calling model.to_langchain() (which bypasses brio_ext), this wrapper
    calls the brio_ext-wrapped chat_complete() method directly.
    """

    def __init__(self, brio_model: LanguageModel):
        """
        Initialize wrapper with a brio_ext model instance.

        Args:
            brio_model: A LanguageModel instance from BrioAIFactory.create_language()
        """
        self.brio_model = brio_model
        self._llm_type = "brio_langchain_wrapper"

        # Check if model has been wrapped by brio_ext
        if not getattr(brio_model, "_brio_wrapped", False):
            raise ValueError(
                "Model must be created via BrioAIFactory to ensure proper wrapping. "
                "Use: BrioAIFactory.create_language(provider='llamacpp', ...)"
            )

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters for the model."""
        return {
            "model_name": self.brio_model.model_name,
            "provider": getattr(self.brio_model, "_brio_provider", "unknown"),
            "model_id": getattr(self.brio_model, "_brio_model_id", "unknown"),
            "temperature": self.brio_model.temperature,
            "max_tokens": self.brio_model.max_tokens,
        }

    def _convert_messages(self, messages: List[Any]) -> List[Dict[str, str]]:
        """
        Convert LangChain messages to brio_ext format.

        Args:
            messages: LangChain message objects (BaseMessage, SystemMessage, HumanMessage, etc.)

        Returns:
            List of dicts with 'role' and 'content' keys
        """
        converted = []

        for msg in messages:
            # Handle LangChain message objects
            if hasattr(msg, "type"):
                role = msg.type
                content = msg.content if hasattr(msg, "content") else str(msg)
            # Handle dict format
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            # Handle string (treat as user message)
            elif isinstance(msg, str):
                role = "user"
                content = msg
            else:
                raise ValueError(f"Unsupported message type: {type(msg)}")

            # Normalize LangChain role names to OpenAI format
            role_map = {
                "human": "user",
                "ai": "assistant",
                "system": "system",
                "user": "user",
                "assistant": "assistant",
            }
            role = role_map.get(role, role)

            converted.append({"role": role, "content": content})

        return converted

    def invoke(self, input: Any, config: Optional[Dict] = None, **kwargs) -> Any:
        """
        Invoke the model with input (LangChain interface).

        Args:
            input: Either a string or list of messages
            config: Optional configuration
            **kwargs: Additional arguments

        Returns:
            AIMessage-like object with content attribute
        """
        # Convert input to messages format
        if isinstance(input, str):
            # LangChain pattern: single string is typically a system prompt or standalone user message
            # Heuristic: if it contains system prompt indicators, treat as system message
            if any(marker in input[:500].lower() for marker in ["you are a", "you are an", "# system", "system role", "your role is"]):
                messages = [{"role": "system", "content": input}]
            else:
                messages = [{"role": "user", "content": input}]
        elif isinstance(input, list):
            messages = self._convert_messages(input)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

        # Call brio_ext's chat_complete (goes through rendering pipeline!)
        response: ChatCompletion = self.brio_model.chat_complete(messages, stream=False)

        # Extract content and parse out <out>...</out> fencing
        raw_content = response.choices[0].message.content
        content = self._parse_fenced_content(raw_content)

        # Return LangChain-compatible message object
        return _AIMessage(content=content, response_metadata={
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }
        })

    async def ainvoke(self, input: Any, config: Optional[Dict] = None, **kwargs) -> Any:
        """
        Async invoke the model with input (LangChain interface).

        Args:
            input: Either a string or list of messages
            config: Optional configuration
            **kwargs: Additional arguments

        Returns:
            AIMessage-like object with content attribute
        """
        # Convert input to messages format
        if isinstance(input, str):
            # LangChain pattern: single string is typically a system prompt or standalone user message
            # Heuristic: if it contains system prompt indicators, treat as system message
            if any(marker in input[:500].lower() for marker in ["you are a", "you are an", "# system", "system role", "your role is"]):
                messages = [{"role": "system", "content": input}]
            else:
                messages = [{"role": "user", "content": input}]
        elif isinstance(input, list):
            messages = self._convert_messages(input)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

        # Call brio_ext's achat_complete (goes through rendering pipeline!)
        response: ChatCompletion = await self.brio_model.achat_complete(messages, stream=False)

        # Extract content and parse out <out>...</out> fencing
        raw_content = response.choices[0].message.content
        content = self._parse_fenced_content(raw_content)

        # Return LangChain-compatible message object
        return _AIMessage(content=content, response_metadata={
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }
        })

    def stream(self, input: Any, config: Optional[Dict] = None, **kwargs) -> Iterator[Any]:
        """Streaming not yet implemented."""
        raise NotImplementedError("Streaming not yet implemented for BrioLangChainWrapper")

    async def astream(self, input: Any, config: Optional[Dict] = None, **kwargs):
        """Async streaming not yet implemented."""
        raise NotImplementedError("Async streaming not yet implemented for BrioLangChainWrapper")

    def _parse_fenced_content(self, raw_content: str) -> str:
        """
        Parse content from brio_ext's <out>...</out> fencing.

        Args:
            raw_content: Content with <out>...</out> tags

        Returns:
            Clean content without fencing
        """
        if "<out>" in raw_content and "</out>" in raw_content:
            start = raw_content.find("<out>") + 5
            end = raw_content.find("</out>")
            return raw_content[start:end].strip()

        # Fallback: return as-is if no fencing (shouldn't happen with brio_ext)
        return raw_content

    # LangChain compatibility methods
    def bind(self, **kwargs):
        """Bind additional parameters (LangChain interface)."""
        return self  # TODO: Implement if needed

    def bind_tools(self, tools):
        """Bind tools (LangChain interface)."""
        return self  # TODO: Implement if tool support is added


class _AIMessage:
    """Minimal AIMessage-like object for LangChain compatibility."""

    def __init__(self, content: str, response_metadata: Optional[Dict] = None):
        self.content = content
        self.response_metadata = response_metadata or {}
        self.type = "ai"

    def __str__(self):
        return self.content

    def __repr__(self):
        return f"AIMessage(content={self.content[:50]}...)"
