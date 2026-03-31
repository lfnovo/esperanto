"""LangChain wrapper for brio_ext models that preserves the rendering pipeline."""

from __future__ import annotations

import re
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict, Field

from langchain_core.messages import AIMessage
from loguru import logger
from esperanto.common_types import ChatCompletion
from esperanto.providers.llm.base import LanguageModel
from esperanto.utils.streaming import StreamingFenceFilter, StreamingThinkTagFilter


class BrioLangChainWrapper:
    """
    LangChain-compatible wrapper for brio_ext models.

    This wrapper allows brio_ext models to be used with LangGraph and other
    LangChain tools while preserving brio_ext's chat template rendering and
    response cleaning pipeline.

    Unlike calling model.to_langchain() (which bypasses brio_ext), this wrapper
    calls the brio_ext-wrapped chat_complete() method directly.
    """

    def __init__(self, brio_model: LanguageModel, no_think: bool = False):
        """
        Initialize wrapper with a brio_ext model instance.

        Args:
            brio_model: A LanguageModel instance from BrioAIFactory.create_language()
            no_think: If True, prepend /no_think to the first user message to disable
                      thinking mode on models that support it (e.g. Qwen3/Qwen3.5).
                      Has no effect on models without thinking mode.
        """
        self.brio_model = brio_model
        self.no_think = no_think
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
        response: ChatCompletion = self.brio_model.chat_complete(messages, stream=False, no_think=self.no_think)

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
        response: ChatCompletion = await self.brio_model.achat_complete(messages, stream=False, no_think=self.no_think)

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
        """Async streaming: yields the full response as a single chunk (no token-by-token streaming)."""
        result = await self.ainvoke(input, config, **kwargs)
        yield result

    def _parse_fenced_content(self, raw_content: str) -> str:
        """
        Parse content from brio_ext's <out>...</out> fencing.

        Handles the case where local models put all useful output inside
        <think> tags, which downstream consumers strip away leaving empty
        string.

        Args:
            raw_content: Content with <out>...</out> tags

        Returns:
            Clean content without fencing
        """
        import re

        # Step 1: Extract from <out> or <output> fencing
        content = raw_content
        if "<output>" in raw_content and "</output>" in raw_content:
            start = raw_content.find("<output>") + 8
            end = raw_content.find("</output>")
            content = raw_content[start:end].strip()
        elif "<out>" in raw_content and "</out>" in raw_content:
            start = raw_content.find("<out>") + 5
            end = raw_content.find("</out>")
            content = raw_content[start:end].strip()

        # Step 2: Remove complete <think>...</think> blocks
        think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        think_matches = think_pattern.findall(content)
        cleaned = think_pattern.sub("", content).strip()

        # Step 2c: Strip stray closing tags that leak through from any model's
        # native format (e.g. Phi-4 Mini emits </assistant>, </think> without openers).
        for stray_tag in ("</think>", "</assistant>", "<|im_end|>", "<|end|>"):
            cleaned = cleaned.replace(stray_tag, "").strip()

        # Step 2b: Handle unclosed <think> (model hit token limit mid-reasoning).
        # When the completion budget is exhausted inside a <think> block,
        # </think> is never generated.  Strip from the opening tag to end-of-string
        # so thinking content is never returned to the user.
        unclosed_think = re.compile(r"<think>.*", re.DOTALL)
        if unclosed_think.search(cleaned):
            cleaned = unclosed_think.sub("", cleaned).strip()

        # Step 3: Determine the best content to return

        # 3a: Cleaned content exists — return it (normal path for models without
        # thinking mode, or thinking models whose reasoning fit within budget)
        if cleaned:
            return cleaned

        # 3b: No content outside <think> tags — extract from complete thinking blocks.
        # This is the fix for models that wrap ALL output in <think> tags.
        if think_matches:
            all_thinking = "\n".join(m.strip() for m in think_matches)
            logger.warning(
                f"Model produced only <think> content ({len(all_thinking)} chars) with no "
                f"answer outside tags. Token budget likely exhausted during reasoning."
            )
            # Try to find a JSON object in the thinking content
            json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', all_thinking)
            if json_match:
                return json_match.group(1)
            # No JSON found — return raw thinking as fallback
            return all_thinking

        # Step 4: If content contained <think> but it was unclosed (model was cut
        # off during reasoning with no actual answer), return empty rather than
        # leaking internal reasoning to the user.
        if "<think>" in content:
            logger.warning(
                "Unclosed <think> block detected — model hit token limit mid-reasoning. "
                "Returning empty string to avoid leaking internal reasoning."
            )
            return ""

        return content

    # LangChain compatibility methods
    def bind(self, **kwargs):
        """Bind additional parameters (LangChain interface)."""
        return self  # TODO: Implement if needed

    def bind_tools(self, tools):
        """Bind tools (LangChain interface)."""
        return self  # TODO: Implement if tool support is added


class _AIMessage(AIMessage):
    """AIMessage subclass for LangChain/LangGraph compatibility (used by legacy BrioLangChainWrapper)."""


class BrioBaseChatModel(BaseChatModel):
    """
    LangChain BaseChatModel wrapper for brio_ext models.

    Extends BaseChatModel so that LangGraph's callback system
    (on_chat_model_start, on_llm_new_token, on_llm_end) works correctly,
    enabling stream_mode="messages" support.

    Usage:
        model = BrioAIFactory.create_language("llamacpp", "qwen2.5-7b-instruct", config={...})
        lc_model = BrioBaseChatModel(brio_model=model)
        result = lc_model.invoke("What is 2+2?")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    brio_model: Any = Field(description="A LanguageModel instance from BrioAIFactory")
    no_think: bool = Field(
        default=False,
        description=(
            "If True, prepend /no_think to the first user message to disable "
            "thinking mode on models that support it (e.g. Qwen3/Qwen3.5). "
            "Has no effect on models without thinking mode."
        ),
    )

    def __init__(self, brio_model: LanguageModel, no_think: bool = False, **kwargs: Any):
        super().__init__(brio_model=brio_model, no_think=no_think, **kwargs)
        if not getattr(brio_model, "_brio_wrapped", False):
            raise ValueError(
                "Model must be created via BrioAIFactory to ensure proper wrapping. "
                "Use: BrioAIFactory.create_language(provider='llamacpp', ...)"
            )

    @property
    def _llm_type(self) -> str:
        return "brio_langchain_wrapper"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.brio_model.model_name,
            "provider": getattr(self.brio_model, "_brio_provider", "unknown"),
            "model_id": getattr(self.brio_model, "_brio_model_id", "unknown"),
            "temperature": self.brio_model.temperature,
            "max_tokens": self.brio_model.max_tokens,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        converted = self._convert_messages(messages)
        response: ChatCompletion = self.brio_model.chat_complete(
            converted, stream=False, no_think=self.no_think
        )
        return self._build_chat_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        converted = self._convert_messages(messages)
        response: ChatCompletion = await self.brio_model.achat_complete(
            converted, stream=False, no_think=self.no_think
        )
        return self._build_chat_result(response)

    def _build_chat_result(self, response: ChatCompletion) -> ChatResult:
        raw_content = response.choices[0].message.content
        content = _parse_fenced_content(raw_content)

        usage = response.usage
        response_metadata = {
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason,
        }
        if usage:
            response_metadata["usage"] = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }

        message = AIMessage(
            content=content,
            response_metadata=response_metadata,
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        converted = self._convert_messages(messages)
        stream_response = self.brio_model.chat_complete(converted, stream=True, no_think=self.no_think)
        fence_filter = StreamingFenceFilter()
        think_filter = StreamingThinkTagFilter()
        for chunk in stream_response:
            token = chunk.choices[0].delta.content if chunk.choices else ""
            if token:
                defenced = fence_filter.process(token)
                filtered = think_filter.process(defenced) if defenced else ""
                if filtered:
                    chat_chunk = ChatGenerationChunk(
                        message=AIMessageChunk(content=filtered)
                    )
                    if run_manager:
                        run_manager.on_llm_new_token(filtered, chunk=chat_chunk)
                    yield chat_chunk
        # Flush both filters in order
        remaining = think_filter.process(fence_filter.flush())
        remaining += think_filter.flush()
        if remaining:
            chat_chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=remaining)
            )
            if run_manager:
                run_manager.on_llm_new_token(remaining, chunk=chat_chunk)
            yield chat_chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        converted = self._convert_messages(messages)
        stream_response = await self.brio_model.achat_complete(
            converted, stream=True, no_think=self.no_think
        )
        fence_filter = StreamingFenceFilter()
        think_filter = StreamingThinkTagFilter()
        async for chunk in stream_response:
            token = chunk.choices[0].delta.content if chunk.choices else ""
            if token:
                defenced = fence_filter.process(token)
                filtered = think_filter.process(defenced) if defenced else ""
                if filtered:
                    chat_chunk = ChatGenerationChunk(
                        message=AIMessageChunk(content=filtered)
                    )
                    if run_manager:
                        await run_manager.on_llm_new_token(filtered, chunk=chat_chunk)
                    yield chat_chunk
        # Flush both filters in order
        remaining = think_filter.process(fence_filter.flush())
        remaining += think_filter.flush()
        if remaining:
            chat_chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=remaining)
            )
            if run_manager:
                await run_manager.on_llm_new_token(remaining, chunk=chat_chunk)
            yield chat_chunk

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain BaseMessage objects to dicts for brio_model.chat_complete()."""
        role_map = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "user": "user",
            "assistant": "assistant",
        }
        return [
            {
                "role": role_map.get(msg.type, msg.type),
                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
            }
            for msg in messages
        ]


def _parse_fenced_content(raw_content: str) -> str:
    """
    Parse content from brio_ext's <out>/<output> fencing and strip <think> tags.

    Shared between BrioLangChainWrapper (legacy) and BrioBaseChatModel.

    Handles:
    - <out>...</out> and <output>...</output> fencing extraction
    - Complete <think>...</think> block removal
    - Stray closing tags (</think>, </assistant>, <|im_end|>, <|end|>) without openers
    - Unclosed <think> blocks when model hits token limit mid-reasoning — returns
      empty string rather than leaking internal reasoning to the user
    - Think-only content — returns the thinking content (or extracted JSON) as fallback
    """
    content = raw_content
    if "<output>" in raw_content and "</output>" in raw_content:
        start = raw_content.find("<output>") + 8
        end = raw_content.find("</output>")
        content = raw_content[start:end].strip()
    elif "<out>" in raw_content and "</out>" in raw_content:
        start = raw_content.find("<out>") + 5
        end = raw_content.find("</out>")
        content = raw_content[start:end].strip()

    # Remove complete <think>...</think> blocks
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    think_matches = think_pattern.findall(content)
    cleaned = think_pattern.sub("", content).strip()

    # Strip stray closing tags that leak through from model-native format
    # (e.g. Phi-4 Mini emits </assistant>, models sometimes emit orphan </think>)
    for stray_tag in ("</think>", "</assistant>", "<|im_end|>", "<|end|>"):
        cleaned = cleaned.replace(stray_tag, "").strip()

    # Handle unclosed <think> blocks: model hit token limit mid-reasoning.
    # Strip everything from the opening tag onwards so we never return
    # raw reasoning content to the user.
    unclosed_think = re.compile(r"<think>.*", re.DOTALL)
    if unclosed_think.search(cleaned):
        pre_think = cleaned[: cleaned.find("<think>")].strip()
        if pre_think:
            return pre_think
        logger.warning(
            "Unclosed <think> block detected — model hit token limit mid-reasoning. "
            "Returning empty string to avoid leaking internal reasoning."
        )
        return ""

    # Normal path: content outside think blocks exists — return it
    if cleaned:
        return cleaned

    # Think-only content: model produced reasoning but no answer outside tags.
    if think_matches:
        all_thinking = "\n".join(m.strip() for m in think_matches)
        logger.warning(
            f"Model produced only <think> content ({len(all_thinking)} chars) with no "
            f"answer outside tags. Token budget likely exhausted during reasoning."
        )
        # Extract JSON object from thinking content if present
        json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', all_thinking)
        if json_match:
            return json_match.group(1)
        return all_thinking

    return content

