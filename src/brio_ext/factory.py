"""BrioDocs-aware factory that extends Esperanto's AIFactory."""

from __future__ import annotations

import os
import time
from copy import deepcopy
from types import MethodType
from typing import Any, Dict, Optional

from esperanto.common_types import ChatCompletion, Message
from esperanto.factory import AIFactory
from esperanto.providers.llm.base import LanguageModel

from brio_ext.langchain_wrapper import BrioBaseChatModel, BrioLangChainWrapper
from brio_ext.metrics import MetricsLogger
from brio_ext.renderer import DEFAULT_STOP, render_for_model

# Module-level metrics logger (disabled by default)
_metrics_logger: Optional[MetricsLogger] = None
_metrics_enabled: bool = False

# Initialize from env var at import time
if os.getenv("BRIO_METRICS_ENABLED", "").lower() in ("1", "true", "yes"):
    _metrics_logger = MetricsLogger()
    _metrics_enabled = True


def enable_metrics(log_path: Optional[str] = None) -> None:
    """Enable metrics logging at runtime. Call from Settings page."""
    global _metrics_logger, _metrics_enabled
    from pathlib import Path
    if _metrics_logger is None:
        _metrics_logger = MetricsLogger(log_path=Path(log_path) if log_path else None)
    _metrics_enabled = True


def disable_metrics() -> None:
    """Disable metrics logging at runtime. Call from Settings page."""
    global _metrics_enabled
    _metrics_enabled = False


def is_metrics_enabled() -> bool:
    """Check if metrics logging is currently enabled."""
    return _metrics_enabled


def _log_completion_metrics(
    result: ChatCompletion,
    model_id: str,
    provider: str,
    tier_id: Optional[str] = None,
    tier_label: Optional[str] = None,
    context_size: Optional[int] = None,
    request_time_ms: Optional[float] = None,
) -> None:
    """Log metrics for a completed (non-streaming) response."""
    if not _metrics_enabled or _metrics_logger is None:
        return

    _metrics_logger.log_from_response(
        tier_id=tier_id or provider,  # Fall back to provider if no tier
        model=model_id,
        timings=result.timings.model_dump() if result.timings else None,
        usage=result.usage.model_dump() if result.usage else None,
        tier_label=tier_label,
        context_size=context_size,
        request_time_ms=request_time_ms,
    )


_LANGUAGE_OVERRIDES = {
    "llamacpp": "brio_ext.providers.llamacpp_provider:LlamaCppLanguageModel",
    "hf_local": "brio_ext.providers.hf_local_provider:HuggingFaceLocalLanguageModel",
}


class BrioAIFactory(AIFactory):
    """Factory that injects Brio prompt rendering before provider calls."""

    _provider_modules = deepcopy(AIFactory._provider_modules)
    _provider_modules["language"] = {
        **AIFactory._provider_modules["language"],
        **_LANGUAGE_OVERRIDES,
    }

    @classmethod
    def create_language(
        cls, provider: str, model_name: str, config: Optional[Dict[str, Any]] = None
    ) -> LanguageModel:
        # Use our own provider lookup to ensure _LANGUAGE_OVERRIDES are used
        provider_class = cls._import_provider_class("language", provider)
        model = provider_class(model_name=model_name, config=config or {})
        # Extract chat_format from config if present
        chat_format = (config or {}).get("chat_format")
        return _wrap_language_model(model, model_name, provider, chat_format=chat_format)


def register_with_factory(factory_cls: type[AIFactory]) -> type[AIFactory]:
    """Patch an existing AIFactory subclass to apply Brio rendering."""

    original_create_language = factory_cls.create_language.__func__  # type: ignore[attr-defined]

    def _patched_create_language(
        cls,
        provider: str,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        model = original_create_language(cls, provider, model_name, config=config or {})
        # Extract chat_format from config if present
        chat_format = (config or {}).get("chat_format")
        return _wrap_language_model(model, model_name, provider, chat_format=chat_format)

    factory_cls.create_language = classmethod(_patched_create_language)  # type: ignore[assignment]
    return factory_cls


def _wrap_language_model(
    model: LanguageModel,
    model_id: str,
    provider: str,
    chat_format: Optional[str] = None,
) -> LanguageModel:
    """Attach prompt rendering hooks to a provider instance."""
    if getattr(model, "_brio_wrapped", False):
        return model

    original_chat = model.chat_complete
    original_achat = model.achat_complete

    # Get adapter for response cleaning - use chat_format hint if available
    from brio_ext.registry import get_adapter
    adapter = get_adapter(model_id, chat_format=chat_format)

    def chat_complete(self, messages, stream=None):
        rendered = render_for_model(model_id, messages, provider, chat_format=chat_format)
        stops = list(rendered.get("stop") or DEFAULT_STOP)

        # Extract tier info from config at call time (allows per-request updates)
        config = getattr(self, "_config", {}) or {}
        tier_id = config.get("tier_id")
        tier_label = config.get("tier_label")
        context_size = config.get("context_size")

        with _stop_config_guard(self, stops):
            if "messages" in rendered:
                start_time = time.perf_counter()
                result = original_chat(rendered["messages"], stream=stream)
                request_time_ms = (time.perf_counter() - start_time) * 1000
                fenced = _ensure_fenced_completion(result, adapter)
                # Log metrics for non-streaming responses
                if isinstance(fenced, ChatCompletion):
                    _log_completion_metrics(fenced, model_id, provider, tier_id, tier_label, context_size, request_time_ms)
                return fenced

            prompt_handler = getattr(self, "prompt_complete", None)
            if callable(prompt_handler):
                start_time = time.perf_counter()
                result = prompt_handler(rendered["prompt"], stop=stops, stream=stream)
                request_time_ms = (time.perf_counter() - start_time) * 1000
                fenced = _ensure_fenced_completion(result, adapter)
                if isinstance(fenced, ChatCompletion):
                    _log_completion_metrics(fenced, model_id, provider, tier_id, tier_label, context_size, request_time_ms)
                return fenced
            raise RuntimeError(
                f"Provider '{provider}' cannot render prompts for model '{model_id}'."
            )

    async def achat_complete(self, messages, stream=None):
        rendered = render_for_model(model_id, messages, provider, chat_format=chat_format)
        stops = list(rendered.get("stop") or DEFAULT_STOP)

        # Extract tier info from config at call time (allows per-request updates)
        config = getattr(self, "_config", {}) or {}
        tier_id = config.get("tier_id")
        tier_label = config.get("tier_label")
        context_size = config.get("context_size")

        with _stop_config_guard(self, stops):
            if "messages" in rendered:
                start_time = time.perf_counter()
                result = await original_achat(rendered["messages"], stream=stream)
                request_time_ms = (time.perf_counter() - start_time) * 1000
                fenced = _ensure_fenced_completion(result, adapter)
                if isinstance(fenced, ChatCompletion):
                    _log_completion_metrics(fenced, model_id, provider, tier_id, tier_label, context_size, request_time_ms)
                return fenced

            prompt_handler = getattr(self, "aprompt_complete", None)
            if callable(prompt_handler):
                start_time = time.perf_counter()
                result = await prompt_handler(
                    rendered["prompt"], stop=stops, stream=stream
                )
                request_time_ms = (time.perf_counter() - start_time) * 1000
                fenced = _ensure_fenced_completion(result, adapter)
                if isinstance(fenced, ChatCompletion):
                    _log_completion_metrics(fenced, model_id, provider, tier_id, tier_label, context_size, request_time_ms)
                return fenced

            sync_handler = getattr(self, "prompt_complete", None)
            if callable(sync_handler):
                start_time = time.perf_counter()
                result = sync_handler(rendered["prompt"], stop=stops, stream=stream)
                request_time_ms = (time.perf_counter() - start_time) * 1000
                fenced = _ensure_fenced_completion(result, adapter)
                if isinstance(fenced, ChatCompletion):
                    _log_completion_metrics(fenced, model_id, provider, tier_id, tier_label, context_size, request_time_ms)
                return fenced

            raise RuntimeError(
                f"Provider '{provider}' cannot render prompts for model '{model_id}'."
            )

    model.chat_complete = MethodType(chat_complete, model)  # type: ignore[assignment]
    model.achat_complete = MethodType(achat_complete, model)  # type: ignore[assignment]
    setattr(model, "_brio_wrapped", True)
    setattr(model, "_brio_model_id", model_id)
    setattr(model, "_brio_provider", provider)
    setattr(model, "_brio_chat_format", chat_format)
    model.to_langchain = lambda m=model: BrioBaseChatModel(brio_model=m)
    return model


def _ensure_fenced_completion(result, adapter=None):
    """Ensure completion content is wrapped in <out>...</out> fences."""
    if not isinstance(result, ChatCompletion):
        return result

    choices = []
    for choice in result.choices:
        message = choice.message
        content = message.content or ""
        fenced = _ensure_fence(content, adapter)
        new_message = Message(
            content=fenced,
            role=message.role,
            function_call=message.function_call,
            tool_calls=message.tool_calls,
        )
        choices.append(
            choice.model_copy(update={"message": new_message})
        )

    return result.model_copy(update={"choices": choices})


def _ensure_fence(text: str, adapter=None) -> str:
    """
    Ensure text is wrapped in <out>...</out> fences.

    If LLM generated <out> tags, strip them first and re-fence cleanly.
    This ensures consistent fencing regardless of whether the LLM
    tried to add its own tags.
    """
    stripped = text.strip()
    if not stripped:
        return "<out>\n</out>"

    # Clean model-specific format markers if adapter available
    if adapter:
        stripped = adapter.clean_response(stripped)

    # Generic cleanup: Strip trailing incomplete tokens that models generate when truncated
    # This catches any incomplete special token at the end (e.g., "[/", "<|", "<<", etc.)
    stripped = _strip_trailing_incomplete_tokens(stripped)

    # Strip any LLM-generated <out>/<output> tags before re-fencing
    for open_tag, close_tag in [("<output>", "</output>"), ("<out>", "</out>")]:
        if stripped.startswith(open_tag):
            stripped = stripped[len(open_tag):].lstrip()
        if stripped.endswith(close_tag):
            stripped = stripped[:-len(close_tag)].rstrip()

    # Re-fence with clean tags
    return f"<out>\n{stripped}\n</out>"


def _strip_trailing_incomplete_tokens(text: str) -> str:
    """
    Remove incomplete special tokens at the end of text.

    When models hit max_tokens, they can truncate mid-token, leaving garbage like:
    - "[/" (start of [/SYS], [/INST])
    - "<|" (start of <|eot_id|>)
    - "<<" (start of <<SYS>>)

    This uses a general heuristic: if the text ends with special characters
    that look like the start of a token, strip them.
    """
    import re

    # Strip trailing incomplete bracket/angle tokens
    # Matches: "[", "[/", "[/S", "<|", "<|e", "<<", "<<S", etc. at end of string
    text = re.sub(r'\s*[<\[]+[/|]?[A-Za-z_]*\s*$', '', text)

    return text.strip()


def create_langchain_wrapper(model: LanguageModel) -> BrioBaseChatModel:
    """
    Create a LangChain-compatible wrapper for a brio_ext model.

    This wrapper allows the model to be used with LangGraph and other LangChain
    tools while preserving brio_ext's chat template rendering and response
    cleaning pipeline.

    Args:
        model: A LanguageModel instance from BrioAIFactory.create_language()

    Returns:
        BrioBaseChatModel that can be used with LangChain/LangGraph

    Example:
        >>> model = BrioAIFactory.create_language("llamacpp", "llama-3.1-8b-instruct")
        >>> langchain_model = create_langchain_wrapper(model)
        >>> result = await langchain_model.ainvoke("What is 2+2?")
        >>> print(result.content)  # Clean output, no <out> tags
    """
    return BrioBaseChatModel(brio_model=model)


class _stop_config_guard:
    """Context manager that temporarily injects stop tokens into provider config."""

    _MISSING = object()

    def __init__(self, model: LanguageModel, stops: Optional[list[str]]):
        self.model = model
        self.stops = stops or list(DEFAULT_STOP)
        self._previous = self._MISSING

    def __enter__(self):
        config = getattr(self.model, "_config", None)
        if config is None:
            return None
        self._previous = config.get("stop", self._MISSING)
        config["stop"] = self.stops
        return config

    def __exit__(self, exc_type, exc_val, exc_tb):
        config = getattr(self.model, "_config", None)
        if config is None:
            return False

        if self._previous is self._MISSING:
            config.pop("stop", None)
        else:
            config["stop"] = self._previous
        return False
