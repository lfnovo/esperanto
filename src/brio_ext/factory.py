"""BrioDocs-aware factory that extends Esperanto's AIFactory."""

from __future__ import annotations

from copy import deepcopy
from types import MethodType
from typing import Any, Dict, Optional

from esperanto.common_types import ChatCompletion, Message
from esperanto.factory import AIFactory
from esperanto.providers.llm.base import LanguageModel

from brio_ext.langchain_wrapper import BrioLangChainWrapper
from brio_ext.renderer import DEFAULT_STOP, render_for_model

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
        model = super().create_language(provider, model_name, config=config or {})
        return _wrap_language_model(model, model_name, provider)


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
        return _wrap_language_model(model, model_name, provider)

    factory_cls.create_language = classmethod(_patched_create_language)  # type: ignore[assignment]
    return factory_cls


def _wrap_language_model(
    model: LanguageModel,
    model_id: str,
    provider: str,
) -> LanguageModel:
    """Attach prompt rendering hooks to a provider instance."""
    if getattr(model, "_brio_wrapped", False):
        return model

    original_chat = model.chat_complete
    original_achat = model.achat_complete

    # Get adapter for response cleaning
    from brio_ext.registry import get_adapter
    adapter = get_adapter(model_id)

    def chat_complete(self, messages, stream=None):
        rendered = render_for_model(model_id, messages, provider)
        stops = list(rendered.get("stop") or DEFAULT_STOP)

        with _stop_config_guard(self, stops):
            if "messages" in rendered:
                result = original_chat(rendered["messages"], stream=stream)
                return _ensure_fenced_completion(result, adapter)

            prompt_handler = getattr(self, "prompt_complete", None)
            if callable(prompt_handler):
                result = prompt_handler(rendered["prompt"], stop=stops, stream=stream)
                return _ensure_fenced_completion(result, adapter)
            raise RuntimeError(
                f"Provider '{provider}' cannot render prompts for model '{model_id}'."
            )

    async def achat_complete(self, messages, stream=None):
        import sys
        print(f"[BRIO_DEBUG] achat_complete called: model_id={model_id}, provider={provider}", file=sys.stderr)
        rendered = render_for_model(model_id, messages, provider)
        print(f"[BRIO_DEBUG] rendered keys: {list(rendered.keys())}", file=sys.stderr)
        stops = list(rendered.get("stop") or DEFAULT_STOP)

        with _stop_config_guard(self, stops):
            if "messages" in rendered:
                print(f"[BRIO_DEBUG] Taking messages-based path (OpenAI-compatible)", file=sys.stderr)
                result = await original_achat(rendered["messages"], stream=stream)
                print(f"[BRIO_DEBUG] Calling _ensure_fenced_completion with adapter={adapter}", file=sys.stderr)
                return _ensure_fenced_completion(result, adapter)

            prompt_handler = getattr(self, "aprompt_complete", None)
            if callable(prompt_handler):
                result = await prompt_handler(
                    rendered["prompt"], stop=stops, stream=stream
                )
                return _ensure_fenced_completion(result, adapter)

            sync_handler = getattr(self, "prompt_complete", None)
            if callable(sync_handler):
                result = sync_handler(rendered["prompt"], stop=stops, stream=stream)
                return _ensure_fenced_completion(result, adapter)

            raise RuntimeError(
                f"Provider '{provider}' cannot render prompts for model '{model_id}'."
            )

    model.chat_complete = MethodType(chat_complete, model)  # type: ignore[assignment]
    model.achat_complete = MethodType(achat_complete, model)  # type: ignore[assignment]
    setattr(model, "_brio_wrapped", True)
    setattr(model, "_brio_model_id", model_id)
    setattr(model, "_brio_provider", provider)
    return model


def _ensure_fenced_completion(result, adapter=None):
    """Ensure completion content is wrapped in <out>...</out> fences."""
    import sys
    print(f"[BRIO_DEBUG] _ensure_fenced_completion called, adapter={adapter}", file=sys.stderr)
    if not isinstance(result, ChatCompletion):
        print(f"[BRIO_DEBUG] Result is not ChatCompletion: {type(result)}", file=sys.stderr)
        return result

    choices = []
    for choice in result.choices:
        message = choice.message
        content = message.content or ""
        print(f"[BRIO_DEBUG] Original content length: {len(content)}, preview: {content[:100]}...", file=sys.stderr)
        fenced = _ensure_fence(content, adapter)
        print(f"[BRIO_DEBUG] Fenced content length: {len(fenced)}, preview: {fenced[:100]}...", file=sys.stderr)
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
    import sys
    print(f"[BRIO_DEBUG] _ensure_fence called, adapter={adapter}", file=sys.stderr)
    stripped = text.strip()
    if not stripped:
        return "<out>\n</out>"

    # Clean model-specific format markers if adapter available
    if adapter:
        print(f"[BRIO_DEBUG] Before cleaning: {stripped[-100:]}", file=sys.stderr)
        stripped = adapter.clean_response(stripped)
        print(f"[BRIO_DEBUG] After cleaning: {stripped[-100:]}", file=sys.stderr)

    # Strip any LLM-generated <out> tags before re-fencing
    if stripped.startswith("<out>"):
        stripped = stripped[5:].lstrip()  # Remove opening tag and whitespace
    if stripped.endswith("</out>"):
        stripped = stripped[:-6].rstrip()  # Remove closing tag and whitespace

    # Re-fence with clean tags
    return f"<out>\n{stripped}\n</out>"


def create_langchain_wrapper(model: LanguageModel) -> BrioLangChainWrapper:
    """
    Create a LangChain-compatible wrapper for a brio_ext model.

    This wrapper allows the model to be used with LangGraph and other LangChain
    tools while preserving brio_ext's chat template rendering and response
    cleaning pipeline.

    Args:
        model: A LanguageModel instance from BrioAIFactory.create_language()

    Returns:
        BrioLangChainWrapper that can be used with LangChain/LangGraph

    Example:
        >>> model = BrioAIFactory.create_language("llamacpp", "llama-3.1-8b-instruct")
        >>> langchain_model = create_langchain_wrapper(model)
        >>> result = await langchain_model.ainvoke("What is 2+2?")
        >>> print(result.content)  # Clean output, no <out> tags
    """
    return BrioLangChainWrapper(model)


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
