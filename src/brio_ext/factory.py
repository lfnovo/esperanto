"""BrioDocs-aware factory that extends Esperanto's AIFactory."""

from __future__ import annotations

from copy import deepcopy
from types import MethodType
from typing import Any, Dict, Optional

from esperanto.factory import AIFactory
from esperanto.providers.llm.base import LanguageModel

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

    def chat_complete(self, messages, stream=None):
        rendered = render_for_model(model_id, messages, provider)
        stops = list(rendered.get("stop") or DEFAULT_STOP)

        with _stop_config_guard(self, stops):
            if "messages" in rendered:
                return original_chat(rendered["messages"], stream=stream)

            prompt_handler = getattr(self, "prompt_complete", None)
            if callable(prompt_handler):
                return prompt_handler(rendered["prompt"], stop=stops, stream=stream)
            raise RuntimeError(
                f"Provider '{provider}' cannot render prompts for model '{model_id}'."
            )

    async def achat_complete(self, messages, stream=None):
        rendered = render_for_model(model_id, messages, provider)
        stops = list(rendered.get("stop") or DEFAULT_STOP)

        with _stop_config_guard(self, stops):
            if "messages" in rendered:
                return await original_achat(rendered["messages"], stream=stream)

            prompt_handler = getattr(self, "aprompt_complete", None)
            if callable(prompt_handler):
                return await prompt_handler(rendered["prompt"], stop=stops, stream=stream)

            sync_handler = getattr(self, "prompt_complete", None)
            if callable(sync_handler):
                return sync_handler(rendered["prompt"], stop=stops, stream=stream)

            raise RuntimeError(
                f"Provider '{provider}' cannot render prompts for model '{model_id}'."
            )

    model.chat_complete = MethodType(chat_complete, model)  # type: ignore[assignment]
    model.achat_complete = MethodType(achat_complete, model)  # type: ignore[assignment]
    setattr(model, "_brio_wrapped", True)
    setattr(model, "_brio_model_id", model_id)
    setattr(model, "_brio_provider", provider)
    return model


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
