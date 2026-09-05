"""Regression tests for keeping credentials out of provider representations."""

from dataclasses import fields

import pytest

from esperanto.providers.embedding.base import EmbeddingModel
from esperanto.providers.embedding.openrouter import OpenRouterEmbeddingModel
from esperanto.providers.llm.base import LanguageModel
from esperanto.providers.llm.openai import OpenAILanguageModel
from esperanto.providers.llm.openai_compatible import OpenAICompatibleLanguageModel
from esperanto.providers.llm.openrouter import OpenRouterLanguageModel
from esperanto.providers.llm.perplexity import PerplexityLanguageModel
from esperanto.providers.reranker.base import RerankerModel
from esperanto.providers.stt.base import SpeechToTextModel
from esperanto.providers.tts.base import TextToSpeechModel


@pytest.mark.parametrize(
    "model_class",
    [
        LanguageModel,
        EmbeddingModel,
        RerankerModel,
        SpeechToTextModel,
        TextToSpeechModel,
        OpenAICompatibleLanguageModel,
        OpenRouterLanguageModel,
        PerplexityLanguageModel,
        OpenRouterEmbeddingModel,
    ],
)
def test_api_keys_are_excluded_from_dataclass_repr(model_class):
    api_key_field = next(field for field in fields(model_class) if field.name == "api_key")

    assert api_key_field.repr is False


@pytest.mark.parametrize(
    "model_class",
    [LanguageModel, EmbeddingModel, RerankerModel, SpeechToTextModel, TextToSpeechModel],
)
def test_config_is_excluded_from_dataclass_repr(model_class):
    config_field = next(field for field in fields(model_class) if field.name == "config")

    assert config_field.repr is False


def test_provider_repr_does_not_contain_direct_or_configured_secrets():
    direct_secret = "direct-secret-value"
    configured_secret = "configured-secret-value"
    model = OpenAILanguageModel(
        api_key=direct_secret,
        config={"api_key": configured_secret, "custom_header": configured_secret},
    )

    try:
        representation = repr(model)
    finally:
        model.close()

    assert direct_secret not in representation
    assert configured_secret not in representation
    assert "api_key=" not in representation
    assert "config=" not in representation

