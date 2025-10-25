"""Unit tests for llama.cpp provider configuration."""

from brio_ext.providers.llamacpp_provider import LlamaCppLanguageModel


def test_llamacpp_max_tokens_maps_to_both_fields():
    model = LlamaCppLanguageModel(
        model_name="qwen2.5-7b-instruct",
        config={"max_tokens": 1200, "temperature": 0.25, "top_p": 0.8},
    )

    kwargs = model._get_api_kwargs()
    assert kwargs["max_tokens"] == 1200
    assert kwargs["n_predict"] == 1200
    assert kwargs["temperature"] == 0.25
    assert kwargs["top_p"] == 0.8

    # Ensure overrides persist on subsequent calls
    model._config["max_tokens"] = 64
    kwargs = model._get_api_kwargs()
    assert kwargs["max_tokens"] == 64
    assert kwargs["n_predict"] == 64
