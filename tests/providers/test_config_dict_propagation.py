"""Tests for ``config`` dict parameter propagation across providers (issue #91).

Naming convention: each provider gets one test per construction path:

* ``test_<provider>_<type>_config_propagates_direct`` — direct instantiation.
* ``test_<provider>_<type>_config_propagates_factory`` — via ``AIFactory``.

Tests do not perform any HTTP calls. Provider construction in this codebase is
hermetic: clients are created but no requests are issued.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import pytest

from esperanto.factory import AIFactory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scrub_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove any provider env vars that could leak into a test.

    Tests in this module set their own env vars (or supply via config) and
    assume nothing else is on the environment. Strip the well-known ones up
    front so test order and host environment cannot influence outcomes.
    """
    for var in [
        # LLM
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_API_BASE_URL",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "OLLAMA_BASE_URL", "OLLAMA_API_BASE",
        "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL",
        "PERPLEXITY_API_KEY", "PERPLEXITY_BASE_URL",
        "OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL",
        "OPENAI_COMPATIBLE_API_KEY_LLM", "OPENAI_COMPATIBLE_BASE_URL_LLM",
        "OPENAI_COMPATIBLE_API_KEY_EMBEDDING", "OPENAI_COMPATIBLE_BASE_URL_EMBEDDING",
        "OPENAI_COMPATIBLE_API_KEY_STT", "OPENAI_COMPATIBLE_BASE_URL_STT",
        "OPENAI_COMPATIBLE_API_KEY_TTS", "OPENAI_COMPATIBLE_BASE_URL_TTS",
        "VERTEX_PROJECT", "GOOGLE_CLOUD_PROJECT", "VERTEX_LOCATION",
        # Embedding / Audio
        "JINA_API_KEY",
        "VOYAGE_API_KEY",
        "ELEVENLABS_API_KEY",
        "XAI_API_KEY", "XAI_BASE_URL",
        # Azure
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION",
        "OPENAI_API_VERSION",
        "AZURE_OPENAI_API_KEY_LLM", "AZURE_OPENAI_ENDPOINT_LLM", "AZURE_OPENAI_API_VERSION_LLM",
        "AZURE_OPENAI_API_KEY_EMBEDDING", "AZURE_OPENAI_ENDPOINT_EMBEDDING", "AZURE_OPENAI_API_VERSION_EMBEDDING",
        "AZURE_OPENAI_API_KEY_STT", "AZURE_OPENAI_ENDPOINT_STT", "AZURE_OPENAI_API_VERSION_STT",
        "AZURE_OPENAI_API_KEY_TTS", "AZURE_OPENAI_ENDPOINT_TTS", "AZURE_OPENAI_API_VERSION_TTS",
    ]:
        monkeypatch.delenv(var, raising=False)


def _assert_attrs(model: Any, expected: Dict[str, Any]) -> None:
    """Assert every (attr, expected) pair on the constructed model."""
    for attr, want in expected.items():
        got = getattr(model, attr)
        assert got == want, (
            f"expected {attr}={want!r} but got {got!r} on "
            f"{type(model).__name__}"
        )


def _check_propagation(
    *,
    direct_factory: Callable[[], Any],
    factory_factory: Callable[[], Any],
    expected: Dict[str, Any],
) -> None:
    """Construct via both paths and assert attributes match expectations."""
    direct_model = direct_factory()
    _assert_attrs(direct_model, expected)

    factory_model = factory_factory()
    _assert_attrs(factory_model, expected)


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------


def test_openai_llm_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.llm.openai import OpenAILanguageModel

    config = {"api_key": "config-openai-key", "base_url": "https://openai.example.com/v1"}
    _check_propagation(
        direct_factory=lambda: OpenAILanguageModel(model_name="gpt-4o", config=dict(config)),
        factory_factory=lambda: AIFactory.create_language("openai", "gpt-4o", config=dict(config)),
        expected={"api_key": "config-openai-key", "base_url": "https://openai.example.com/v1"},
    )


def test_anthropic_llm_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.llm.anthropic import AnthropicLanguageModel

    config = {"api_key": "config-anthropic-key", "base_url": "https://anthropic.example.com/v1"}
    _check_propagation(
        direct_factory=lambda: AnthropicLanguageModel(model_name="claude-3-5-sonnet-latest", config=dict(config)),
        factory_factory=lambda: AIFactory.create_language("anthropic", "claude-3-5-sonnet-latest", config=dict(config)),
        expected={"api_key": "config-anthropic-key", "base_url": "https://anthropic.example.com/v1"},
    )


def test_mistral_llm_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.llm.mistral import MistralLanguageModel

    config = {"api_key": "config-mistral-key", "base_url": "https://mistral.example.com/v1"}
    _check_propagation(
        direct_factory=lambda: MistralLanguageModel(model_name="mistral-large-latest", config=dict(config)),
        factory_factory=lambda: AIFactory.create_language("mistral", "mistral-large-latest", config=dict(config)),
        expected={"api_key": "config-mistral-key", "base_url": "https://mistral.example.com/v1"},
    )


def test_ollama_llm_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.llm.ollama import OllamaLanguageModel

    config = {"base_url": "http://ollama.example.com:11434"}
    _check_propagation(
        direct_factory=lambda: OllamaLanguageModel(model_name="llama3", config=dict(config)),
        factory_factory=lambda: AIFactory.create_language("ollama", "llama3", config=dict(config)),
        expected={"base_url": "http://ollama.example.com:11434"},
    )


def test_openrouter_llm_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.llm.openrouter import OpenRouterLanguageModel

    config = {"api_key": "config-openrouter-key", "base_url": "https://openrouter.example.com/api/v1"}
    _check_propagation(
        direct_factory=lambda: OpenRouterLanguageModel(model_name="meta-llama/llama-3-70b-instruct", config=dict(config)),
        factory_factory=lambda: AIFactory.create_language(
            "openrouter", "meta-llama/llama-3-70b-instruct", config=dict(config)
        ),
        expected={
            "api_key": "config-openrouter-key",
            "base_url": "https://openrouter.example.com/api/v1",
        },
    )


def test_perplexity_llm_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.llm.perplexity import PerplexityLanguageModel

    config = {"api_key": "config-perplexity-key", "base_url": "https://perplexity.example.com"}
    _check_propagation(
        direct_factory=lambda: PerplexityLanguageModel(model_name="sonar", config=dict(config)),
        factory_factory=lambda: AIFactory.create_language("perplexity", "sonar", config=dict(config)),
        expected={
            "api_key": "config-perplexity-key",
            "base_url": "https://perplexity.example.com",
        },
    )


def test_openai_compatible_llm_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.llm.openai_compatible import OpenAICompatibleLanguageModel

    config = {"api_key": "config-key", "base_url": "http://compat.example.com:1234/v1"}
    _check_propagation(
        direct_factory=lambda: OpenAICompatibleLanguageModel(model_name="local-model", config=dict(config)),
        factory_factory=lambda: AIFactory.create_language(
            "openai-compatible", "local-model", config=dict(config)
        ),
        expected={"api_key": "config-key", "base_url": "http://compat.example.com:1234/v1"},
    )


def test_google_llm_api_key_config_propagates(monkeypatch):
    """Google LLM derives ``base_url`` from ``GEMINI_API_BASE_URL`` and ignores
    config-supplied ``base_url``; only ``api_key`` is honored from config."""
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.llm.google import GoogleLanguageModel

    config = {"api_key": "config-google-key"}
    _check_propagation(
        direct_factory=lambda: GoogleLanguageModel(model_name="gemini-2.0-flash", config=dict(config)),
        factory_factory=lambda: AIFactory.create_language("google", "gemini-2.0-flash", config=dict(config)),
        expected={"api_key": "config-google-key"},
    )


def test_groq_llm_api_key_config_propagates(monkeypatch):
    """Groq LLM hardcodes ``base_url``; only ``api_key`` is honored."""
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.llm.groq import GroqLanguageModel

    config = {"api_key": "config-groq-key"}
    _check_propagation(
        direct_factory=lambda: GroqLanguageModel(model_name="llama-3.1-70b-versatile", config=dict(config)),
        factory_factory=lambda: AIFactory.create_language(
            "groq", "llama-3.1-70b-versatile", config=dict(config)
        ),
        expected={"api_key": "config-groq-key"},
    )


def test_azure_llm_config_propagates(monkeypatch):
    """Azure LLM uses ``azure_endpoint`` (or ``base_url``) from config."""
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.llm.azure import AzureLanguageModel

    config = {
        "api_key": "config-azure-key",
        "azure_endpoint": "https://my-azure.example.com",
        "api_version": "2024-08-01-preview",
    }
    _check_propagation(
        direct_factory=lambda: AzureLanguageModel(model_name="gpt-4o", config=dict(config)),
        factory_factory=lambda: AIFactory.create_language("azure", "gpt-4o", config=dict(config)),
        expected={
            "api_key": "config-azure-key",
            "azure_endpoint": "https://my-azure.example.com",
            "api_version": "2024-08-01-preview",
        },
    )


# ---------------------------------------------------------------------------
# Embedding providers
# ---------------------------------------------------------------------------


def test_openai_embedding_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.embedding.openai import OpenAIEmbeddingModel

    config = {"api_key": "config-openai-key", "base_url": "https://openai.example.com/v1"}
    _check_propagation(
        direct_factory=lambda: OpenAIEmbeddingModel(model_name="text-embedding-3-small", config=dict(config)),
        factory_factory=lambda: AIFactory.create_embedding(
            "openai", "text-embedding-3-small", config=dict(config)
        ),
        expected={"api_key": "config-openai-key", "base_url": "https://openai.example.com/v1"},
    )


def test_openai_compatible_embedding_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.embedding.openai_compatible import (
        OpenAICompatibleEmbeddingModel,
    )

    config = {"api_key": "config-key", "base_url": "http://compat.example.com:1234/v1"}
    _check_propagation(
        direct_factory=lambda: OpenAICompatibleEmbeddingModel(
            model_name="nomic-embed-text", config=dict(config)
        ),
        factory_factory=lambda: AIFactory.create_embedding(
            "openai-compatible", "nomic-embed-text", config=dict(config)
        ),
        expected={"api_key": "config-key", "base_url": "http://compat.example.com:1234/v1"},
    )


def test_ollama_embedding_config_propagates(monkeypatch):
    """Regression test for issue #90: config-supplied ``base_url`` must reach
    ``self.base_url`` on the Ollama embedding provider."""
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.embedding.ollama import OllamaEmbeddingModel

    config = {"base_url": "http://ollama.example.com:11434"}
    _check_propagation(
        direct_factory=lambda: OllamaEmbeddingModel(model_name="nomic-embed-text", config=dict(config)),
        factory_factory=lambda: AIFactory.create_embedding(
            "ollama", "nomic-embed-text", config=dict(config)
        ),
        expected={"base_url": "http://ollama.example.com:11434"},
    )


def test_voyage_embedding_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.embedding.voyage import VoyageEmbeddingModel

    config = {"api_key": "config-voyage-key", "base_url": "https://voyage.example.com/v1"}
    _check_propagation(
        direct_factory=lambda: VoyageEmbeddingModel(model_name="voyage-3", config=dict(config)),
        factory_factory=lambda: AIFactory.create_embedding("voyage", "voyage-3", config=dict(config)),
        expected={"api_key": "config-voyage-key", "base_url": "https://voyage.example.com/v1"},
    )


def test_jina_embedding_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.embedding.jina import JinaEmbeddingModel

    config = {"api_key": "config-jina-key", "base_url": "https://jina.example.com/v1/embeddings"}
    _check_propagation(
        direct_factory=lambda: JinaEmbeddingModel(model_name="jina-embeddings-v3", config=dict(config)),
        factory_factory=lambda: AIFactory.create_embedding(
            "jina", "jina-embeddings-v3", config=dict(config)
        ),
        expected={
            "api_key": "config-jina-key",
            "base_url": "https://jina.example.com/v1/embeddings",
        },
    )


def test_openrouter_embedding_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.embedding.openrouter import OpenRouterEmbeddingModel

    config = {
        "api_key": "config-openrouter-key",
        "base_url": "https://openrouter.example.com/api/v1",
    }
    _check_propagation(
        direct_factory=lambda: OpenRouterEmbeddingModel(model_name="text-embedding-3-small", config=dict(config)),
        factory_factory=lambda: AIFactory.create_embedding(
            "openrouter", "text-embedding-3-small", config=dict(config)
        ),
        expected={
            "api_key": "config-openrouter-key",
            "base_url": "https://openrouter.example.com/api/v1",
        },
    )


def test_mistral_embedding_api_key_config_propagates(monkeypatch):
    """Mistral embedding hardcodes ``base_url``; only ``api_key`` is honored."""
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.embedding.mistral import MistralEmbeddingModel

    config = {"api_key": "config-mistral-key"}
    _check_propagation(
        direct_factory=lambda: MistralEmbeddingModel(model_name="mistral-embed", config=dict(config)),
        factory_factory=lambda: AIFactory.create_embedding(
            "mistral", "mistral-embed", config=dict(config)
        ),
        expected={"api_key": "config-mistral-key"},
    )


def test_google_embedding_api_key_config_propagates(monkeypatch):
    """Google embedding derives ``base_url`` from env; only ``api_key`` is honored."""
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.embedding.google import GoogleEmbeddingModel

    config = {"api_key": "config-google-key"}
    _check_propagation(
        direct_factory=lambda: GoogleEmbeddingModel(model_name="text-embedding-004", config=dict(config)),
        factory_factory=lambda: AIFactory.create_embedding(
            "google", "text-embedding-004", config=dict(config)
        ),
        expected={"api_key": "config-google-key"},
    )


def test_azure_embedding_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.embedding.azure import AzureEmbeddingModel

    config = {
        "api_key": "config-azure-key",
        "azure_endpoint": "https://my-azure.example.com",
        "api_version": "2024-08-01-preview",
    }
    _check_propagation(
        direct_factory=lambda: AzureEmbeddingModel(model_name="text-embedding-ada-002", config=dict(config)),
        factory_factory=lambda: AIFactory.create_embedding(
            "azure", "text-embedding-ada-002", config=dict(config)
        ),
        expected={
            "api_key": "config-azure-key",
            "azure_endpoint": "https://my-azure.example.com",
            "api_version": "2024-08-01-preview",
        },
    )


# ---------------------------------------------------------------------------
# Reranker providers
# ---------------------------------------------------------------------------


def test_jina_reranker_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.reranker.jina import JinaRerankerModel

    config = {"api_key": "config-jina-key", "base_url": "https://jina.example.com/v1"}
    _check_propagation(
        direct_factory=lambda: JinaRerankerModel(model_name="jina-reranker-v2-base-multilingual", config=dict(config)),
        factory_factory=lambda: AIFactory.create_reranker(
            "jina", "jina-reranker-v2-base-multilingual", config=dict(config)
        ),
        expected={"api_key": "config-jina-key", "base_url": "https://jina.example.com/v1"},
    )


def test_voyage_reranker_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.reranker.voyage import VoyageRerankerModel

    config = {"api_key": "config-voyage-key", "base_url": "https://voyage.example.com/v1"}
    _check_propagation(
        direct_factory=lambda: VoyageRerankerModel(model_name="rerank-2", config=dict(config)),
        factory_factory=lambda: AIFactory.create_reranker(
            "voyage", "rerank-2", config=dict(config)
        ),
        expected={"api_key": "config-voyage-key", "base_url": "https://voyage.example.com/v1"},
    )


# ---------------------------------------------------------------------------
# Speech-to-Text providers
# ---------------------------------------------------------------------------


def test_openai_stt_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.stt.openai import OpenAISpeechToTextModel

    config = {"api_key": "config-openai-key", "base_url": "https://openai.example.com/v1"}
    _check_propagation(
        direct_factory=lambda: OpenAISpeechToTextModel(model_name="whisper-1", config=dict(config)),
        factory_factory=lambda: AIFactory.create_speech_to_text(
            "openai", "whisper-1", config=dict(config)
        ),
        expected={"api_key": "config-openai-key", "base_url": "https://openai.example.com/v1"},
    )


def test_groq_stt_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.stt.groq import GroqSpeechToTextModel

    config = {"api_key": "config-groq-key", "base_url": "https://groq.example.com/openai/v1"}
    _check_propagation(
        direct_factory=lambda: GroqSpeechToTextModel(model_name="whisper-large-v3", config=dict(config)),
        factory_factory=lambda: AIFactory.create_speech_to_text(
            "groq", "whisper-large-v3", config=dict(config)
        ),
        expected={"api_key": "config-groq-key", "base_url": "https://groq.example.com/openai/v1"},
    )


def test_elevenlabs_stt_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.stt.elevenlabs import ElevenLabsSpeechToTextModel

    config = {"api_key": "config-elevenlabs-key", "base_url": "https://elevenlabs.example.com/v1"}
    _check_propagation(
        direct_factory=lambda: ElevenLabsSpeechToTextModel(model_name="scribe_v1", config=dict(config)),
        factory_factory=lambda: AIFactory.create_speech_to_text(
            "elevenlabs", "scribe_v1", config=dict(config)
        ),
        expected={
            "api_key": "config-elevenlabs-key",
            "base_url": "https://elevenlabs.example.com/v1",
        },
    )


def test_mistral_stt_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.stt.mistral import MistralSpeechToTextModel

    config = {"api_key": "config-mistral-key", "base_url": "https://mistral.example.com/v1"}
    _check_propagation(
        direct_factory=lambda: MistralSpeechToTextModel(model_name="voxtral-mini-latest", config=dict(config)),
        factory_factory=lambda: AIFactory.create_speech_to_text(
            "mistral", "voxtral-mini-latest", config=dict(config)
        ),
        expected={
            "api_key": "config-mistral-key",
            "base_url": "https://mistral.example.com/v1",
        },
    )


def test_openai_compatible_stt_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.stt.openai_compatible import (
        OpenAICompatibleSpeechToTextModel,
    )

    config = {"api_key": "config-key", "base_url": "http://compat.example.com:8000"}
    _check_propagation(
        direct_factory=lambda: OpenAICompatibleSpeechToTextModel(
            model_name="faster-whisper", config=dict(config)
        ),
        factory_factory=lambda: AIFactory.create_speech_to_text(
            "openai-compatible", "faster-whisper", config=dict(config)
        ),
        expected={"api_key": "config-key", "base_url": "http://compat.example.com:8000"},
    )


def test_google_stt_api_key_config_propagates(monkeypatch):
    """Google STT derives ``base_url`` from env; only ``api_key`` is honored."""
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.stt.google import GoogleSpeechToTextModel

    config = {"api_key": "config-google-key"}
    _check_propagation(
        direct_factory=lambda: GoogleSpeechToTextModel(model_name="gemini-2.5-flash", config=dict(config)),
        factory_factory=lambda: AIFactory.create_speech_to_text(
            "google", "gemini-2.5-flash", config=dict(config)
        ),
        expected={"api_key": "config-google-key"},
    )


def test_azure_stt_config_propagates_direct(monkeypatch):
    """Azure STT honors config dict on direct instantiation.

    The factory path is exercised separately in
    ``test_azure_stt_config_propagates_factory_xfail`` because the factory
    unpacks ``**config`` as kwargs, and ``AzureSpeechToTextModel`` does not
    declare ``azure_endpoint`` as a dataclass field, so ``azure_endpoint``
    cannot reach the constructor that way today.
    """
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.stt.azure import AzureSpeechToTextModel

    config = {
        "api_key": "config-azure-key",
        "azure_endpoint": "https://my-azure.example.com",
        "api_version": "2024-08-01-preview",
    }
    direct = AzureSpeechToTextModel(model_name="whisper-1", config=dict(config))
    _assert_attrs(direct, {
        "api_key": "config-azure-key",
        "azure_endpoint": "https://my-azure.example.com",
        "api_version": "2024-08-01-preview",
    })


@pytest.mark.xfail(
    reason=(
        "Tracking issue: AIFactory.create_speech_to_text unpacks **config as "
        "kwargs, but AzureSpeechToTextModel does not declare azure_endpoint "
        "as a dataclass field, so it cannot be supplied via the factory's "
        "config dict. Other Azure providers (LLM/embedding/TTS) accept "
        "config={'azure_endpoint': ...} fine. This xfail is intentionally "
        "left as a regression marker for issue #91-class bugs."
    ),
    raises=TypeError,
    strict=True,
)
def test_azure_stt_config_propagates_factory_xfail(monkeypatch):
    _scrub_provider_env(monkeypatch)

    config = {
        "api_key": "config-azure-key",
        "azure_endpoint": "https://my-azure.example.com",
        "api_version": "2024-08-01-preview",
    }
    factory_model = AIFactory.create_speech_to_text(
        "azure", "whisper-1", config=dict(config)
    )
    _assert_attrs(factory_model, {
        "api_key": "config-azure-key",
        "azure_endpoint": "https://my-azure.example.com",
        "api_version": "2024-08-01-preview",
    })


# ---------------------------------------------------------------------------
# Text-to-Speech providers
#
# TTS factory dispatch unpacks ``config`` as kwargs (see ``factory.py``), so
# these tests indirectly verify that several providers' custom ``__init__``
# implementations re-wrap kwargs back into ``config`` correctly.
# ---------------------------------------------------------------------------


def test_openai_tts_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.tts.openai import OpenAITextToSpeechModel

    # OpenAITextToSpeechModel.__init__ takes ``api_key`` and ``base_url`` as
    # explicit kwargs; ``**kwargs`` is forwarded to the base class as
    # ``config=kwargs``. Direct construction therefore uses keyword args, not
    # a ``config=`` dict (unlike LLM/embedding providers).
    direct = OpenAITextToSpeechModel(
        model_name="tts-1",
        api_key="config-openai-key",
        base_url="https://openai.example.com/v1",
    )
    _assert_attrs(direct, {
        "api_key": "config-openai-key",
        "base_url": "https://openai.example.com/v1",
    })

    # The factory unpacks ``**config`` into kwargs, which routes the same way.
    factory_model = AIFactory.create_text_to_speech(
        "openai",
        model_name="tts-1",
        config={"api_key": "config-openai-key", "base_url": "https://openai.example.com/v1"},
    )
    _assert_attrs(factory_model, {
        "api_key": "config-openai-key",
        "base_url": "https://openai.example.com/v1",
    })


def test_elevenlabs_tts_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "env-elevenlabs-key")
    from esperanto.providers.tts.elevenlabs import ElevenLabsTextToSpeechModel

    direct = ElevenLabsTextToSpeechModel(
        model_name="eleven_multilingual_v2",
        api_key="config-elevenlabs-key",
        base_url="https://elevenlabs.example.com",
    )
    _assert_attrs(direct, {
        "api_key": "config-elevenlabs-key",
        "base_url": "https://elevenlabs.example.com",
    })

    factory_model = AIFactory.create_text_to_speech(
        "elevenlabs",
        model_name="eleven_multilingual_v2",
        config={"api_key": "config-elevenlabs-key", "base_url": "https://elevenlabs.example.com"},
    )
    _assert_attrs(factory_model, {
        "api_key": "config-elevenlabs-key",
        "base_url": "https://elevenlabs.example.com",
    })


def test_openai_compatible_tts_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.tts.openai_compatible import (
        OpenAICompatibleTextToSpeechModel,
    )

    direct = OpenAICompatibleTextToSpeechModel(
        model_name="piper-tts",
        config={"api_key": "config-key", "base_url": "http://compat.example.com:8000"},
    )
    _assert_attrs(direct, {
        "api_key": "config-key",
        "base_url": "http://compat.example.com:8000",
    })

    factory_model = AIFactory.create_text_to_speech(
        "openai-compatible",
        model_name="piper-tts",
        config={"api_key": "config-key", "base_url": "http://compat.example.com:8000"},
    )
    _assert_attrs(factory_model, {
        "api_key": "config-key",
        "base_url": "http://compat.example.com:8000",
    })


def test_xai_tts_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.tts.xai import XAITextToSpeechModel

    direct = XAITextToSpeechModel(
        api_key="config-xai-key",
        base_url="https://xai.example.com",
    )
    _assert_attrs(direct, {
        "api_key": "config-xai-key",
        "base_url": "https://xai.example.com",
    })

    factory_model = AIFactory.create_text_to_speech(
        "xai",
        config={"api_key": "config-xai-key", "base_url": "https://xai.example.com"},
    )
    _assert_attrs(factory_model, {
        "api_key": "config-xai-key",
        "base_url": "https://xai.example.com",
    })


def test_mistral_tts_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.tts.mistral import MistralTextToSpeechModel

    config = {"api_key": "config-mistral-key", "base_url": "https://mistral.example.com/v1"}
    direct = MistralTextToSpeechModel(
        model_name="voxtral-mini-tts-2603", config=dict(config)
    )
    _assert_attrs(direct, {
        "api_key": "config-mistral-key",
        "base_url": "https://mistral.example.com/v1",
    })

    factory_model = AIFactory.create_text_to_speech(
        "mistral",
        model_name="voxtral-mini-tts-2603",
        config=dict(config),
    )
    _assert_attrs(factory_model, {
        "api_key": "config-mistral-key",
        "base_url": "https://mistral.example.com/v1",
    })


def test_google_tts_api_key_config_propagates(monkeypatch):
    """Google TTS derives ``base_url`` from env; only ``api_key`` is honored."""
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.tts.google import GoogleTextToSpeechModel

    direct = GoogleTextToSpeechModel(
        model_name="gemini-2.5-flash-preview-tts",
        config={"api_key": "config-google-key"},
    )
    _assert_attrs(direct, {"api_key": "config-google-key"})

    factory_model = AIFactory.create_text_to_speech(
        "google",
        model_name="gemini-2.5-flash-preview-tts",
        config={"api_key": "config-google-key"},
    )
    _assert_attrs(factory_model, {"api_key": "config-google-key"})


def test_azure_tts_config_propagates(monkeypatch):
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.tts.azure import AzureTextToSpeechModel

    config = {
        "api_key": "config-azure-key",
        "azure_endpoint": "https://my-azure.example.com",
        "api_version": "2024-08-01-preview",
    }
    direct = AzureTextToSpeechModel(model_name="tts-1", **config)
    _assert_attrs(direct, {
        "api_key": "config-azure-key",
        "azure_endpoint": "https://my-azure.example.com",
        "api_version": "2024-08-01-preview",
    })

    factory_model = AIFactory.create_text_to_speech(
        "azure", model_name="tts-1", config=dict(config)
    )
    _assert_attrs(factory_model, {
        "api_key": "config-azure-key",
        "azure_endpoint": "https://my-azure.example.com",
        "api_version": "2024-08-01-preview",
    })


# ---------------------------------------------------------------------------
# OpenAI-compatible profile flow (issue #91 also implicitly covers this path,
# since profile-routed providers like ``xai`` for LLM go through a different
# branch in ``factory.create_language``).
# ---------------------------------------------------------------------------


def test_xai_llm_profile_config_propagates(monkeypatch):
    """LLM providers routed via ``OpenAICompatibleProfile`` (e.g. ``xai``) must
    still propagate ``base_url`` and ``api_key`` from the user-supplied config
    after the profile merge in ``AIFactory.create_language``."""
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.llm.openai_compatible import OpenAICompatibleLanguageModel

    config = {
        "api_key": "config-xai-key",
        "base_url": "https://xai.example.com/v1",
    }
    factory_model = AIFactory.create_language(
        "xai", "grok-2-latest", config=dict(config)
    )
    assert isinstance(factory_model, OpenAICompatibleLanguageModel)
    _assert_attrs(factory_model, {
        "api_key": "config-xai-key",
        "base_url": "https://xai.example.com/v1",
    })


def test_deepseek_llm_profile_config_propagates(monkeypatch):
    """Same as the xai test but for the deepseek profile, to confirm the
    profile merge path in ``AIFactory.create_language`` is generic."""
    _scrub_provider_env(monkeypatch)
    from esperanto.providers.llm.openai_compatible import OpenAICompatibleLanguageModel

    config = {
        "api_key": "config-deepseek-key",
        "base_url": "https://deepseek.example.com/v1",
    }
    factory_model = AIFactory.create_language(
        "deepseek", "deepseek-chat", config=dict(config)
    )
    assert isinstance(factory_model, OpenAICompatibleLanguageModel)
    _assert_attrs(factory_model, {
        "api_key": "config-deepseek-key",
        "base_url": "https://deepseek.example.com/v1",
    })
