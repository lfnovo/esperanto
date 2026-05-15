"""Tests for base URL trailing slash normalization across providers (issue #149).

Every provider that composes endpoint URLs from ``self.base_url`` must strip
trailing slashes to avoid accidental double-slashes in the composed URL when a
user (or environment variable) supplies a ``base_url`` that ends with ``/``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_no_trailing_slash(base_url: str) -> None:
    """Assert that a base_url does not end with a slash."""
    assert base_url, "base_url must be set"
    assert not base_url.endswith("/"), f"base_url should not end with '/': {base_url!r}"


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------


def test_anthropic_llm_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from esperanto.providers.llm.anthropic import AnthropicLanguageModel

    model = AnthropicLanguageModel(base_url="https://api.anthropic.com/v1/")
    assert model.base_url == "https://api.anthropic.com/v1"


def test_anthropic_llm_base_url_default_has_no_trailing_slash(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from esperanto.providers.llm.anthropic import AnthropicLanguageModel

    model = AnthropicLanguageModel()
    _assert_no_trailing_slash(model.base_url)


def test_google_llm_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_BASE_URL", "https://gemini.example.com/")
    from esperanto.providers.llm.google import GoogleLanguageModel

    model = GoogleLanguageModel()
    assert model.base_url == "https://gemini.example.com/v1beta"


def test_google_llm_base_url_default_has_no_trailing_slash(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_API_BASE_URL", raising=False)
    from esperanto.providers.llm.google import GoogleLanguageModel

    model = GoogleLanguageModel()
    _assert_no_trailing_slash(model.base_url)


def test_groq_llm_base_url_default_has_no_trailing_slash(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    from esperanto.providers.llm.groq import GroqLanguageModel

    # Groq LLM hardcodes its base_url; this still asserts the invariant holds.
    model = GroqLanguageModel()
    _assert_no_trailing_slash(model.base_url)


def test_mistral_llm_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    from esperanto.providers.llm.mistral import MistralLanguageModel

    model = MistralLanguageModel(base_url="https://api.mistral.ai/v1/")
    assert model.base_url == "https://api.mistral.ai/v1"


def test_openai_llm_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from esperanto.providers.llm.openai import OpenAILanguageModel

    model = OpenAILanguageModel(base_url="https://api.openai.com/v1/")
    assert model.base_url == "https://api.openai.com/v1"


def test_openai_llm_base_url_multiple_trailing_slashes_stripped(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from esperanto.providers.llm.openai import OpenAILanguageModel

    model = OpenAILanguageModel(base_url="https://api.openai.com/v1///")
    assert model.base_url == "https://api.openai.com/v1"


def test_openrouter_llm_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    from esperanto.providers.llm.openrouter import OpenRouterLanguageModel

    model = OpenRouterLanguageModel(base_url="https://openrouter.ai/api/v1/")
    assert model.base_url == "https://openrouter.ai/api/v1"


def test_openrouter_llm_base_url_env_var_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://router.example.com/api/v1/")
    from esperanto.providers.llm.openrouter import OpenRouterLanguageModel

    model = OpenRouterLanguageModel()
    assert model.base_url == "https://router.example.com/api/v1"


def test_perplexity_llm_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
    monkeypatch.delenv("PERPLEXITY_BASE_URL", raising=False)
    from esperanto.providers.llm.perplexity import PerplexityLanguageModel

    model = PerplexityLanguageModel(base_url="https://api.perplexity.ai/")
    assert model.base_url == "https://api.perplexity.ai"


def test_vertex_llm_base_url_default_has_no_trailing_slash(monkeypatch):
    monkeypatch.setenv("VERTEX_PROJECT", "test-project")
    monkeypatch.setenv("VERTEX_LOCATION", "us-central1")
    # Mock google.auth.default to avoid real ADC pickup
    with patch("google.auth.default") as mock_default:
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.token = "mock-token"
        mock_default.return_value = (mock_creds, "test-project")
        from esperanto.providers.llm.vertex import VertexLanguageModel

        model = VertexLanguageModel(model_name="gemini-2.0-flash")
        _assert_no_trailing_slash(model.base_url)


# ---------------------------------------------------------------------------
# Embedding providers
# ---------------------------------------------------------------------------


def test_google_embedding_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_BASE_URL", "https://gemini.example.com/")
    from esperanto.providers.embedding.google import GoogleEmbeddingModel

    model = GoogleEmbeddingModel()
    assert model.base_url == "https://gemini.example.com/v1beta"


def test_jina_embedding_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "test-key")
    from esperanto.providers.embedding.jina import JinaEmbeddingModel

    model = JinaEmbeddingModel(base_url="https://api.jina.ai/v1/embeddings/")
    assert model.base_url == "https://api.jina.ai/v1/embeddings"


def test_mistral_embedding_base_url_default_has_no_trailing_slash(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    from esperanto.providers.embedding.mistral import MistralEmbeddingModel

    model = MistralEmbeddingModel()
    _assert_no_trailing_slash(model.base_url)


def test_ollama_embedding_base_url_trailing_slash_stripped():
    from esperanto.providers.embedding.ollama import OllamaEmbeddingModel

    model = OllamaEmbeddingModel(base_url="http://localhost:11434/")
    assert model.base_url == "http://localhost:11434"


def test_ollama_embedding_base_url_env_var_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://env:11434/")
    from esperanto.providers.embedding.ollama import OllamaEmbeddingModel

    model = OllamaEmbeddingModel()
    assert model.base_url == "http://env:11434"


def test_openai_embedding_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from esperanto.providers.embedding.openai import OpenAIEmbeddingModel

    model = OpenAIEmbeddingModel(base_url="https://api.openai.com/v1/")
    assert model.base_url == "https://api.openai.com/v1"


def test_openrouter_embedding_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    from esperanto.providers.embedding.openrouter import OpenRouterEmbeddingModel

    model = OpenRouterEmbeddingModel(base_url="https://openrouter.ai/api/v1/")
    assert model.base_url == "https://openrouter.ai/api/v1"


def test_vertex_embedding_base_url_default_has_no_trailing_slash(monkeypatch):
    monkeypatch.setenv("VERTEX_PROJECT", "test-project")
    monkeypatch.setenv("VERTEX_LOCATION", "us-central1")
    from esperanto.providers.embedding.vertex import VertexEmbeddingModel

    model = VertexEmbeddingModel()
    _assert_no_trailing_slash(model.base_url)


def test_voyage_embedding_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    from esperanto.providers.embedding.voyage import VoyageEmbeddingModel

    model = VoyageEmbeddingModel(base_url="https://api.voyageai.com/v1/")
    assert model.base_url == "https://api.voyageai.com/v1"


# ---------------------------------------------------------------------------
# Reranker providers
# ---------------------------------------------------------------------------


def test_jina_reranker_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "test-key")
    from esperanto.providers.reranker.jina import JinaRerankerModel

    model = JinaRerankerModel(base_url="https://api.jina.ai/v1/")
    assert model.base_url == "https://api.jina.ai/v1"


def test_voyage_reranker_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    from esperanto.providers.reranker.voyage import VoyageRerankerModel

    model = VoyageRerankerModel(base_url="https://api.voyageai.com/v1/")
    assert model.base_url == "https://api.voyageai.com/v1"


# ---------------------------------------------------------------------------
# Speech-to-Text providers
# ---------------------------------------------------------------------------


def test_elevenlabs_stt_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    from esperanto.providers.stt.elevenlabs import ElevenLabsSpeechToTextModel

    model = ElevenLabsSpeechToTextModel(base_url="https://api.elevenlabs.io/v1/")
    assert model.base_url == "https://api.elevenlabs.io/v1"


def test_google_stt_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_BASE_URL", "https://gemini.example.com/")
    from esperanto.providers.stt.google import GoogleSpeechToTextModel

    model = GoogleSpeechToTextModel()
    assert model.base_url == "https://gemini.example.com/v1beta"


def test_groq_stt_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    from esperanto.providers.stt.groq import GroqSpeechToTextModel

    model = GroqSpeechToTextModel(base_url="https://api.groq.com/openai/v1/")
    assert model.base_url == "https://api.groq.com/openai/v1"


def test_mistral_stt_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    from esperanto.providers.stt.mistral import MistralSpeechToTextModel

    model = MistralSpeechToTextModel(base_url="https://api.mistral.ai/v1/")
    assert model.base_url == "https://api.mistral.ai/v1"


def test_openai_stt_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from esperanto.providers.stt.openai import OpenAISpeechToTextModel

    model = OpenAISpeechToTextModel(base_url="https://api.openai.com/v1/")
    assert model.base_url == "https://api.openai.com/v1"


# ---------------------------------------------------------------------------
# Text-to-Speech providers
# ---------------------------------------------------------------------------


def test_elevenlabs_tts_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    from esperanto.providers.tts.elevenlabs import ElevenLabsTextToSpeechModel

    model = ElevenLabsTextToSpeechModel(base_url="https://api.elevenlabs.io/")
    assert model.base_url == "https://api.elevenlabs.io"


def test_google_tts_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_BASE_URL", "https://gemini.example.com/")
    from esperanto.providers.tts.google import GoogleTextToSpeechModel

    model = GoogleTextToSpeechModel()
    assert model.base_url == "https://gemini.example.com/v1beta"


def test_mistral_tts_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    from esperanto.providers.tts.mistral import MistralTextToSpeechModel

    model = MistralTextToSpeechModel(base_url="https://api.mistral.ai/v1/")
    assert model.base_url == "https://api.mistral.ai/v1"


def test_openai_tts_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from esperanto.providers.tts.openai import OpenAITextToSpeechModel

    model = OpenAITextToSpeechModel(base_url="https://api.openai.com/v1/")
    assert model.base_url == "https://api.openai.com/v1"


def test_vertex_tts_base_url_default_has_no_trailing_slash(monkeypatch):
    monkeypatch.setenv("VERTEX_PROJECT", "test-project")
    from esperanto.providers.tts.vertex import VertexTextToSpeechModel

    model = VertexTextToSpeechModel()
    _assert_no_trailing_slash(model.base_url)


def test_xai_tts_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.delenv("XAI_BASE_URL", raising=False)
    from esperanto.providers.tts.xai import XAITextToSpeechModel

    model = XAITextToSpeechModel(base_url="https://api.x.ai/")
    assert model.base_url == "https://api.x.ai"


def test_xai_tts_base_url_env_var_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setenv("XAI_BASE_URL", "https://xai.example.com/v1/")
    from esperanto.providers.tts.xai import XAITextToSpeechModel

    model = XAITextToSpeechModel()
    assert model.base_url == "https://xai.example.com/v1"


def _azure_env(monkeypatch, modality_suffix: str) -> None:
    """Set common Azure env vars with trailing slash on the endpoint."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv(
        "AZURE_OPENAI_ENDPOINT",
        "https://test-endpoint.openai.azure.com/",
    )
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", f"2024-{modality_suffix}-01")


def test_azure_llm_endpoint_trailing_slash_stripped(monkeypatch):
    _azure_env(monkeypatch, "01")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-01-01")
    from esperanto.providers.llm.azure import AzureLanguageModel

    model = AzureLanguageModel(model_name="test-deployment")
    assert model.azure_endpoint == "https://test-endpoint.openai.azure.com"


def test_azure_llm_endpoint_via_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-01-01")
    from esperanto.providers.llm.azure import AzureLanguageModel

    model = AzureLanguageModel(
        model_name="test-deployment",
        base_url="https://direct-endpoint.com/",
    )
    assert model.azure_endpoint == "https://direct-endpoint.com"


def test_azure_embedding_endpoint_trailing_slash_stripped(monkeypatch):
    _azure_env(monkeypatch, "02")
    from esperanto.providers.embedding.azure import AzureEmbeddingModel

    model = AzureEmbeddingModel(model_name="test-deployment")
    assert model.azure_endpoint == "https://test-endpoint.openai.azure.com"


def test_azure_stt_endpoint_trailing_slash_stripped(monkeypatch):
    _azure_env(monkeypatch, "03")
    from esperanto.providers.stt.azure import AzureSpeechToTextModel

    model = AzureSpeechToTextModel(model_name="test-deployment")
    assert model.azure_endpoint == "https://test-endpoint.openai.azure.com"


def test_azure_tts_endpoint_trailing_slash_stripped(monkeypatch):
    _azure_env(monkeypatch, "04")
    from esperanto.providers.tts.azure import AzureTextToSpeechModel

    model = AzureTextToSpeechModel(model_name="test-deployment")
    assert model.azure_endpoint == "https://test-endpoint.openai.azure.com"
