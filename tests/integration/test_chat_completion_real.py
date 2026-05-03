"""Real integration tests for chat completion - these call actual APIs.

These tests verify that basic chat completion (sync/async, streaming/non-streaming)
works correctly with real API calls across all LLM providers.

Run with: uv run pytest tests/integration/test_chat_completion_real.py -v -s
"""

import os

import pytest

from esperanto import AIFactory
from esperanto.common_types import ChatCompletion, ChatCompletionChunk


def _ollama_available() -> bool:
    """Probe for a reachable Ollama instance.

    Ollama defaults to ``http://localhost:11434`` per its own provider source,
    so the test should run whenever Ollama is reachable — locally OR via the
    optional ``OLLAMA_BASE_URL`` / ``OLLAMA_API_BASE`` env override. Avoids
    skipping tests when the user has Ollama running locally without setting
    an env var.
    """
    import httpx
    base_url = (
        os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_API_BASE")
        or "http://localhost:11434"
    )
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


MESSAGES = [{"role": "user", "content": "Say hello in one sentence."}]


# =============================================================================
# OpenAI Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not configured",
)
class TestOpenAIChat:
    """Real integration tests for OpenAI chat completion."""

    def test_sync_chat_complete(self):
        model = AIFactory.create_language("openai", "gpt-4o-mini")
        response = model.chat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    async def test_async_chat_complete(self):
        model = AIFactory.create_language("openai", "gpt-4o-mini")
        response = await model.achat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_sync_streaming(self):
        model = AIFactory.create_language("openai", "gpt-4o-mini")
        response = model.chat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0

    async def test_async_streaming(self):
        model = AIFactory.create_language("openai", "gpt-4o-mini")
        response = await model.achat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        async for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0


# =============================================================================
# Anthropic Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not configured",
)
class TestAnthropicChat:
    """Real integration tests for Anthropic chat completion."""

    def test_sync_chat_complete(self):
        model = AIFactory.create_language("anthropic", "claude-3-5-haiku-latest")
        response = model.chat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    async def test_async_chat_complete(self):
        model = AIFactory.create_language("anthropic", "claude-3-5-haiku-latest")
        response = await model.achat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_sync_streaming(self):
        model = AIFactory.create_language("anthropic", "claude-3-5-haiku-latest")
        response = model.chat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0

    async def test_async_streaming(self):
        model = AIFactory.create_language("anthropic", "claude-3-5-haiku-latest")
        response = await model.achat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        async for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0


# =============================================================================
# Google Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="GOOGLE_API_KEY or GEMINI_API_KEY not configured",
)
class TestGoogleChat:
    """Real integration tests for Google chat completion."""

    def test_sync_chat_complete(self):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        model = AIFactory.create_language("google", "gemini-2.0-flash", config={"api_key": api_key})
        response = model.chat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    async def test_async_chat_complete(self):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        model = AIFactory.create_language("google", "gemini-2.0-flash", config={"api_key": api_key})
        response = await model.achat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_sync_streaming(self):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        model = AIFactory.create_language("google", "gemini-2.0-flash", config={"api_key": api_key})
        response = model.chat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0

    async def test_async_streaming(self):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        model = AIFactory.create_language("google", "gemini-2.0-flash", config={"api_key": api_key})
        response = await model.achat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        async for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0


# =============================================================================
# Vertex AI Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skip(
    reason="Vertex AI requires ADC or GOOGLE_APPLICATION_CREDENTIALS — not a simple API-key env var; omitted in tool-calling tests for the same reason"
)
class TestVertexChat:
    """Real integration tests for Vertex AI chat completion.

    NOTE: Vertex AI requires Google Application Default Credentials (ADC) or
    GOOGLE_APPLICATION_CREDENTIALS, not a simple API key. These tests are skipped.
    """

    def test_sync_chat_complete(self):
        model = AIFactory.create_language("vertex", "gemini-2.0-flash")
        response = model.chat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    async def test_async_chat_complete(self):
        model = AIFactory.create_language("vertex", "gemini-2.0-flash")
        response = await model.achat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_sync_streaming(self):
        model = AIFactory.create_language("vertex", "gemini-2.0-flash")
        response = model.chat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0

    async def test_async_streaming(self):
        model = AIFactory.create_language("vertex", "gemini-2.0-flash")
        response = await model.achat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        async for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0


# =============================================================================
# Azure Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not (
        (os.getenv("AZURE_OPENAI_API_KEY_LLM") or os.getenv("AZURE_OPENAI_API_KEY"))
        and (os.getenv("AZURE_OPENAI_ENDPOINT_LLM") or os.getenv("AZURE_OPENAI_ENDPOINT"))
        and (
            os.getenv("AZURE_OPENAI_API_VERSION_LLM")
            or os.getenv("OPENAI_API_VERSION")
            or os.getenv("AZURE_OPENAI_API_VERSION")
        )
    ),
    reason="Azure LLM requires API key, endpoint, and API version (AZURE_OPENAI_API_KEY[_LLM] + AZURE_OPENAI_ENDPOINT[_LLM] + AZURE_OPENAI_API_VERSION[_LLM])",
)
class TestAzureChat:
    """Real integration tests for Azure OpenAI chat completion."""

    def test_sync_chat_complete(self):
        model = AIFactory.create_language(
            "azure",
            os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_LLM", "gpt-4o-mini"),
            config={
                "api_key": os.getenv("AZURE_OPENAI_API_KEY_LLM"),
                "base_url": os.getenv("AZURE_OPENAI_ENDPOINT_LLM"),
                "api_version": os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
            },
        )
        response = model.chat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    async def test_async_chat_complete(self):
        model = AIFactory.create_language(
            "azure",
            os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_LLM", "gpt-4o-mini"),
            config={
                "api_key": os.getenv("AZURE_OPENAI_API_KEY_LLM"),
                "base_url": os.getenv("AZURE_OPENAI_ENDPOINT_LLM"),
                "api_version": os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
            },
        )
        response = await model.achat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_sync_streaming(self):
        model = AIFactory.create_language(
            "azure",
            os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_LLM", "gpt-4o-mini"),
            config={
                "api_key": os.getenv("AZURE_OPENAI_API_KEY_LLM"),
                "base_url": os.getenv("AZURE_OPENAI_ENDPOINT_LLM"),
                "api_version": os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
            },
        )
        response = model.chat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0

    async def test_async_streaming(self):
        model = AIFactory.create_language(
            "azure",
            os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_LLM", "gpt-4o-mini"),
            config={
                "api_key": os.getenv("AZURE_OPENAI_API_KEY_LLM"),
                "base_url": os.getenv("AZURE_OPENAI_ENDPOINT_LLM"),
                "api_version": os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
            },
        )
        response = await model.achat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        async for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0


# =============================================================================
# Mistral Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY not configured",
)
class TestMistralChat:
    """Real integration tests for Mistral chat completion."""

    def test_sync_chat_complete(self):
        model = AIFactory.create_language("mistral", "mistral-small-latest")
        response = model.chat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    async def test_async_chat_complete(self):
        model = AIFactory.create_language("mistral", "mistral-small-latest")
        response = await model.achat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_sync_streaming(self):
        model = AIFactory.create_language("mistral", "mistral-small-latest")
        response = model.chat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0

    async def test_async_streaming(self):
        model = AIFactory.create_language("mistral", "mistral-small-latest")
        response = await model.achat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        async for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0


# =============================================================================
# Ollama Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not reachable at configured base URL or localhost:11434",
)
class TestOllamaChat:
    """Real integration tests for Ollama chat completion."""

    def test_sync_chat_complete(self):
        model = AIFactory.create_language(
            "ollama", "qwen3:32b", config={"base_url": os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_BASE")}
        )
        response = model.chat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    async def test_async_chat_complete(self):
        model = AIFactory.create_language(
            "ollama", "qwen3:32b", config={"base_url": os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_BASE")}
        )
        response = await model.achat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_sync_streaming(self):
        model = AIFactory.create_language(
            "ollama", "qwen3:32b", config={"base_url": os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_BASE")}
        )
        response = model.chat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0

    async def test_async_streaming(self):
        model = AIFactory.create_language(
            "ollama", "qwen3:32b", config={"base_url": os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_BASE")}
        )
        response = await model.achat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        async for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0


# =============================================================================
# Groq Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not configured",
)
class TestGroqChat:
    """Real integration tests for Groq chat completion."""

    def test_sync_chat_complete(self):
        model = AIFactory.create_language("groq", "llama-3.3-70b-versatile")
        response = model.chat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    async def test_async_chat_complete(self):
        model = AIFactory.create_language("groq", "llama-3.3-70b-versatile")
        response = await model.achat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_sync_streaming(self):
        model = AIFactory.create_language("groq", "llama-3.3-70b-versatile")
        response = model.chat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0

    async def test_async_streaming(self):
        model = AIFactory.create_language("groq", "llama-3.3-70b-versatile")
        response = await model.achat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        async for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0


# =============================================================================
# OpenRouter Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not configured",
)
class TestOpenRouterChat:
    """Real integration tests for OpenRouter chat completion."""

    def test_sync_chat_complete(self):
        model = AIFactory.create_language("openrouter", "openai/gpt-4o-mini")
        response = model.chat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    async def test_async_chat_complete(self):
        model = AIFactory.create_language("openrouter", "openai/gpt-4o-mini")
        response = await model.achat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_sync_streaming(self):
        model = AIFactory.create_language("openrouter", "openai/gpt-4o-mini")
        response = model.chat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0

    async def test_async_streaming(self):
        model = AIFactory.create_language("openrouter", "openai/gpt-4o-mini")
        response = await model.achat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        async for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0


# =============================================================================
# Perplexity Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("PERPLEXITY_API_KEY"),
    reason="PERPLEXITY_API_KEY not configured",
)
class TestPerplexityChat:
    """Real integration tests for Perplexity chat completion."""

    def test_sync_chat_complete(self):
        model = AIFactory.create_language("perplexity", "sonar")
        response = model.chat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    async def test_async_chat_complete(self):
        model = AIFactory.create_language("perplexity", "sonar")
        response = await model.achat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_sync_streaming(self):
        model = AIFactory.create_language("perplexity", "sonar")
        response = model.chat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0

    async def test_async_streaming(self):
        model = AIFactory.create_language("perplexity", "sonar")
        response = await model.achat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        async for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0


# =============================================================================
# DeepSeek Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY not configured",
)
class TestDeepSeekChat:
    """Real integration tests for DeepSeek chat completion."""

    def test_sync_chat_complete(self):
        model = AIFactory.create_language("deepseek", "deepseek-chat")
        response = model.chat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    async def test_async_chat_complete(self):
        model = AIFactory.create_language("deepseek", "deepseek-chat")
        response = await model.achat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_sync_streaming(self):
        model = AIFactory.create_language("deepseek", "deepseek-chat")
        response = model.chat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0

    async def test_async_streaming(self):
        model = AIFactory.create_language("deepseek", "deepseek-chat")
        response = await model.achat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        async for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0


# =============================================================================
# xAI Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("XAI_API_KEY"),
    reason="XAI_API_KEY not configured",
)
class TestXAIChat:
    """Real integration tests for xAI chat completion."""

    def test_sync_chat_complete(self):
        model = AIFactory.create_language("xai", "grok-3")
        response = model.chat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    async def test_async_chat_complete(self):
        model = AIFactory.create_language("xai", "grok-3")
        response = await model.achat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_sync_streaming(self):
        model = AIFactory.create_language("xai", "grok-3")
        response = model.chat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0

    async def test_async_streaming(self):
        model = AIFactory.create_language("xai", "grok-3")
        response = await model.achat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        async for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0


# =============================================================================
# DashScope Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("DASHSCOPE_API_KEY"),
    reason="DASHSCOPE_API_KEY not configured",
)
class TestDashScopeChat:
    """Real integration tests for DashScope chat completion."""

    def test_sync_chat_complete(self):
        model = AIFactory.create_language("dashscope", "qwen-plus")
        response = model.chat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    async def test_async_chat_complete(self):
        model = AIFactory.create_language("dashscope", "qwen-plus")
        response = await model.achat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_sync_streaming(self):
        model = AIFactory.create_language("dashscope", "qwen-plus")
        response = model.chat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0

    async def test_async_streaming(self):
        model = AIFactory.create_language("dashscope", "qwen-plus")
        response = await model.achat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        async for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0


# =============================================================================
# MiniMax Tests
# =============================================================================


@pytest.mark.release
@pytest.mark.skipif(
    not os.getenv("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not configured",
)
class TestMiniMaxChat:
    """Real integration tests for MiniMax chat completion."""

    def test_sync_chat_complete(self):
        model = AIFactory.create_language("minimax", "MiniMax-M2.5")
        response = model.chat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    async def test_async_chat_complete(self):
        model = AIFactory.create_language("minimax", "MiniMax-M2.5")
        response = await model.achat_complete(messages=MESSAGES)
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_sync_streaming(self):
        model = AIFactory.create_language("minimax", "MiniMax-M2.5")
        response = model.chat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0

    async def test_async_streaming(self):
        model = AIFactory.create_language("minimax", "MiniMax-M2.5")
        response = await model.achat_complete(messages=MESSAGES, stream=True)
        total_content = ""
        async for chunk in response:
            assert isinstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
        assert len(total_content) > 0
