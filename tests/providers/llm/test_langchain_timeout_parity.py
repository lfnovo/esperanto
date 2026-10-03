"""to_langchain() carries Esperanto's resolved timeout into the LangChain model (#305).

Esperanto resolves the timeout as config["timeout"], then ESPERANTO_LLM_TIMEOUT,
then 60 seconds. Each provider's converted model must use the same value.
"""

import math
from unittest.mock import MagicMock, Mock, patch

import pytest

from esperanto import AIFactory


def _anthropic(chat):
    return chat.default_request_timeout


def _google(chat):
    return chat.timeout


def _mistral(chat):
    return chat.timeout


def _cohere(chat):
    return chat.timeout_seconds


def _openrouter(chat):
    return chat.http_client.timeout.read


PROVIDERS = [
    pytest.param("anthropic", "claude-sonnet-5", _anthropic, id="anthropic"),
    pytest.param("google", "gemini-2.5-flash", _google, id="google"),
    pytest.param("vertex", "gemini-2.5-flash", _google, id="vertex"),
    pytest.param("mistral", "mistral-small-latest", _mistral, id="mistral"),
    pytest.param("cohere", "command-a-03-2025", _cohere, id="cohere"),
    pytest.param("openrouter", "anthropic/claude-sonnet-5.5", _openrouter, id="openrouter"),
]

PRECEDENCE = [
    pytest.param(None, {}, 60.0, id="default"),
    pytest.param("90", {}, 90.0, id="env"),
    pytest.param("90", {"timeout": 12.5}, 12.5, id="config-over-env"),
]


@pytest.fixture
def provider_env(monkeypatch):
    for name in ("ESPERANTO_LLM_TIMEOUT", "ESPERANTO_SSL_VERIFY", "ESPERANTO_SSL_CA_BUNDLE"):
        monkeypatch.delenv(name, raising=False)
    for name in (
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "MISTRAL_API_KEY",
        "COHERE_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.setenv(name, "test-key")
    monkeypatch.setenv("VERTEX_PROJECT", "test-project")

    creds = MagicMock()
    creds.valid = True
    creds.token = "mock-adc-token"
    with patch("google.auth.default", return_value=(creds, "test-project")), patch(
        "subprocess.run", return_value=Mock(stdout="mock-access-token")
    ):
        yield monkeypatch


@pytest.mark.parametrize("env_timeout,config,expected", PRECEDENCE)
@pytest.mark.parametrize("provider,model_name,read_timeout", PROVIDERS)
def test_to_langchain_uses_resolved_timeout(
    provider_env, provider, model_name, read_timeout, env_timeout, config, expected
):
    if env_timeout is not None:
        provider_env.setenv("ESPERANTO_LLM_TIMEOUT", env_timeout)

    model = AIFactory.create_language(provider, model_name, config=dict(config))
    chat = model.to_langchain()

    assert model._get_timeout() == expected
    if provider == "mistral":
        # ChatMistralAI takes whole seconds; Esperanto rounds up.
        assert read_timeout(chat) == math.ceil(expected)
    else:
        assert read_timeout(chat) == expected


def test_mistral_rounds_fractional_timeout_up(provider_env):
    model = AIFactory.create_language(
        "mistral", "mistral-small-latest", config={"timeout": 0.4}
    )
    assert model.to_langchain().timeout == 1


def test_openrouter_langchain_clients_carry_timeout(provider_env):
    model = AIFactory.create_language(
        "openrouter", "anthropic/claude-sonnet-5.5", config={"timeout": 12.5}
    )
    chat = model.to_langchain()

    assert chat.http_client.timeout.read == 12.5
    assert chat.http_async_client.timeout.read == 12.5
