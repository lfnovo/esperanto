"""Tests for the OpenAI-compatible provider profiles system."""

import os
import warnings
from unittest.mock import Mock, patch

import pytest

from esperanto.factory import AIFactory
from esperanto.providers.llm.profiles import (
    _USER_PROFILES,
    BUILTIN_PROFILES,
    OpenAICompatibleProfile,
    get_all_profile_names,
    get_profile,
    register_profile,
)


@pytest.fixture(autouse=True)
def _clean_user_profiles():
    """Ensure user profiles are clean for each test."""
    _USER_PROFILES.clear()
    yield
    _USER_PROFILES.clear()


# =============================================================================
# Profile dataclass tests
# =============================================================================


class TestOpenAICompatibleProfile:
    def test_profile_is_frozen(self):
        profile = OpenAICompatibleProfile(
            name="test", base_url="http://localhost", api_key_env="TEST_KEY", default_model="m"
        )
        with pytest.raises(AttributeError):
            profile.name = "changed"

    def test_defaults(self):
        profile = OpenAICompatibleProfile(
            name="test", base_url="http://localhost", api_key_env="TEST_KEY", default_model="m"
        )
        assert profile.base_url_env is None
        assert profile.supports_response_format is True
        assert profile.model_prefix_filter is None
        assert profile.owned_by is None
        assert profile.display_name is None


# =============================================================================
# Profile registry tests
# =============================================================================


class TestProfileRegistry:
    def test_builtin_profiles_exist(self):
        assert "deepseek" in BUILTIN_PROFILES
        assert "xai" in BUILTIN_PROFILES
        assert "novita" in BUILTIN_PROFILES

    def test_get_builtin_profile(self):
        profile = get_profile("deepseek")
        assert profile is not None
        assert profile.name == "deepseek"
        assert profile.base_url == "https://api.deepseek.com/v1"
        assert profile.api_key_env == "DEEPSEEK_API_KEY"
        assert profile.default_model == "deepseek-chat"

    def test_get_xai_profile(self):
        profile = get_profile("xai")
        assert profile is not None
        assert profile.supports_response_format is False
        assert profile.model_prefix_filter == "grok"

    def test_get_dashscope_profile(self):
        profile = get_profile("dashscope")
        assert profile is not None
        assert profile.name == "dashscope"
        assert profile.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        assert profile.api_key_env == "DASHSCOPE_API_KEY"
        assert profile.default_model == "qwen-plus"

    def test_get_novita_profile(self):
        profile = get_profile("novita")
        assert profile is not None
        assert profile.name == "novita"
        assert profile.base_url == "https://api.novita.ai/openai"
        assert profile.api_key_env == "NOVITA_API_KEY"
        assert profile.default_model == "moonshotai/kimi-k2.5"

    def test_get_unknown_profile_returns_none(self):
        assert get_profile("unknown-provider") is None

    def test_register_user_profile(self):
        profile = OpenAICompatibleProfile(
            name="together",
            base_url="https://api.together.xyz/v1",
            api_key_env="TOGETHER_API_KEY",
            default_model="meta-llama/Llama-3-70b-chat-hf",
        )
        register_profile(profile)
        assert get_profile("together") is profile

    def test_user_profile_overrides_builtin(self):
        custom = OpenAICompatibleProfile(
            name="deepseek",
            base_url="https://custom.deepseek.com/v1",
            api_key_env="CUSTOM_DEEPSEEK_KEY",
            default_model="deepseek-v3",
        )
        register_profile(custom)
        assert get_profile("deepseek") is custom
        assert get_profile("deepseek").base_url == "https://custom.deepseek.com/v1"

    def test_register_invalid_type_raises(self):
        with pytest.raises(TypeError):
            register_profile({"name": "bad"})

    def test_register_empty_name_raises(self):
        with pytest.raises(ValueError):
            register_profile(
                OpenAICompatibleProfile(
                    name="", base_url="http://x", api_key_env="X", default_model="m"
                )
            )

    def test_get_all_profile_names(self):
        names = get_all_profile_names()
        assert "deepseek" in names
        assert "xai" in names
        assert "novita" in names

    def test_get_all_profile_names_includes_user(self):
        register_profile(
            OpenAICompatibleProfile(
                name="custom", base_url="http://x", api_key_env="X", default_model="m"
            )
        )
        assert "custom" in get_all_profile_names()


# =============================================================================
# Factory integration tests
# =============================================================================


class TestFactoryIntegration:
    def test_create_language_with_deepseek_profile(self):
        model = AIFactory.create_language(
            "deepseek", "deepseek-chat", config={"api_key": "test-key"}
        )
        assert model.provider == "deepseek"
        assert model.base_url == "https://api.deepseek.com/v1"
        assert model.api_key == "test-key"

    def test_create_language_with_xai_profile(self):
        model = AIFactory.create_language(
            "xai", "grok-2-latest", config={"api_key": "test-key"}
        )
        assert model.provider == "xai"
        assert model.base_url == "https://api.x.ai/v1"
        assert model._response_format_unsupported is True

    def test_create_language_with_novita_profile(self):
        model = AIFactory.create_language(
            "novita", "moonshotai/kimi-k2.5", config={"api_key": "test-key"}
        )
        assert model.provider == "novita"
        assert model.base_url == "https://api.novita.ai/openai"
        assert model.api_key == "test-key"

    def test_create_language_with_user_profile(self):
        AIFactory.register_openai_compatible_profile(
            OpenAICompatibleProfile(
                name="together",
                base_url="https://api.together.xyz/v1",
                api_key_env="TOGETHER_API_KEY",
                default_model="meta-llama/Llama-3-70b-chat-hf",
            )
        )
        model = AIFactory.create_language(
            "together", "meta-llama/Llama-3-70b-chat-hf", config={"api_key": "test"}
        )
        assert model.provider == "together"
        assert model.base_url == "https://api.together.xyz/v1"

    def test_unknown_provider_falls_through(self):
        """Non-profile providers still use _provider_modules."""
        model = AIFactory.create_language(
            "openai", "gpt-4", config={"api_key": "test-key"}
        )
        assert model.provider == "openai"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            AIFactory.create_language("nonexistent", "model")

    def test_get_available_providers_includes_profiles(self):
        providers = AIFactory.get_available_providers()
        assert "deepseek" in providers["language"]
        assert "xai" in providers["language"]
        assert "novita" in providers["language"]
        # Class-based providers also present
        assert "openai" in providers["language"]

    def test_get_available_providers_includes_user_profiles(self):
        AIFactory.register_openai_compatible_profile(
            OpenAICompatibleProfile(
                name="fireworks",
                base_url="https://api.fireworks.ai/v1",
                api_key_env="FIREWORKS_API_KEY",
                default_model="llama-v3-70b",
            )
        )
        providers = AIFactory.get_available_providers()
        assert "fireworks" in providers["language"]


# =============================================================================
# Profile-based provider behavior tests
# =============================================================================


class TestProfileBehavior:
    def test_deepseek_default_model(self):
        model = AIFactory.create_language(
            "deepseek", "deepseek-chat", config={"api_key": "test-key"}
        )
        assert model._get_default_model() == "deepseek-chat"

    def test_xai_default_model(self):
        model = AIFactory.create_language(
            "xai", "grok-2-latest", config={"api_key": "test-key"}
        )
        assert model._get_default_model() == "grok-2-latest"

    def test_deepseek_env_var_api_key(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-key"}, clear=False):
            model = AIFactory.create_language("deepseek", "deepseek-chat")
            assert model.api_key == "env-key"

    def test_xai_env_var_api_key(self):
        with patch.dict(os.environ, {"XAI_API_KEY": "env-key"}, clear=False):
            model = AIFactory.create_language("xai", "grok-2-latest")
            assert model.api_key == "env-key"

    def test_deepseek_env_var_base_url(self):
        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "key", "DEEPSEEK_BASE_URL": "https://custom.deepseek.com"},
            clear=False,
        ):
            model = AIFactory.create_language("deepseek", "deepseek-chat")
            assert model.base_url == "https://custom.deepseek.com"

    def test_config_overrides_profile_defaults(self):
        model = AIFactory.create_language(
            "deepseek",
            "deepseek-chat",
            config={
                "api_key": "test-key",
                "base_url": "https://override.deepseek.com/v1",
            },
        )
        assert model.base_url == "https://override.deepseek.com/v1"

    def test_missing_api_key_raises_with_provider_name(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="DeepSeek API key not found"):
                AIFactory.create_language("deepseek", "deepseek-chat")

    def test_missing_xai_api_key_raises_with_provider_name(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="XAI API key not found"):
                AIFactory.create_language("xai", "grok-2-latest")

    def test_dashscope_creation(self):
        model = AIFactory.create_language(
            "dashscope", "qwen-plus", config={"api_key": "test-key"}
        )
        assert model.provider == "dashscope"
        assert model.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        assert model._get_default_model() == "qwen-plus"

    def test_dashscope_env_var(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env-key"}, clear=False):
            model = AIFactory.create_language("dashscope", "qwen-max")
            assert model.api_key == "env-key"

    def test_dashscope_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="DashScope API key not found"):
                AIFactory.create_language("dashscope", "qwen-plus")

    def test_novita_creation(self):
        model = AIFactory.create_language(
            "novita", "moonshotai/kimi-k2.5", config={"api_key": "test-key"}
        )
        assert model.provider == "novita"
        assert model.base_url == "https://api.novita.ai/openai"
        assert model._get_default_model() == "moonshotai/kimi-k2.5"

    def test_novita_env_var(self):
        with patch.dict(os.environ, {"NOVITA_API_KEY": "env-key"}, clear=False):
            model = AIFactory.create_language("novita", "moonshotai/kimi-k2.5")
            assert model.api_key == "env-key"

    def test_novita_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Novita API key not found"):
                AIFactory.create_language("novita", "moonshotai/kimi-k2.5")

    def test_minimax_creation(self):
        model = AIFactory.create_language(
            "minimax", "MiniMax-M2.5", config={"api_key": "test-key"}
        )
        assert model.provider == "minimax"
        assert model.base_url == "https://api.minimax.io/v1"
        assert model._get_default_model() == "MiniMax-M2.5"

    def test_minimax_env_var(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}, clear=False):
            model = AIFactory.create_language("minimax", "MiniMax-M2.5")
            assert model.api_key == "env-key"

    def test_minimax_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="MiniMax API key not found"):
                AIFactory.create_language("minimax", "MiniMax-M2.5")

    def test_xai_response_format_stripped(self):
        model = AIFactory.create_language(
            "xai",
            "grok-2-latest",
            config={"api_key": "test-key", "structured": {"type": "json_object"}},
        )
        kwargs = model._get_api_kwargs()
        assert "response_format" not in kwargs

    def test_model_prefix_filter(self):
        """xAI profile filters models to grok-* prefix."""
        model = AIFactory.create_language(
            "xai", "grok-2-latest", config={"api_key": "test-key"}
        )
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "grok-2-latest", "owned_by": "xai"},
                {"id": "grok-beta", "owned_by": "xai"},
                {"id": "some-other-model", "owned_by": "xai"},
            ]
        }
        mock_client = Mock()
        mock_client.get.return_value = mock_response
        model.client = mock_client

        models = model._get_models()
        assert len(models) == 2
        assert all(m.id.startswith("grok") for m in models)
        assert all(m.owned_by == "X.AI" for m in models)


# =============================================================================
# Deprecated wrapper tests
# =============================================================================


class TestDeprecatedWrappers:
    def test_deepseek_direct_import_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from esperanto.providers.llm.deepseek import DeepSeekLanguageModel

            DeepSeekLanguageModel(api_key="test-key")
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "deprecated" in str(deprecation_warnings[0].message).lower()

    def test_xai_direct_import_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from esperanto.providers.llm.xai import XAILanguageModel

            XAILanguageModel(api_key="test-key")
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "deprecated" in str(deprecation_warnings[0].message).lower()

    def test_deprecated_deepseek_still_works(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from esperanto.providers.llm.deepseek import DeepSeekLanguageModel

            model = DeepSeekLanguageModel(api_key="test-key")
            assert model.provider == "deepseek"
            assert model.base_url == "https://api.deepseek.com/v1"

    def test_deprecated_xai_still_works(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from esperanto.providers.llm.xai import XAILanguageModel

            model = XAILanguageModel(api_key="test-key")
            assert model.provider == "xai"
            assert model.base_url == "https://api.x.ai/v1"
