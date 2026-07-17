"""Multi-modality OpenAI-compatible profile tests (issue #230).

Profiles used to resolve only for ``create_language()``. These tests cover the
extension to embedding / STT / TTS: opt-in capabilities, per-modality defaults,
the profile-vs-class precedence (including the live xAI hybrid), the typed
capability error, and the "no default beats a wrong default" rule.
"""

import warnings

import pytest

from esperanto.common_types import EsperantoError, ProviderCapabilityError
from esperanto.factory import AIFactory
from esperanto.providers.llm.profiles import (
    _USER_PROFILES,
    OpenAICompatibleProfile,
    get_profile_capabilities,
)


@pytest.fixture(autouse=True)
def _clean_user_profiles():
    _USER_PROFILES.clear()
    yield
    _USER_PROFILES.clear()


@pytest.fixture
def _env(monkeypatch):
    """Provide API keys the profiles below resolve from, without touching host env."""
    monkeypatch.setenv("MULTI_KEY", "k")
    monkeypatch.setenv("NODEF_KEY", "k")
    return monkeypatch


def _register(**kwargs) -> OpenAICompatibleProfile:
    profile = OpenAICompatibleProfile(**kwargs)
    AIFactory.register_openai_compatible_profile(profile)
    return profile


# =============================================================================
# Profile schema
# =============================================================================


class TestProfileSchema:
    def test_capabilities_default_is_language_only(self):
        p = OpenAICompatibleProfile(
            name="x", base_url="http://x/v1", api_key_env="X_KEY"
        )
        assert p.capabilities == {"language"}

    def test_default_models_default_empty(self):
        p = OpenAICompatibleProfile(
            name="x", base_url="http://x/v1", api_key_env="X_KEY"
        )
        assert p.default_models == {}
        assert p.default_model_for("language") is None

    def test_default_model_for_returns_declared(self):
        p = OpenAICompatibleProfile(
            name="x",
            base_url="http://x/v1",
            api_key_env="X_KEY",
            capabilities={"language", "embedding"},
            default_models={"language": "a", "embedding": "b"},
        )
        assert p.default_model_for("language") == "a"
        assert p.default_model_for("embedding") == "b"
        assert p.default_model_for("speech_to_text") is None

    def test_deprecated_default_model_folds_into_language(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            p = OpenAICompatibleProfile(
                name="x", base_url="http://x/v1", api_key_env="X_KEY", default_model="m"
            )
        assert p.default_model_for("language") == "m"
        assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_explicit_default_models_wins_over_deprecated_alias(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = OpenAICompatibleProfile(
                name="x",
                base_url="http://x/v1",
                api_key_env="X_KEY",
                default_model="deprecated",
                default_models={"language": "explicit"},
            )
        assert p.default_model_for("language") == "explicit"

    def test_builtins_do_not_warn_and_resolve_language_default(self, recwarn):
        # Re-importing built-ins must not emit deprecation noise; they use
        # default_models now, not default_model.
        from esperanto.providers.llm.profiles import get_profile

        p = get_profile("deepseek")
        assert p.default_model_for("language") == "deepseek-chat"
        assert p.default_model is None
        assert not [x for x in recwarn.list if issubclass(x.category, DeprecationWarning)]


# =============================================================================
# Exception hierarchy (seeds #227)
# =============================================================================


class TestExceptionHierarchy:
    def test_capability_error_is_esperanto_error(self):
        assert issubclass(ProviderCapabilityError, EsperantoError)
        assert issubclass(EsperantoError, Exception)


# =============================================================================
# get_profile_capabilities + get_available_providers
# =============================================================================


class TestCapabilityDiscovery:
    def test_builtins_are_language_only(self):
        caps = get_profile_capabilities()
        for name in ("deepseek", "xai", "dashscope", "minimax", "novita"):
            assert caps[name] == {"language"}

    def test_available_providers_merges_by_capability(self, _env):
        _register(
            name="multi",
            base_url="http://localhost:9/v1",
            api_key_env="MULTI_KEY",
            capabilities={"language", "embedding"},
            default_models={"language": "l", "embedding": "e"},
        )
        providers = AIFactory.get_available_providers()
        assert "multi" in providers["language"]
        assert "multi" in providers["embedding"]
        assert "multi" not in providers["speech_to_text"]
        assert "multi" not in providers["text_to_speech"]

    def test_chat_only_profile_absent_from_embedding_matrix(self):
        providers = AIFactory.get_available_providers()
        assert "deepseek" in providers["language"]
        assert "deepseek" not in providers["embedding"]


# =============================================================================
# Factory resolution across modalities
# =============================================================================


class TestFactoryResolution:
    @pytest.mark.parametrize(
        "modality, create, default_key, default_val",
        [
            ("embedding", AIFactory.create_embedding, "embedding", "m-emb"),
            (
                "speech_to_text",
                AIFactory.create_speech_to_text,
                "speech_to_text",
                "m-stt",
            ),
            (
                "text_to_speech",
                AIFactory.create_text_to_speech,
                "text_to_speech",
                "m-tts",
            ),
        ],
    )
    def test_profile_resolves_for_declared_modality(
        self, _env, modality, create, default_key, default_val
    ):
        _register(
            name="multi",
            base_url="http://localhost:9/v1",
            api_key_env="MULTI_KEY",
            capabilities={modality},
            default_models={default_key: default_val},
        )
        model = create("multi", None)
        assert type(model).__name__.startswith("OpenAICompatible")
        assert model.base_url == "http://localhost:9/v1"
        assert model.model_name == default_val

    def test_undeclared_modality_without_class_raises_capability_error(self, _env):
        _register(
            name="chatonly",
            base_url="http://localhost:9/v1",
            api_key_env="MULTI_KEY",
            capabilities={"language"},
            default_models={"language": "l"},
        )
        with pytest.raises(ProviderCapabilityError):
            AIFactory.create_embedding("chatonly", "whatever")

    def test_deepseek_embedding_raises_capability_error(self):
        with pytest.raises(ProviderCapabilityError):
            AIFactory.create_embedding("deepseek", "x")

    def test_no_default_beats_wrong_default(self, _env):
        _register(
            name="nodef",
            base_url="http://localhost:9/v1",
            api_key_env="NODEF_KEY",
            capabilities={"embedding"},
            default_models={},  # declares embedding but no default
        )
        with pytest.raises(ValueError, match="no default model"):
            AIFactory.create_embedding("nodef", None)

    def test_no_default_but_explicit_model_ok(self, _env):
        _register(
            name="nodef",
            base_url="http://localhost:9/v1",
            api_key_env="NODEF_KEY",
            capabilities={"embedding"},
            default_models={},
        )
        model = AIFactory.create_embedding("nodef", "user-model")
        assert model.model_name == "user-model"


# =============================================================================
# Hybrid provider: xAI is a language profile AND a first-class TTS class
# =============================================================================


class TestHybridProvider:
    def test_xai_tts_falls_through_to_first_class(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        model = AIFactory.create_text_to_speech("xai", "grok-tts")
        assert type(model).__name__ == "XAITextToSpeechModel"

    def test_xai_language_uses_profile(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        model = AIFactory.create_language("xai", "grok-2-latest")
        assert type(model).__name__ == "OpenAICompatibleLanguageModel"


# =============================================================================
# Collision warning
# =============================================================================


class TestCollisionWarning:
    def test_declaring_capability_that_shadows_class_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AIFactory.register_openai_compatible_profile(
                OpenAICompatibleProfile(
                    name="xai",
                    base_url="http://x/v1",
                    api_key_env="XAI_API_KEY",
                    capabilities={"language", "text_to_speech"},
                    default_models={"language": "grok"},
                )
            )
        assert any("shadows" in str(x.message) for x in w)

    def test_language_only_profile_does_not_warn_for_tts_class(self):
        # xAI default (language-only) must NOT warn even though a TTS class exists.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AIFactory.register_openai_compatible_profile(
                OpenAICompatibleProfile(
                    name="xai2",
                    base_url="http://x/v1",
                    api_key_env="XAI_API_KEY",
                    capabilities={"language"},
                    default_models={"language": "grok"},
                )
            )
        assert not [x for x in w if "shadows" in str(x.message)]
