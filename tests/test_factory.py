"""Tests for the AIFactory class."""

import warnings
from unittest.mock import MagicMock, patch

from esperanto.factory import AIFactory

# Note: Caching functionality has been removed from AIFactory.
# Tests now verify that factory creates new instances each time.


class TestCreateTextToSpeech:
    """Tests for create_text_to_speech config dict support."""

    @patch.object(AIFactory, "_import_provider_class")
    def test_config_dict_passes_api_key_and_base_url(self, mock_import):
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls

        AIFactory.create_text_to_speech(
            "openai",
            model_name="tts-1",
            config={"api_key": "sk-test", "base_url": "https://custom.api"},
        )

        mock_cls.assert_called_once_with(
            model_name="tts-1",
            api_key="sk-test",
            base_url="https://custom.api",
        )

    @patch.object(AIFactory, "_import_provider_class")
    def test_direct_api_key_emits_deprecation_warning(self, mock_import):
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AIFactory.create_text_to_speech(
                "openai", model_name="tts-1", api_key="sk-test"
            )

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1
        assert "api_key" in str(deprecation_warnings[0].message)

        mock_cls.assert_called_once_with(
            model_name="tts-1",
            api_key="sk-test",
        )

    @patch.object(AIFactory, "_import_provider_class")
    def test_direct_base_url_emits_deprecation_warning(self, mock_import):
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AIFactory.create_text_to_speech(
                "openai", model_name="tts-1", base_url="https://custom.api"
            )

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1
        assert "base_url" in str(deprecation_warnings[0].message)

    @patch.object(AIFactory, "_import_provider_class")
    def test_direct_params_override_config(self, mock_import):
        """Direct api_key/base_url take precedence when config also has them."""
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            AIFactory.create_text_to_speech(
                "openai",
                model_name="tts-1",
                config={"api_key": "config-key", "base_url": "https://config.api"},
                api_key="direct-key",
                base_url="https://direct.api",
            )

        mock_cls.assert_called_once_with(
            model_name="tts-1",
            api_key="direct-key",
            base_url="https://direct.api",
        )

    @patch.object(AIFactory, "_import_provider_class")
    def test_config_without_api_key(self, mock_import):
        """Config dict works even without api_key/base_url."""
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls

        AIFactory.create_text_to_speech(
            "openai",
            model_name="tts-1",
            config={"voice": "alloy"},
        )

        mock_cls.assert_called_once_with(
            model_name="tts-1",
            voice="alloy",
        )

    @patch.object(AIFactory, "_import_provider_class")
    def test_create_tts_alias_forwards_config(self, mock_import):
        """The deprecated create_tts alias correctly forwards config."""
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AIFactory.create_tts(
                "openai",
                model_name="tts-1",
                config={"api_key": "sk-test"},
            )

        # Should have the create_tts deprecation warning
        tts_warnings = [
            x for x in w
            if issubclass(x.category, DeprecationWarning) and "create_tts" in str(x.message)
        ]
        assert len(tts_warnings) == 1

        mock_cls.assert_called_once_with(
            model_name="tts-1",
            api_key="sk-test",
        )


class TestCreateLanguage:
    """Tests for create_language config dict dispatch.

    LLM providers receive ``config`` as a single kwarg (passthrough pattern);
    the base ``LanguageModel.__post_init__`` is responsible for unpacking
    ``api_key``/``base_url`` onto instance attributes.
    """

    @patch.object(AIFactory, "_import_provider_class")
    def test_config_dict_forwarded_unchanged(self, mock_import):
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls

        AIFactory.create_language(
            "openai",
            "gpt-4o",
            config={"api_key": "sk-test", "base_url": "https://custom.api/v1"},
        )

        mock_cls.assert_called_once_with(
            model_name="gpt-4o",
            config={"api_key": "sk-test", "base_url": "https://custom.api/v1"},
        )

    @patch.object(AIFactory, "_import_provider_class")
    def test_none_config_becomes_empty_dict(self, mock_import):
        """``config=None`` must reach the provider as an empty dict."""
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls

        AIFactory.create_language("openai", "gpt-4o")

        mock_cls.assert_called_once_with(model_name="gpt-4o", config={})

    @patch.object(AIFactory, "_import_provider_class")
    def test_create_llm_alias_forwards_config(self, mock_import):
        """The deprecated ``create_llm`` alias preserves the config dict."""
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AIFactory.create_llm(
                "openai",
                "gpt-4o",
                config={"api_key": "sk-test", "base_url": "https://custom.api/v1"},
            )

        llm_warnings = [
            x for x in w
            if issubclass(x.category, DeprecationWarning) and "create_llm" in str(x.message)
        ]
        assert len(llm_warnings) == 1

        mock_cls.assert_called_once_with(
            model_name="gpt-4o",
            config={"api_key": "sk-test", "base_url": "https://custom.api/v1"},
        )


class TestCreateEmbedding:
    """Tests for create_embedding config dict dispatch (passthrough pattern)."""

    @patch.object(AIFactory, "_import_provider_class")
    def test_config_dict_forwarded_unchanged(self, mock_import):
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls

        AIFactory.create_embedding(
            "openai",
            "text-embedding-3-small",
            config={"api_key": "sk-test", "base_url": "https://custom.api/v1"},
        )

        mock_cls.assert_called_once_with(
            model_name="text-embedding-3-small",
            config={"api_key": "sk-test", "base_url": "https://custom.api/v1"},
        )

    @patch.object(AIFactory, "_import_provider_class")
    def test_none_config_becomes_empty_dict(self, mock_import):
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls

        AIFactory.create_embedding("openai", "text-embedding-3-small")

        mock_cls.assert_called_once_with(
            model_name="text-embedding-3-small", config={}
        )


class TestCreateReranker:
    """Tests for create_reranker config dict dispatch (passthrough pattern)."""

    @patch.object(AIFactory, "_import_provider_class")
    def test_config_dict_forwarded_unchanged(self, mock_import):
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls

        AIFactory.create_reranker(
            "jina",
            "jina-reranker-v2-base-multilingual",
            config={"api_key": "jina-test", "base_url": "https://custom.api/v1"},
        )

        mock_cls.assert_called_once_with(
            model_name="jina-reranker-v2-base-multilingual",
            config={"api_key": "jina-test", "base_url": "https://custom.api/v1"},
        )

    @patch.object(AIFactory, "_import_provider_class")
    def test_none_config_becomes_empty_dict(self, mock_import):
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls

        AIFactory.create_reranker("jina")

        mock_cls.assert_called_once_with(model_name=None, config={})


class TestCreateSpeechToText:
    """Tests for create_speech_to_text config dict dispatch.

    STT dispatch unpacks ``**config`` as direct kwargs (mirrors TTS); the
    provider ``__init__``/``__post_init__`` is responsible for honoring them.
    """

    @patch.object(AIFactory, "_import_provider_class")
    def test_config_dict_unpacked_to_kwargs(self, mock_import):
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls

        AIFactory.create_speech_to_text(
            "openai",
            model_name="whisper-1",
            config={"api_key": "sk-test", "base_url": "https://custom.api/v1"},
        )

        mock_cls.assert_called_once_with(
            model_name="whisper-1",
            api_key="sk-test",
            base_url="https://custom.api/v1",
        )

    @patch.object(AIFactory, "_import_provider_class")
    def test_none_config_passes_only_model_name(self, mock_import):
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls

        AIFactory.create_speech_to_text("openai", model_name="whisper-1")

        mock_cls.assert_called_once_with(model_name="whisper-1")

    @patch.object(AIFactory, "_import_provider_class")
    def test_create_stt_alias_forwards_config(self, mock_import):
        """The deprecated ``create_stt`` alias correctly forwards config."""
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AIFactory.create_stt(
                "openai",
                model_name="whisper-1",
                config={"api_key": "sk-test"},
            )

        stt_warnings = [
            x for x in w
            if issubclass(x.category, DeprecationWarning) and "create_stt" in str(x.message)
        ]
        assert len(stt_warnings) == 1

        mock_cls.assert_called_once_with(
            model_name="whisper-1",
            api_key="sk-test",
        )