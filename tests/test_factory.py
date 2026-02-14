"""Tests for the AIFactory class."""

import warnings

import pytest
from unittest.mock import patch, MagicMock

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