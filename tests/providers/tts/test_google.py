"""Tests for the Google TTS provider."""
import base64
from unittest.mock import AsyncMock, Mock

import pytest

from esperanto.providers.tts.google import GoogleTextToSpeechModel


def test_init():
    """Test model initialization."""
    model = GoogleTextToSpeechModel(api_key="test-key")
    assert model.provider == "google"


def _make_mock_response():
    """Build a mock Google TTS HTTP response with valid base64-encoded PCM data."""
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "data": base64.b64encode(b"test audio data").decode()
                            }
                        }
                    ]
                }
            }
        ]
    }
    return response


def test_default_model_is_gemini_31():
    """Default model is gemini-3.1-flash-tts-preview (newer; gemini-2.5-* preview models
    are now flaky on Google's side and require the legacy systemInstruction quirk)."""
    model = GoogleTextToSpeechModel(api_key="test-key")
    assert model._get_default_model() == "gemini-3.1-flash-tts-preview"


def test_generate_speech_default_model_omits_system_instruction():
    """Default model (gemini-3.1-*) must NOT include systemInstruction — the new API
    rejects it with 'Developer instruction is not enabled for this model'."""
    model = GoogleTextToSpeechModel(api_key="test-key")
    mock_client = Mock()
    mock_client.post.return_value = _make_mock_response()
    model.client = mock_client

    response = model.generate_speech(text="Hello world", voice="achernar")

    assert response.model == "gemini-3.1-flash-tts-preview"
    assert response.voice == "achernar"
    assert response.provider == "google"
    payload = mock_client.post.call_args.kwargs["json"]
    assert "systemInstruction" not in payload


def test_generate_speech_legacy_model_includes_system_instruction():
    """Legacy gemini-2.5-* models still require the systemInstruction quirk from #178."""
    model = GoogleTextToSpeechModel(
        api_key="test-key",
        model_name="gemini-2.5-flash-preview-tts",
    )
    mock_client = Mock()
    mock_client.post.return_value = _make_mock_response()
    model.client = mock_client

    model.generate_speech(text="Hello world", voice="achernar")

    payload = mock_client.post.call_args.kwargs["json"]
    assert payload["systemInstruction"] == {
        "parts": [{"text": "Read aloud the following text."}]
    }


@pytest.mark.asyncio
async def test_agenerate_speech_default_model_omits_system_instruction():
    """Async path: default model (3.1) omits systemInstruction."""
    model = GoogleTextToSpeechModel(api_key="test-key")
    mock_async_client = AsyncMock()
    mock_async_client.post.return_value = _make_mock_response()
    model.async_client = mock_async_client

    response = await model.agenerate_speech(text="Hello world", voice="achernar")

    assert response.model == "gemini-3.1-flash-tts-preview"
    payload = mock_async_client.post.call_args.kwargs["json"]
    assert "systemInstruction" not in payload


@pytest.mark.asyncio
async def test_agenerate_speech_legacy_model_includes_system_instruction():
    """Async path: legacy gemini-2.5-* models include systemInstruction."""
    model = GoogleTextToSpeechModel(
        api_key="test-key",
        model_name="gemini-2.5-pro-preview-tts",
    )
    mock_async_client = AsyncMock()
    mock_async_client.post.return_value = _make_mock_response()
    model.async_client = mock_async_client

    await model.agenerate_speech(text="Hello world", voice="achernar")

    payload = mock_async_client.post.call_args.kwargs["json"]
    assert payload["systemInstruction"] == {
        "parts": [{"text": "Read aloud the following text."}]
    }


def test_multi_speaker_default_model_omits_system_instruction():
    """Multi-speaker variant honors the same conditional rule."""
    model = GoogleTextToSpeechModel(api_key="test-key")
    mock_client = Mock()
    mock_client.post.return_value = _make_mock_response()
    model.client = mock_client

    model.generate_multi_speaker_speech(
        text="Joe: hi\nJane: hello",
        speaker_configs=[
            {"speaker": "Joe", "voice": "Kore"},
            {"speaker": "Jane", "voice": "Puck"},
        ],
    )

    payload = mock_client.post.call_args.kwargs["json"]
    assert "systemInstruction" not in payload


def test_multi_speaker_legacy_model_includes_system_instruction():
    """Multi-speaker variant on legacy gemini-2.5-* still emits systemInstruction."""
    model = GoogleTextToSpeechModel(
        api_key="test-key",
        model_name="gemini-2.5-flash-preview-tts",
    )
    mock_client = Mock()
    mock_client.post.return_value = _make_mock_response()
    model.client = mock_client

    model.generate_multi_speaker_speech(
        text="Joe: hi\nJane: hello",
        speaker_configs=[
            {"speaker": "Joe", "voice": "Kore"},
            {"speaker": "Jane", "voice": "Puck"},
        ],
    )

    payload = mock_client.post.call_args.kwargs["json"]
    assert payload["systemInstruction"] == {
        "parts": [{"text": "Read aloud the following text."}]
    }


def test_available_voices():
    """Test getting available voices (predefined list)."""
    from esperanto.providers.tts.google import GoogleTextToSpeechModel
    
    # Create fresh model instance
    model = GoogleTextToSpeechModel(api_key="test-key")

    # Test getting voices (Google TTS uses predefined voices)
    voices = model.available_voices
    assert len(voices) == 30  # Google has 30 predefined voices
    
    # Test a few specific voices to ensure structure is correct
    assert "achernar" in voices
    assert voices["achernar"].name == "UpbeatAchernar"
    assert voices["achernar"].id == "achernar"
    assert voices["achernar"].gender == "FEMALE"
    
    assert "charon" in voices
    assert voices["charon"].name == "UpbeatCharon"
    assert voices["charon"].id == "charon"
    assert voices["charon"].gender == "MALE"
