"""Tests for the Google TTS provider."""
import base64
import io
import wave
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


def test_default_model_is_gemini_38():
    """The default must stay available after the Gemini 3.1 preview shuts down."""
    model = GoogleTextToSpeechModel(api_key="test-key")
    assert model._get_default_model() == "gemini-3.8-flash-tts"


def test_generate_speech_default_model_omits_system_instruction():
    """Default model must NOT include systemInstruction — the new API
    rejects it with 'Developer instruction is not enabled for this model'."""
    model = GoogleTextToSpeechModel(api_key="test-key")
    mock_client = Mock()
    mock_client.post.return_value = _make_mock_response()
    model.client = mock_client

    response = model.generate_speech(text="Hello world", voice="achernar")

    assert response.model == "gemini-3.8-flash-tts"
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
    """Async path: default model omits systemInstruction."""
    model = GoogleTextToSpeechModel(api_key="test-key")
    mock_async_client = AsyncMock()
    mock_async_client.post.return_value = _make_mock_response()
    model.async_client = mock_async_client

    response = await model.agenerate_speech(text="Hello world", voice="achernar")

    assert response.model == "gemini-3.8-flash-tts"
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


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate_multi_speaker_speech", "agenerate_multi_speaker_speech"])
@pytest.mark.parametrize("model_name", [None, "gemini-3.8-flash-lite-tts"])
async def test_gemini_38_sends_speaker_metadata_for_each_turn(method, model_name):
    """GenerateContent needs speaker metadata on every 3.8 dialogue turn."""
    options = {"model_name": model_name} if model_name else {}
    model = GoogleTextToSpeechModel(api_key="test-key", **options)
    response = _make_mock_response()
    if method.startswith("a"):
        client = AsyncMock()
        model.async_client = client
    else:
        client = Mock()
        model.client = client
    client.post.return_value = response

    audio = getattr(model, method)(
        text=(
            "\nToday's topic: Testing\n\n joe : Wait: what?\n"
            "Note: Here's my point.\nJANE: I agree.\nJoe: Thanks!\n"
        ),
        speaker_configs=SPEAKERS,
    )
    if method.startswith("a"):
        audio = await audio
    expected_model = model_name or "gemini-3.8-flash-tts"
    assert audio.model == expected_model
    assert f"/v1beta/models/{expected_model}:generateContent" in client.post.call_args.args[0]
    payload = client.post.call_args.kwargs["json"]
    assert payload["contents"][0]["parts"] == [
        {
            "text": "Today's topic: Testing\n\nWait: what?\nNote: Here's my point.",
            "speechMetadata": {"speaker": "Joe"},
        },
        {"text": "I agree.", "speechMetadata": {"speaker": "Jane"}},
        {"text": "Thanks!", "speechMetadata": {"speaker": "Joe"}},
    ]
    assert "systemInstruction" not in payload


def test_gemini_38_unlabeled_dialogue_explains_migration():
    model = GoogleTextToSpeechModel(api_key="test-key")
    model.client = Mock()

    with pytest.raises(ValueError, match="label each turn with a configured speaker"):
        model.generate_multi_speaker_speech(
            text="This is an unlabeled paragraph.",
            speaker_configs=SPEAKERS,
        )

    model.client.post.assert_not_called()


def test_gemini_31_remains_discoverable_and_sends_unmodified_dialogue():
    model = GoogleTextToSpeechModel(
        api_key="test-key", model_name="gemini-3.1-flash-tts-preview"
    )
    assert model.model_name in [item.id for item in model._get_models()]

    client = Mock()
    client.post.return_value = _make_mock_response()
    model.client = client
    dialogue = "Intro: Today's topic.\njoe: Let's start."
    model.generate_multi_speaker_speech(text=dialogue, speaker_configs=SPEAKERS)
    payload = client.post.call_args.kwargs["json"]
    assert payload["contents"] == [{"parts": [{"text": dialogue}]}]
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
    assert payload["contents"] == [{"parts": [{"text": "Joe: hi\nJane: hello"}]}]


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


def _wav_bytes(pcm: bytes, sample_rate: int = 24000) -> bytes:
    """Build a complete WAV file, as Gemini 3.8 TTS returns it."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


def _make_audio_response(data: bytes, mime_type: str):
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": base64.b64encode(data).decode(),
                            }
                        }
                    ]
                }
            }
        ]
    }
    return response


PCM = b"\x01\x00\x02\x00" * 100
SPEAKERS = [
    {"speaker": "Joe", "voice": "Kore"},
    {"speaker": "Jane", "voice": "Puck"},
]


async def _call(model, method: str, response):
    """Run any of the four generation paths against a mocked response."""
    if method.startswith("a"):
        model.async_client = AsyncMock()
        model.async_client.post.return_value = response
    else:
        model.client = Mock()
        model.client.post.return_value = response

    if "multi_speaker" in method:
        result = getattr(model, method)(text="Joe: hi\nJane: hello", speaker_configs=SPEAKERS)
    else:
        result = getattr(model, method)(text="Hello world", voice="kore")
    if method.startswith("a"):
        result = await result
    return result


GENERATION_METHODS = [
    "generate_speech",
    "agenerate_speech",
    "generate_multi_speaker_speech",
    "agenerate_multi_speaker_speech",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", GENERATION_METHODS)
async def test_wav_response_is_returned_unchanged(method):
    """Gemini 3.8 returns audio/wav; it must not be wrapped in a second RIFF header."""
    model = GoogleTextToSpeechModel(api_key="test-key", model_name="gemini-3.8-flash-tts")
    wav = _wav_bytes(PCM)

    response = await _call(model, method, _make_audio_response(wav, "audio/wav"))

    assert response.audio_data == wav
    assert response.audio_data.count(b"RIFF") == 1
    assert response.content_type == "audio/wav"


@pytest.mark.asyncio
@pytest.mark.parametrize("method", GENERATION_METHODS)
async def test_riff_bytes_without_wav_mime_type_are_returned_unchanged(method):
    """RIFF bytes are detected even when the mime type does not say audio/wav."""
    model = GoogleTextToSpeechModel(api_key="test-key", model_name="gemini-3.8-flash-tts")
    wav = _wav_bytes(PCM)

    response = await _call(model, method, _make_audio_response(wav, ""))

    assert response.audio_data == wav


@pytest.mark.asyncio
@pytest.mark.parametrize("method", GENERATION_METHODS)
async def test_pcm_response_is_wrapped_in_wav(method):
    """Gemini 3.1 / 2.5 return headerless PCM; it is wrapped as before."""
    model = GoogleTextToSpeechModel(api_key="test-key")

    response = await _call(
        model, method, _make_audio_response(PCM, "audio/L16;codec=pcm;rate=24000")
    )

    assert response.audio_data == _wav_bytes(PCM)
    with wave.open(io.BytesIO(response.audio_data)) as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.readframes(wav_file.getnframes()) == PCM


def test_pcm_sample_rate_is_read_from_mime_type():
    model = GoogleTextToSpeechModel(api_key="test-key")
    model.client = Mock()
    model.client.post.return_value = _make_audio_response(
        PCM, "audio/L16;codec=pcm;rate=16000"
    )

    response = model.generate_speech(text="Hello world", voice="kore")

    with wave.open(io.BytesIO(response.audio_data)) as wav_file:
        assert wav_file.getframerate() == 16000


def test_pcm_without_mime_type_defaults_to_24khz():
    model = GoogleTextToSpeechModel(api_key="test-key")
    model.client = Mock()
    model.client.post.return_value = _make_mock_response()

    response = model.generate_speech(text="Hello world", voice="kore")

    assert response.audio_data == _wav_bytes(b"test audio data")


def test_gemini_38_models_are_listed():
    model = GoogleTextToSpeechModel(api_key="test-key")
    model_ids = {m.id for m in model._get_models()}
    assert {"gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"} <= model_ids


def test_bitrate_parameter_is_not_read_as_sample_rate():
    model = GoogleTextToSpeechModel(api_key="test-key")
    model.client = Mock()
    model.client.post.return_value = _make_audio_response(
        PCM, "audio/L16;bitrate=384000;rate=16000"
    )

    response = model.generate_speech(text="Hello world", voice="kore")

    with wave.open(io.BytesIO(response.audio_data)) as wav_file:
        assert wav_file.getframerate() == 16000


def test_riff_bytes_without_wave_marker_are_treated_as_pcm():
    """Only a real WAV container (RIFF....WAVE) is passed through unlabelled."""
    model = GoogleTextToSpeechModel(api_key="test-key")
    model.client = Mock()
    not_wav = b"RIFF\x00\x00\x00\x00AVI " + PCM
    model.client.post.return_value = _make_audio_response(not_wav, "")

    response = model.generate_speech(text="Hello world", voice="kore")

    assert response.audio_data == _wav_bytes(not_wav)
