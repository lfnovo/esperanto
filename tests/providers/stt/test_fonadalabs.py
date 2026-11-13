"""Native REST contract tests, with no paid requests."""

from io import BytesIO

import httpx
import pytest

from esperanto import AIFactory
from esperanto.providers.stt.fonadalabs import MAX_UPLOAD_BYTES


@pytest.fixture
def model():
    return AIFactory.create_speech_to_text(
        "fonadalabs", "kapture-v3", {"api_key": "test-only"}
    )


@pytest.mark.asyncio
async def test_native_request_and_response(model):
    calls = []

    async def handler(request):
        calls.append(request)
        assert str(request.url) == "https://api.fonada.ai/v3/asr/transcribe"
        assert request.headers["authorization"] == "Bearer test-only"
        body = await request.aread()
        assert b'name="language_id"\r\n\r\nhi' in body
        assert b'name="file"' in body
        return httpx.Response(200, json={"result": {"transcript": "lecture text"}})

    await model.async_client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model.async_client = client
        result = await model.atranscribe(BytesIO(b"audio"), language="hi")
    model.client.close()
    assert result.text == "lecture text"
    assert result.segments is None
    assert result.duration is None
    assert result.provider == "fonadalabs"
    assert len(calls) == 1


@pytest.mark.parametrize("status", [401, 402, 403, 413, 422, 429, 500, 502])
def test_errors_are_sanitized_without_retries(model, status, monkeypatch):
    waits = []
    monkeypatch.setattr("esperanto.providers.stt.fonadalabs.time.sleep", waits.append)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, text="sensitive upstream contents")

    model.client.close()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model.client = client
        with pytest.raises(RuntimeError, match=f"HTTP {status}") as error:
            model.transcribe(BytesIO(b"audio"), language="en")
    assert "sensitive" not in str(error.value)
    assert len(calls) == (2 if status == 429 else 1)
    assert waits == ([60] if status == 429 else [])


@pytest.mark.asyncio
async def test_timeout_is_not_retried(model):
    calls = []

    async def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("private details")

    await model.async_client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model.async_client = client
        with pytest.raises(RuntimeError, match="already be billed"):
            await model.atranscribe(BytesIO(b"audio"))
    model.client.close()
    assert len(calls) == 1


def test_size_and_prompt_validation(model):
    for data in (b"", b"a" * (MAX_UPLOAD_BYTES + 1)):
        with pytest.raises(ValueError, match="25 MB"):
            model.transcribe(BytesIO(data))
    with pytest.raises(ValueError, match="prompts"):
        model.transcribe(BytesIO(b"audio"), prompt="unsupported")
    model.client.close()


def test_static_discovery():
    assert AIFactory.get_provider_models("fonadalabs")[0].id == "kapture-v3"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.fonada.ai/v3/asr",
        "https://api.fonada.ai/v3/asr/",
        "https://api.fonada.ai/v3/asr/transcribe",
        "https://api.fonada.ai/v3/asr/transcribe/",
    ],
)
@pytest.mark.asyncio
async def test_full_endpoint_and_base_url(base_url):
    model = AIFactory.create_speech_to_text(
        "fonadalabs",
        "kapture-v3",
        {
            "api_key": "test-only",
            "base_url": base_url,
        },
    )

    def handler(request):
        assert str(request.url) == "https://api.fonada.ai/v3/asr/transcribe"
        return httpx.Response(200, json={"text": "ok"})

    model.client.close()
    await model.async_client.aclose()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model.client = client
        assert model.transcribe(BytesIO(b"test")).text == "ok"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model.async_client = client
        assert (await model.atranscribe(BytesIO(b"test"))).text == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("second_status", [200, 429, 502])
async def test_rate_limit_waits_one_minute_then_retries_once(
    model, monkeypatch, second_status
):
    from unittest.mock import AsyncMock

    wait = AsyncMock()
    monkeypatch.setattr("esperanto.providers.stt.fonadalabs.asyncio.sleep", wait)
    calls = []

    async def handler(request):
        calls.append(await request.aread())
        status = 429 if len(calls) == 1 else second_status
        return httpx.Response(status, json={"text": "resumed"})

    await model.async_client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model.async_client = client
        if second_status == 200:
            assert (await model.atranscribe(BytesIO(b"same audio"))).text == "resumed"
        else:
            with pytest.raises(RuntimeError, match=f"HTTP {second_status}"):
                await model.atranscribe(BytesIO(b"same audio"))
    model.client.close()
    wait.assert_awaited_once_with(60)
    assert len(calls) == 2
    assert all(b"same audio" in body for body in calls)


@pytest.mark.parametrize(
    "value,expected", [("120", 120), ("5", 60), ("invalid", 60), ("-2", 60)]
)
def test_retry_after_numeric_and_invalid(model, value, expected):
    response = httpx.Response(429, headers={"Retry-After": value})
    assert model._rate_limit_delay(response) == expected
    model.client.close()


def test_retry_after_date_and_long_suspension(model):
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime

    deadline = datetime.now(timezone.utc) + timedelta(seconds=180)
    delay = model._rate_limit_delay(
        httpx.Response(429, headers={"Retry-After": format_datetime(deadline)})
    )
    assert 178 <= delay <= 180
    with pytest.raises(RuntimeError, match="longer than one hour"):
        model._rate_limit_delay(httpx.Response(429, headers={"Retry-After": "7200"}))
    model.client.close()
