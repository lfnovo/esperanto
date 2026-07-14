"""Test cases for the Cohere language model provider."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from esperanto.common_types import (
    ChatCompletion,
    ChatCompletionChunk,
    StructuredOutputValidationError,
    Tool,
    ToolFunction,
)
from esperanto.providers.llm.cohere import CohereLanguageModel

NON_STREAM_RESPONSE = {
    "id": "chat-123",
    "message": {
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello there!"}],
    },
    "finish_reason": "COMPLETE",
    "usage": {"tokens": {"input_tokens": 12, "output_tokens": 5}},
}

TOOL_RESPONSE = {
    "id": "chat-456",
    "message": {
        "role": "assistant",
        "tool_plan": "I will check the weather.",
        "tool_calls": [
            {
                "id": "tc-1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
            }
        ],
    },
    "finish_reason": "TOOL_CALL",
    "usage": {"tokens": {"input_tokens": 20, "output_tokens": 8}},
}

STREAM_LINES = [
    'data: {"type":"message-start","id":"chat-789"}\n',
    'data: {"type":"content-start","index":0,"delta":{"message":{"content":{"type":"text","text":""}}}}\n',
    'data: {"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"Hello"}}}}\n',
    'data: {"type":"content-delta","index":0,"delta":{"message":{"content":{"text":" world"}}}}\n',
    'data: {"type":"content-end","index":0}\n',
    'data: {"type":"message-end","delta":{"finish_reason":"COMPLETE"}}\n',
]


def _make_model(response_data=NON_STREAM_RESPONSE, config=None):
    # Avoid creating real httpx clients we immediately discard for mocks.
    with patch.object(CohereLanguageModel, "_create_http_clients", lambda self: None):
        model = CohereLanguageModel(
            model_name="command-a-03-2025", api_key="test-key", config=config or {}
        )

    def post(url, **kwargs):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = response_data
        resp.iter_text = Mock(return_value=iter(STREAM_LINES))

        async def aiter_text():
            for line in STREAM_LINES:
                yield line

        resp.aiter_text = Mock(return_value=aiter_text())
        return resp

    async def apost(url, **kwargs):
        return post(url, **kwargs)

    mock_client = Mock()
    mock_async_client = AsyncMock()
    mock_client.post.side_effect = post
    mock_async_client.post.side_effect = apost
    model.client = mock_client
    model.async_client = mock_async_client
    return model


def _weather_tool():
    return Tool(
        type="function",
        function=ToolFunction(
            name="get_weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        ),
    )


class TestCohereLanguageModel:
    def test_initialization(self):
        model = CohereLanguageModel(model_name="command-a-03-2025", api_key="test-key", config={})
        assert model.api_key == "test-key"
        assert model.provider == "cohere"
        assert model.base_url == "https://api.cohere.com"

    def test_missing_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Cohere API key not found"):
                CohereLanguageModel(model_name="command-a-03-2025", api_key=None, config={})

    def test_env_var(self):
        with patch.dict("os.environ", {"COHERE_API_KEY": "env-key"}):
            model = CohereLanguageModel(model_name="command-a-03-2025", api_key=None, config={})
            assert model.api_key == "env-key"

    def test_default_model(self):
        model = CohereLanguageModel(api_key="test-key", config={})
        assert model.get_model_name() == "command-a-03-2025"

    def test_headers(self):
        model = CohereLanguageModel(model_name="command-a-03-2025", api_key="secret", config={})
        headers = model._get_headers()
        assert headers["Authorization"] == "Bearer secret"

    def test_chat_complete(self):
        model = _make_model()
        response = model.chat_complete([{"role": "user", "content": "Hi"}])

        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content == "Hello there!"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 12
        assert response.usage.completion_tokens == 5

        url = model.client.post.call_args[0][0]
        assert url == "https://api.cohere.com/v2/chat"

    def test_payload_uses_p_not_top_p(self):
        model = _make_model()
        model.chat_complete(
            [{"role": "user", "content": "Hi"}], temperature=0.5, top_p=0.8, max_tokens=100
        )
        payload = model.client.post.call_args[1]["json"]
        assert payload["model"] == "command-a-03-2025"
        assert payload["temperature"] == 0.5
        assert payload["p"] == 0.8
        assert "top_p" not in payload
        assert payload["max_tokens"] == 100

    def test_documents_passthrough(self):
        docs = [{"id": "doc1", "data": {"text": "Paris is in France"}}]
        model = _make_model(config={"documents": docs})
        model.chat_complete([{"role": "user", "content": "Where is Paris?"}])
        payload = model.client.post.call_args[1]["json"]
        assert payload["documents"] == docs

    def test_tool_conversion(self):
        model = _make_model()
        cohere_tools = model._convert_tools_to_cohere([_weather_tool()])
        assert cohere_tools[0]["type"] == "function"
        assert cohere_tools[0]["function"]["name"] == "get_weather"
        assert "parameters" in cohere_tools[0]["function"]

    def test_tool_choice_conversion(self):
        model = _make_model()
        assert model._convert_tool_choice_to_cohere("auto") is None
        assert model._convert_tool_choice_to_cohere("required") == "REQUIRED"
        assert model._convert_tool_choice_to_cohere("none") == "NONE"

    def test_tool_calling_response(self):
        model = _make_model(response_data=TOOL_RESPONSE)
        response = model.chat_complete(
            [{"role": "user", "content": "Weather in Tokyo?"}], tools=[_weather_tool()]
        )
        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls is not None
        assert tool_calls[0].function.name == "get_weather"
        assert tool_calls[0].function.arguments == '{"city": "Tokyo"}'
        assert response.choices[0].finish_reason == "tool_calls"

    def test_finish_reason_mapping(self):
        model = _make_model()
        assert model._map_finish_reason("COMPLETE") == "stop"
        assert model._map_finish_reason("MAX_TOKENS") == "length"
        assert model._map_finish_reason("TOOL_CALL") == "tool_calls"
        assert model._map_finish_reason("STOP_SEQUENCE") == "stop"
        assert model._map_finish_reason(None) == "stop"

    def test_streaming(self):
        model = _make_model()
        chunks = list(model.chat_complete([{"role": "user", "content": "Hi"}], stream=True))
        assert all(isinstance(c, ChatCompletionChunk) for c in chunks)
        text = "".join(
            c.choices[0].delta.content or "" for c in chunks
        )
        assert text == "Hello world"
        assert chunks[-1].choices[0].finish_reason == "stop"

    def test_tool_message_formatting(self):
        model = _make_model()
        messages = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [
                    {"id": "tc-1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "tc-1", "content": "Sunny"},
        ]
        formatted = model._format_messages(messages)
        assert formatted[1]["role"] == "assistant"
        assert formatted[1]["tool_plan"] == "checking"
        assert formatted[1]["tool_calls"][0]["function"]["arguments"] == '{"city":"Tokyo"}'
        assert formatted[2]["role"] == "tool"
        assert formatted[2]["tool_call_id"] == "tc-1"

    @pytest.mark.asyncio
    async def test_achat_complete(self):
        model = _make_model()
        response = await model.achat_complete([{"role": "user", "content": "Hi"}])
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content == "Hello there!"
        model.async_client.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_streaming(self):
        model = _make_model()
        gen = await model.achat_complete([{"role": "user", "content": "Hi"}], stream=True)
        chunks = [c async for c in gen]
        text = "".join(c.choices[0].delta.content or "" for c in chunks)
        assert text == "Hello world"


class CityInfo(BaseModel):
    city: str
    country: str


def _structured_response(json_text: str):
    """Build a Cohere v2 native response whose text content is ``json_text``."""
    return {
        "id": "chat-structured",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": json_text}],
        },
        "finish_reason": "COMPLETE",
        "usage": {"tokens": {"input_tokens": 12, "output_tokens": 8}},
    }


class TestCohereStructuredOutput:
    def test_json_schema_pydantic(self):
        model = _make_model(
            response_data=_structured_response('{"city": "Paris", "country": "France"}')
        )
        model.structured = {"type": "json_schema", "schema": CityInfo}

        response = model.chat_complete([{"role": "user", "content": "Capital of France?"}])

        # Request shape: json_object + schema present.
        payload = model.client.post.call_args[1]["json"]
        assert payload["response_format"]["type"] == "json_object"
        assert payload["response_format"]["schema"] == CityInfo.model_json_schema()

        # Parsed structured result is a validated Pydantic instance.
        assert isinstance(response.structured, CityInfo)
        assert response.structured.city == "Paris"
        assert response.structured.country == "France"

    def test_json_schema_dict(self):
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
            "required": ["city", "country"],
        }
        model = _make_model(
            response_data=_structured_response('{"city": "Rome", "country": "Italy"}')
        )
        model.structured = {"type": "json_schema", "schema": schema}

        response = model.chat_complete([{"role": "user", "content": "Capital of Italy?"}])

        payload = model.client.post.call_args[1]["json"]
        assert payload["response_format"]["type"] == "json_object"
        assert payload["response_format"]["schema"] == schema

        assert response.structured == {"city": "Rome", "country": "Italy"}

    def test_json_object_mode(self):
        model = _make_model(
            response_data=_structured_response('{"city": "Madrid"}')
        )
        model.structured = {"type": "json_object"}

        response = model.chat_complete([{"role": "user", "content": "Give JSON"}])

        payload = model.client.post.call_args[1]["json"]
        assert payload["response_format"] == {"type": "json_object"}
        assert "schema" not in payload["response_format"]
        # json_object mode does not populate the parsed structured attribute.
        assert response.structured is None

    def test_invalid_json_raises(self):
        model = _make_model(response_data=_structured_response("not valid json"))
        model.structured = {"type": "json_schema", "schema": CityInfo}

        with pytest.raises(StructuredOutputValidationError):
            model.chat_complete([{"role": "user", "content": "Capital?"}])

    def test_streaming_schema_mode_raises(self):
        model = _make_model()
        model.structured = {"type": "json_schema", "schema": CityInfo}

        with pytest.raises(ValueError, match="not supported with streaming"):
            model.chat_complete([{"role": "user", "content": "Capital?"}], stream=True)

    @pytest.mark.asyncio
    async def test_streaming_schema_mode_raises_async(self):
        model = _make_model()
        model.structured = {"type": "json_schema", "schema": CityInfo}

        with pytest.raises(ValueError, match="not supported with streaming"):
            await model.achat_complete(
                [{"role": "user", "content": "Capital?"}], stream=True
            )

    @pytest.mark.asyncio
    async def test_json_schema_pydantic_async(self):
        model = _make_model(
            response_data=_structured_response('{"city": "Berlin", "country": "Germany"}')
        )
        model.structured = {"type": "json_schema", "schema": CityInfo}

        response = await model.achat_complete(
            [{"role": "user", "content": "Capital of Germany?"}]
        )

        payload = model.async_client.post.call_args[1]["json"]
        assert payload["response_format"]["type"] == "json_object"
        assert "schema" in payload["response_format"]

        assert isinstance(response.structured, CityInfo)
        assert response.structured.city == "Berlin"
        assert response.structured.country == "Germany"
