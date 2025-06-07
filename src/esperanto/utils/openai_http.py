import json
from typing import Any, AsyncGenerator, Dict, Generator, Optional
from types import SimpleNamespace

import requests
import httpx


class _Base:
    def __init__(self, api_key: str, base_url: Optional[str], organization: Optional[str], extra_headers: Optional[Dict[str, str]] = None) -> None:
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"}
        if organization:
            headers["OpenAI-Organization"] = organization
        if extra_headers:
            headers.update(extra_headers)
        self.headers = headers


class OpenAIHTTPClient(_Base):
    def __init__(self, api_key: str, base_url: Optional[str] = None, organization: Optional[str] = None, extra_headers: Optional[Dict[str, str]] = None) -> None:
        super().__init__(api_key, base_url, organization, extra_headers)
        self.session = requests.Session()
        self.chat = _Chat(self)
        self.embeddings = _Embeddings(self)
        self.models = _Models(self)
        self.audio = _Audio(self)


class AsyncOpenAIHTTPClient(_Base):
    def __init__(self, api_key: str, base_url: Optional[str] = None, organization: Optional[str] = None, extra_headers: Optional[Dict[str, str]] = None) -> None:
        super().__init__(api_key, base_url, organization, extra_headers)
        self.async_session = httpx.AsyncClient()
        self.chat = _AsyncChat(self)
        self.embeddings = _AsyncEmbeddings(self)
        self.models = _AsyncModels(self)
        self.audio = _AsyncAudio(self)


class _Chat:
    def __init__(self, client: OpenAIHTTPClient) -> None:
        self.completions = _ChatCompletions(client)


class _AsyncChat:
    def __init__(self, client: AsyncOpenAIHTTPClient) -> None:
        self.completions = _AsyncChatCompletions(client)


class _ChatCompletions:
    def __init__(self, client: OpenAIHTTPClient) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        url = f"{self._client.base_url}/chat/completions"
        stream = bool(kwargs.get("stream"))
        response = self._client.session.post(url, headers=self._client.headers, json=kwargs, stream=stream)
        response.raise_for_status()
        if stream:
            def gen() -> Generator[Dict[str, Any], None, None]:
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith(b"data: "):
                        data = line[6:]
                        if data.strip() == b"[DONE]":
                            break
                        yield json.loads(data.decode())
            return gen()
        return response.json()


class _AsyncChatCompletions:
    def __init__(self, client: AsyncOpenAIHTTPClient) -> None:
        self._client = client

    async def create(self, **kwargs: Any) -> Any:
        url = f"{self._client.base_url}/chat/completions"
        stream = bool(kwargs.get("stream"))
        async with self._client.async_session.stream("POST", url, headers=self._client.headers, json=kwargs) as resp:
            resp.raise_for_status()
            if stream:
                async def agen() -> AsyncGenerator[Dict[str, Any], None]:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                break
                            yield json.loads(data)
                return agen()
            return await resp.json()


class _Embeddings:
    def __init__(self, client: OpenAIHTTPClient) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        url = f"{self._client.base_url}/embeddings"
        res = self._client.session.post(url, headers=self._client.headers, json=kwargs)
        res.raise_for_status()
        return res.json()


class _AsyncEmbeddings:
    def __init__(self, client: AsyncOpenAIHTTPClient) -> None:
        self._client = client

    async def create(self, **kwargs: Any) -> Any:
        url = f"{self._client.base_url}/embeddings"
        res = await self._client.async_session.post(url, headers=self._client.headers, json=kwargs)
        res.raise_for_status()
        return res.json()


class _Models:
    def __init__(self, client: OpenAIHTTPClient) -> None:
        self._client = client

    def list(self) -> Any:
        url = f"{self._client.base_url}/models"
        res = self._client.session.get(url, headers=self._client.headers)
        res.raise_for_status()
        return res.json().get("data", [])


class _AsyncModels:
    def __init__(self, client: AsyncOpenAIHTTPClient) -> None:
        self._client = client

    async def list(self) -> Any:
        url = f"{self._client.base_url}/models"
        res = await self._client.async_session.get(url, headers=self._client.headers)
        res.raise_for_status()
        return res.json().get("data", [])


class _Audio:
    def __init__(self, client: OpenAIHTTPClient) -> None:
        self.speech = _Speech(client)
        self.transcriptions = _Transcriptions(client)


class _AsyncAudio:
    def __init__(self, client: AsyncOpenAIHTTPClient) -> None:
        self.speech = _AsyncSpeech(client)
        self.transcriptions = _AsyncTranscriptions(client)


class _Speech:
    def __init__(self, client: OpenAIHTTPClient) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        url = f"{self._client.base_url}/audio/speech"
        res = self._client.session.post(url, headers=self._client.headers, json=kwargs)
        res.raise_for_status()
        return res


class _AsyncSpeech:
    def __init__(self, client: AsyncOpenAIHTTPClient) -> None:
        self._client = client

    async def create(self, **kwargs: Any) -> Any:
        url = f"{self._client.base_url}/audio/speech"
        res = await self._client.async_session.post(url, headers=self._client.headers, json=kwargs)
        res.raise_for_status()
        return res


class _Transcriptions:
    def __init__(self, client: OpenAIHTTPClient) -> None:
        self._client = client

    def create(self, file: Any, **kwargs: Any) -> Any:
        url = f"{self._client.base_url}/audio/transcriptions"
        files = {"file": file if not isinstance(file, str) else open(file, "rb")}
        res = self._client.session.post(url, headers=self._client.headers, files=files, data=kwargs)
        res.raise_for_status()
        return res.json()


class _AsyncTranscriptions:
    def __init__(self, client: AsyncOpenAIHTTPClient) -> None:
        self._client = client

    async def create(self, file: Any, **kwargs: Any) -> Any:
        url = f"{self._client.base_url}/audio/transcriptions"
        if isinstance(file, str):
            with open(file, "rb") as f:
                res = await self._client.async_session.post(url, headers=self._client.headers, files={"file": f}, data=kwargs)
        else:
            res = await self._client.async_session.post(url, headers=self._client.headers, files={"file": file}, data=kwargs)
        res.raise_for_status()
        return await res.json()
