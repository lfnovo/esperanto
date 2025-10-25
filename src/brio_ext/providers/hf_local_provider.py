"""Local Hugging Face (TGI/Transformers) provider integration."""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Union

import httpx

from esperanto.common_types import ChatCompletion, Choice, Message, Model, Usage
from esperanto.providers.llm.base import LanguageModel


class HuggingFaceLocalLanguageModel(LanguageModel):
    """Thin wrapper around a text-generation-inference (TGI) server."""

    def __post_init__(self):
        super().__post_init__()
        self.base_url = (
            self.base_url
            or os.getenv("HF_LOCAL_BASE_URL")
            or "http://localhost:8080"
        )
        self._create_http_clients()

    def models(self) -> List[Model]:
        # TGI does not expose a models route by default; return the configured model.
        model_name = self.get_model_name()
        return [
            Model(
                id=model_name,
                owned_by="local",
                context_window=None,
                type="language",
            )
        ]

    def prompt_complete(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        stream: Optional[bool] = None,
    ) -> ChatCompletion:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            raise NotImplementedError("Streaming not yet supported for hf_local provider")

        payload = {
            "inputs": prompt,
            "parameters": self._build_parameters(stop),
        }

        response = self.client.post(
            f"{self.base_url}/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        self._handle_error(response)
        data = response.json()
        return self._normalize_response(data)

    async def aprompt_complete(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        stream: Optional[bool] = None,
    ) -> ChatCompletion:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            raise NotImplementedError("Streaming not yet supported for hf_local provider")

        payload = {
            "inputs": prompt,
            "parameters": self._build_parameters(stop),
        }

        response = await self.async_client.post(
            f"{self.base_url}/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        self._handle_error(response)
        data = response.json()
        return self._normalize_response(data)

    def chat_complete(self, messages, stream: Optional[bool] = None):
        prompt = "\n".join(message.get("content", "") for message in messages)
        stop = getattr(self, "_config", {}).get("stop")
        return self.prompt_complete(prompt, stop=stop, stream=stream)

    async def achat_complete(self, messages, stream: Optional[bool] = None):
        prompt = "\n".join(message.get("content", "") for message in messages)
        stop = getattr(self, "_config", {}).get("stop")
        return await self.aprompt_complete(prompt, stop=stop, stream=stream)

    def _build_parameters(self, stop: Optional[List[str]]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        config = self.get_completion_kwargs()

        if config.get("max_tokens") is not None:
            params["max_new_tokens"] = config["max_tokens"]
        if config.get("temperature") is not None:
            params["temperature"] = config["temperature"]
        if config.get("top_p") is not None:
            params["top_p"] = config["top_p"]

        extras = getattr(self, "_config", {}) or {}
        if stop:
            params["stop"] = stop
        elif extras.get("stop"):
            params["stop"] = extras["stop"]

        if extras.get("repetition_penalty") is not None:
            params["repetition_penalty"] = extras["repetition_penalty"]

        return params

    def _normalize_response(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> ChatCompletion:
        payload = data[0] if isinstance(data, list) else data
        text = payload.get("generated_text") or payload.get("text") or ""
        details = payload.get("details") or {}
        usage = Usage(
            prompt_tokens=details.get("prompt_tokens", 0),
            completion_tokens=details.get("generated_tokens", 0),
            total_tokens=details.get("prompt_tokens", 0) + details.get("generated_tokens", 0),
        )

        return ChatCompletion(
            id=payload.get("id", str(uuid.uuid4())),
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=text),
                    finish_reason=details.get("finish_reason"),
                )
            ],
            created=None,
            model=self.get_model_name(),
            provider=self.provider,
            usage=usage,
        )

    def _handle_error(self, response: httpx.Response) -> None:
        if response.status_code >= 400:
            try:
                payload = response.json()
                message = payload.get("error") or payload.get("detail") or response.text
            except Exception:
                message = response.text
            raise RuntimeError(f"hf_local error ({response.status_code}): {message}")

    def _get_default_model(self) -> str:
        return "local-transformer"

    @property
    def provider(self) -> str:
        return "hf_local"

    def to_langchain(self) -> Any:
        raise NotImplementedError("LangChain integration not implemented for hf_local provider")
