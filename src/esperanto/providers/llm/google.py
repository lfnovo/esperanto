"""Google GenAI language model provider."""

import json
import os
import time
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Union,
)

import httpx

from esperanto.common_types import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    DeltaMessage,
    Message,
    Model,
    StreamChoice,
    Usage,
)
from esperanto.providers.llm.base import LanguageModel

if TYPE_CHECKING:
    pass  # Removed unused import


class GoogleLanguageModel(LanguageModel):
    """Google GenAI language model implementation."""

    def __post_init__(self):
        """Initialize HTTP clients."""
        super().__post_init__()

        # Get API key
        self.api_key = (
            self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Google API key not found. Please set GOOGLE_API_KEY environment variable."
            )

        # Set base URL
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

        # Initialize HTTP clients
        self.client = httpx.Client(timeout=30.0)
        self.async_client = httpx.AsyncClient(timeout=30.0)
        
        self._langchain_model = None

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Google API requests."""
        return {
            "Content-Type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Google API error: {error_message}")

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        try:
            response = self.client.get(
                f"{self.base_url}/models?key={self.api_key}",
                headers=self._get_headers()
            )
            self._handle_error(response)
            
            models_data = response.json()
            return [
                Model(
                    id=model["name"].split("/")[-1],
                    owned_by="Google",
                    context_window=model.get("inputTokenLimit"),
                    type="language",
                )
                for model in models_data.get("models", [])
                if "generateContent" in model.get("supportedGenerationMethods", [])
            ]
        except Exception:
            # Fallback to known models if API call fails
            return [
                Model(id="gemini-2.0-flash", owned_by="Google", context_window=1000000, type="language"),
                Model(id="gemini-1.5-pro", owned_by="Google", context_window=2000000, type="language"),
                Model(id="gemini-1.5-flash", owned_by="Google", context_window=1000000, type="language"),
            ]

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "google"

    def _get_default_model(self) -> str:
        """Get the default model name.

        Returns:
            str: The default model name.
        """
        return "gemini-2.0-flash"

    def to_langchain(self):
        """Convert to a LangChain chat model.

        Returns:
            BaseChatModel: A LangChain chat model instance specific to the provider.

        Raises:
            ImportError: If langchain_google_genai is not installed.
        """
        try:
            from langchain_core.language_models.chat_models import BaseChatModel
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "Langchain integration requires langchain_google_genai. "
                "Install with: uv add langchain_google_genai or pip install langchain_google_genai"
            ) from e    

        if not self._langchain_model:
            # Ensure model name is a string
            model_name = self.get_model_name()
            if not model_name:
                raise ValueError("Model name must be set to use Langchain integration.")

            self._langchain_model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                # Removed streaming and google_api_key as they cause errors
            )
        return self._langchain_model

    def _format_messages(self, messages: List[Dict[str, str]]) -> tuple:
        """Return (formatted_messages, system_instruction) tuple.
        - formatted_messages: list of message dicts with only user/model roles
        - system_instruction: dict or None
        """
        formatted = []
        system_instruction = None
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                # Only the first system message is used
                if system_instruction is None:
                    system_instruction = {
                        "parts": [{"text": content}]
                    }
            elif role == "user":
                formatted.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                formatted.append({
                    "role": "model", 
                    "parts": [{"text": content}]
                })
        
        return formatted, system_instruction

    def _create_generation_config(self) -> Dict[str, Any]:
        """Create generation config for Google API."""
        config = {
            "temperature": float(self.temperature),
            "topP": float(self.top_p),
        }
        
        if self.max_tokens:
            config["maxOutputTokens"] = int(self.max_tokens)
            
        if self.structured:
            if not isinstance(self.structured, dict):
                raise TypeError("structured parameter must be a dictionary")
            structured_type = self.structured.get("type")
            if structured_type in ["json", "json_object"]:
                config["responseMimeType"] = "application/json"
                
        return config

    def _parse_sse_stream(self, response: httpx.Response) -> Generator[Dict[str, Any], None, None]:
        """Parse Server-Sent Events stream from Google chat completions."""
        for chunk in response.iter_text():
            for line in chunk.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data.strip() == "[DONE]":
                        break
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue

    async def _parse_sse_stream_async(self, response: httpx.Response) -> AsyncGenerator[Dict[str, Any], None]:
        """Parse Server-Sent Events stream from Google chat completions asynchronously."""
        async for chunk in response.aiter_text():
            for line in chunk.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data.strip() == "[DONE]":
                        return
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue

    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        stream: Optional[bool] = None,
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Send a chat completion request.

        Args:
            messages: List of messages in the conversation
            stream: Whether to stream the response

        Returns:
            Either a ChatCompletion or a Generator yielding ChatCompletionChunks if streaming
        """
        should_stream = stream if stream is not None else self.streaming
        formatted_messages, system_instruction = self._format_messages(messages)
        
        # Prepare request payload
        payload = {
            "contents": formatted_messages,
            "generationConfig": self._create_generation_config(),
        }
        
        if system_instruction:
            payload["system_instruction"] = system_instruction

        model_name = self.get_model_name()
        if should_stream:
            endpoint = "streamGenerateContent"
            url = f"{self.base_url}/models/{model_name}:{endpoint}?alt=sse&key={self.api_key}"
        else:
            endpoint = "generateContent"
            url = f"{self.base_url}/models/{model_name}:{endpoint}?key={self.api_key}"
        
        # Make HTTP request
        response = self.client.post(
            url,
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        if should_stream:
            def generate():
                for chunk_data in self._parse_sse_stream(response):
                    chunk = self._normalize_chunk(chunk_data)
                    if chunk:  # Only yield if chunk is not None
                        yield chunk
            return generate()
        
        response_data = response.json()
        return self._normalize_response(response_data)

    def _normalize_response(self, response_data: Dict[str, Any]) -> ChatCompletion:
        """Normalize Google response to our format."""
        candidate = response_data["candidates"][0]
        content = candidate["content"]
        text = content["parts"][0]["text"]
        
        finish_reason = "stop"
        if "finishReason" in candidate:
            finish_reason = candidate["finishReason"].lower()
            if finish_reason == "stop":
                finish_reason = "stop"
        
        return ChatCompletion(
            id=str(uuid.uuid4()),
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=text),
                    finish_reason=finish_reason,
                )
            ],
            created=int(time.time()),
            model=self.get_model_name(),
            provider=self.provider,
            usage=Usage(
                completion_tokens=response_data.get("usageMetadata", {}).get("candidatesTokenCount", 0),
                prompt_tokens=response_data.get("usageMetadata", {}).get("promptTokenCount", 0),
                total_tokens=response_data.get("usageMetadata", {}).get("totalTokenCount", 0),
            ),
        )

    def _normalize_chunk(self, chunk_data: Dict[str, Any]) -> Optional[ChatCompletionChunk]:
        """Normalize Google stream chunk to our format."""
        if "candidates" not in chunk_data or not chunk_data["candidates"]:
            return None
            
        candidate = chunk_data["candidates"][0]
        if "content" not in candidate or "parts" not in candidate["content"]:
            return None
            
        text = candidate["content"]["parts"][0].get("text", "")
        
        return ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant", content=text),
                    finish_reason=candidate.get("finishReason", "stop").lower() if "finishReason" in candidate else None,
                )
            ],
            model=self.get_model_name(),
            created=int(time.time()),
        )

    async def achat_complete(
        self,
        messages: List[Dict[str, str]],
        stream: Optional[bool] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Send an async chat completion request.

        Args:
            messages: List of messages in the conversation
            stream: Whether to stream the response

        Returns:
            Either a ChatCompletion or an AsyncGenerator yielding ChatCompletionChunks if streaming
        """
        should_stream = stream if stream is not None else self.streaming
        formatted_messages, system_instruction = self._format_messages(messages)
        
        # Prepare request payload
        payload = {
            "contents": formatted_messages,
            "generationConfig": self._create_generation_config(),
        }
        
        if system_instruction:
            payload["system_instruction"] = system_instruction

        model_name = self.get_model_name()
        if should_stream:
            endpoint = "streamGenerateContent"
            url = f"{self.base_url}/models/{model_name}:{endpoint}?alt=sse&key={self.api_key}"
        else:
            endpoint = "generateContent"
            url = f"{self.base_url}/models/{model_name}:{endpoint}?key={self.api_key}"
        
        # Make async HTTP request
        response = await self.async_client.post(
            url,
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        if should_stream:
            async def generate():
                async for chunk_data in self._parse_sse_stream_async(response):
                    chunk = self._normalize_chunk(chunk_data)
                    if chunk:  # Only yield if chunk is not None
                        yield chunk

            return generate()
        
        response_data = response.json()
        return self._normalize_response(response_data)
