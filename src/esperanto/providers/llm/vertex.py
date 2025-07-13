"""Google Vertex AI language model provider."""

import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import (
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


@dataclass
class VertexLanguageModel(LanguageModel):
    """Google Vertex AI language model implementation."""

    vertex_project: Optional[str] = None
    vertex_location: Optional[str] = None

    def __post_init__(self):
        """Initialize HTTP clients and authentication."""
        super().__post_init__()

        # Get project and location
        self.project_id = self.vertex_project or os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = self.vertex_location or os.getenv("VERTEX_LOCATION", "us-central1")
        
        if not self.project_id:
            raise ValueError(
                "Google Cloud project ID not found. Please set VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT environment variable."
            )

        # Set base URL for Vertex AI
        self.base_url = f"https://{self.location}-aiplatform.googleapis.com/v1"

        # Initialize HTTP clients
        self.client = httpx.Client(timeout=30.0)
        self.async_client = httpx.AsyncClient(timeout=30.0)

        # Cache for access token
        self._access_token = None
        self._token_expiry = 0

    def _get_access_token(self) -> str:
        """Get OAuth 2.0 access token for Google Cloud APIs."""
        current_time = time.time()
        
        # Check if token is still valid (with 5-minute buffer)
        if self._access_token and current_time < (self._token_expiry - 300):
            return self._access_token
            
        try:
            # Use gcloud to get access token
            result = subprocess.run(
                ["gcloud", "auth", "application-default", "print-access-token"],
                capture_output=True,
                text=True,
                check=True
            )
            self._access_token = result.stdout.strip()
            # Tokens typically expire in 1 hour
            self._token_expiry = current_time + 3600
            return self._access_token
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to get access token. Make sure you're authenticated with 'gcloud auth application-default login': {e}"
            )

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Vertex AI API requests."""
        return {
            "Authorization": f"Bearer {self._get_access_token()}",
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
            raise RuntimeError(f"Vertex AI API error: {error_message}")

    def _get_model_path(self) -> str:
        """Get the full model path for Vertex AI."""
        model_name = self.get_model_name()
        return f"projects/{self.project_id}/locations/{self.location}/publishers/google/models/{model_name}"

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        return [
            Model(
                id="gemini-2.0-flash",
                owned_by="Google",
                context_window=1000000,
                type="language",
            ),
            Model(
                id="gemini-1.5-pro",
                owned_by="Google",
                context_window=2000000,
                type="language",
            ),
            Model(
                id="gemini-1.5-flash",
                owned_by="Google",
                context_window=1000000,
                type="language",
            ),
            Model(
                id="gemini-pro",
                owned_by="Google",
                context_window=30720,
                type="language",
            ),
        ]

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "vertex"

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "gemini-2.0-flash"

    def _format_messages(self, messages: List[Dict[str, str]]) -> tuple:
        """Return (formatted_messages, system_instruction) tuple."""
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
        """Create generation config for Vertex AI."""
        config = {}
        
        if self.temperature is not None:
            config["temperature"] = float(self.temperature)
            
        if self.top_p is not None:
            config["topP"] = float(self.top_p)
        
        if self.max_tokens:
            config["maxOutputTokens"] = int(self.max_tokens)
            
        return config

    def _parse_sse_stream(self, response: httpx.Response) -> Generator[Dict[str, Any], None, None]:
        """Parse Server-Sent Events stream from Vertex AI."""
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
        """Parse Server-Sent Events stream from Vertex AI asynchronously."""
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

    def _normalize_chunk(self, chunk_data: Dict[str, Any]) -> Optional[ChatCompletionChunk]:
        """Normalize Vertex AI stream chunk to our format."""
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

    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        stream: Optional[bool] = None,
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Send a chat completion request."""
        should_stream = stream if stream is not None else self.streaming
        formatted_messages, system_instruction = self._format_messages(messages)
        
        # Prepare request payload
        payload = {
            "contents": formatted_messages,
        }
        
        # Add generation config if provided
        generation_config = self._create_generation_config()
        if generation_config:
            payload["generationConfig"] = generation_config
        
        if system_instruction:
            payload["system_instruction"] = system_instruction

        model_path = self._get_model_path()
        
        # Use regular endpoint for both streaming and non-streaming
        # Vertex AI REST API streaming may not be supported the same way
        url = f"{self.base_url}/{model_path}:generateContent"
        
        # Make HTTP request
        response = self.client.post(
            url,
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        if should_stream:
            # Vertex AI REST API doesn't support true streaming like other providers
            # So we'll simulate streaming by returning the complete response as a single chunk
            def generate():
                response_data = response.json()
                # Create a single chunk from the complete response
                candidate = response_data["candidates"][0]
                content = candidate["content"]
                text = content["parts"][0]["text"]
                
                chunk = ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(role="assistant", content=text),
                            finish_reason="stop",
                        )
                    ],
                    model=self.get_model_name(),
                    created=int(time.time()),
                )
                yield chunk
                        
            return generate()
        
        response_data = response.json()
        return self._normalize_response(response_data)

    def _normalize_response(self, response_data: Dict[str, Any]) -> ChatCompletion:
        """Normalize Vertex AI response to our format."""
        candidate = response_data["candidates"][0]
        content = candidate["content"]
        text = content["parts"][0]["text"]
        
        finish_reason = "stop"
        if "finishReason" in candidate:
            finish_reason = candidate["finishReason"].lower()
        
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

    async def achat_complete(
        self,
        messages: List[Dict[str, str]],
        stream: Optional[bool] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Send an async chat completion request."""
        should_stream = stream if stream is not None else self.streaming
        formatted_messages, system_instruction = self._format_messages(messages)
        
        # Prepare request payload
        payload = {
            "contents": formatted_messages,
        }
        
        # Add generation config if provided
        generation_config = self._create_generation_config()
        if generation_config:
            payload["generationConfig"] = generation_config
        
        if system_instruction:
            payload["system_instruction"] = system_instruction

        model_path = self._get_model_path()
        
        # Use regular endpoint for both streaming and non-streaming
        # Vertex AI REST API streaming may not be supported the same way
        url = f"{self.base_url}/{model_path}:generateContent"
        
        # Make async HTTP request
        response = await self.async_client.post(
            url,
            headers=self._get_headers(),
            json=payload
        )
        self._handle_error(response)

        if should_stream:
            # Vertex AI REST API doesn't support true streaming like other providers
            # So we'll simulate streaming by returning the complete response as a single chunk
            async def generate():
                response_data = response.json()
                # Create a single chunk from the complete response
                candidate = response_data["candidates"][0]
                content = candidate["content"]
                text = content["parts"][0]["text"]
                
                chunk = ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(role="assistant", content=text),
                            finish_reason="stop",
                        )
                    ],
                    model=self.get_model_name(),
                    created=int(time.time()),
                )
                yield chunk
                        
            return generate()
        
        response_data = response.json()
        return self._normalize_response(response_data)

    def to_langchain(self):
        """Convert to a LangChain chat model."""
        try:
            from langchain_google_vertexai import ChatVertexAI
        except ImportError as e:
            raise ImportError(
                "Langchain integration requires langchain_google_vertexai. "
                "Install with: uv add langchain_google_vertexai or pip install langchain_google_vertexai"
            ) from e

        # Ensure model name is set
        model_name = self.get_model_name()
        if not model_name:
            raise ValueError("Model name must be set to use Langchain integration.")

        return ChatVertexAI(
            model_name=model_name,
            project=self.project_id,
            location=self.location,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            top_p=self.top_p,
        )