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
    FunctionCall,
    Message,
    Model,
    StreamChoice,
    Tool,
    ToolCall,
    Usage,
)
from esperanto.common_types.validation import (
    validate_tool_calls as _validate_tool_calls,
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

        # Initialize HTTP clients with configurable timeout
        self._create_http_clients()

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

    def _get_models(self) -> List[Model]:
        """List all available models for this provider."""
        return [
            Model(
                id="gemini-2.0-flash",
                owned_by="Google",
                context_window=1000000,
            ),
            Model(
                id="gemini-1.5-pro",
                owned_by="Google",
                context_window=2000000,
            ),
            Model(
                id="gemini-1.5-flash",
                owned_by="Google",
                context_window=1000000,
            ),
            Model(
                id="gemini-pro",
                owned_by="Google",
                context_window=30720,
            ),
        ]

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "vertex"

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "gemini-2.0-flash"

    def _format_messages(self, messages: List[Dict[str, Any]]) -> tuple:
        """Return (formatted_messages, system_instruction) tuple.

        Converts messages to Vertex AI format, including handling:
        - System messages (extracted as system_instruction)
        - Tool result messages (converted to functionResponse format)
        - Assistant messages with tool_calls (converted to functionCall format)

        Args:
            messages: List of messages in Esperanto/OpenAI format.

        Returns:
            Tuple of (formatted_messages, system_instruction) for Vertex AI API.
        """
        formatted = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content")

            if role == "system":
                # Only the first system message is used
                if system_instruction is None:
                    system_instruction = {
                        "parts": [{"text": content}]
                    }

            elif role == "tool":
                # Convert tool result to Vertex AI's functionResponse format
                tool_call_id = msg.get("tool_call_id", "")
                # Try to parse content as JSON, otherwise use as string
                try:
                    response_content = json.loads(content) if content else {}
                except (json.JSONDecodeError, TypeError):
                    response_content = {"result": content}

                formatted.append({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": tool_call_id,  # Vertex uses name, we use the tool_call_id
                            "response": response_content,
                        }
                    }]
                })

            elif role == "assistant" and msg.get("tool_calls"):
                # Convert assistant tool_calls to Vertex AI's functionCall format
                parts: List[Dict[str, Any]] = []

                # Add text content if present
                if content:
                    parts.append({"text": content})

                # Add functionCall parts for each tool call
                for tc in msg["tool_calls"]:
                    # Handle both dict and ToolCall objects
                    if isinstance(tc, dict):
                        func_info = tc.get("function", {})
                        func_name = func_info.get("name", "")
                        func_args = func_info.get("arguments", "{}")
                    else:
                        # ToolCall object
                        func_name = tc.function.name
                        func_args = tc.function.arguments

                    # Parse arguments JSON to dict (Vertex expects dict, not JSON string)
                    try:
                        args_dict = json.loads(func_args) if isinstance(func_args, str) else func_args
                    except json.JSONDecodeError:
                        args_dict = {}

                    parts.append({
                        "functionCall": {
                            "name": func_name,
                            "args": args_dict,
                        }
                    })

                formatted.append({"role": "model", "parts": parts})

            elif role == "user":
                formatted.append({
                    "role": "user",
                    "parts": [{"text": content or ""}]
                })

            elif role == "assistant":
                formatted.append({
                    "role": "model",
                    "parts": [{"text": content or ""}]
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

    def _convert_tools_to_vertex(
        self, tools: Optional[List[Tool]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert Esperanto tools to Vertex AI format.

        Vertex AI uses the same format as Google GenAI with function_declarations
        wrapped in a tools array.

        Args:
            tools: List of Esperanto Tool objects.

        Returns:
            List of tools in Vertex AI format, or None if no tools provided.
        """
        if not tools:
            return None
        return [{
            "function_declarations": [
                {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters,
                }
                for tool in tools
            ]
        }]

    def _convert_tool_choice_to_vertex(
        self, tool_choice: Optional[Union[str, Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        """Convert tool_choice to Vertex AI format.

        Args:
            tool_choice: The tool choice setting (Esperanto/OpenAI format).

        Returns:
            Tool config in Vertex AI format, or None if not configured.
        """
        if tool_choice is None:
            return None
        if tool_choice == "auto":
            return {"function_calling_config": {"mode": "AUTO"}}
        if tool_choice == "required":
            return {"function_calling_config": {"mode": "ANY"}}
        if tool_choice == "none":
            return {"function_calling_config": {"mode": "NONE"}}
        if isinstance(tool_choice, dict):
            # Handle specific tool: {"type": "function", "function": {"name": "..."}}
            func_info = tool_choice.get("function", {})
            tool_name = func_info.get("name")
            if tool_name:
                return {
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": [tool_name],
                    }
                }
        return None

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
        """Normalize Vertex AI stream chunk to our format.

        Handles both text content and functionCall parts in streaming responses.
        """
        if "candidates" not in chunk_data or not chunk_data["candidates"]:
            return None

        candidate = chunk_data["candidates"][0]
        if "content" not in candidate or "parts" not in candidate["content"]:
            return None

        parts = candidate["content"]["parts"]

        # Extract text and tool calls from parts
        text_parts: List[str] = []
        tool_calls_data: List[Dict[str, Any]] = []

        for idx, part in enumerate(parts):
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                func_args = fc.get("args", {})
                arguments_str = json.dumps(func_args) if isinstance(func_args, dict) else str(func_args)

                tool_calls_data.append({
                    "index": idx,
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": arguments_str,
                    },
                })

        text_content = "".join(text_parts) if text_parts else None

        # Determine finish_reason
        finish_reason = None
        if "finishReason" in candidate:
            vertex_reason = candidate["finishReason"].upper()
            if vertex_reason == "STOP":
                finish_reason = "stop"
            elif vertex_reason == "MAX_TOKENS":
                finish_reason = "length"
            else:
                finish_reason = vertex_reason.lower()

            # If we had tool calls, set finish_reason to tool_calls
            if tool_calls_data:
                finish_reason = "tool_calls"

        return ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(
                        role="assistant",
                        content=text_content,
                        tool_calls=tool_calls_data if tool_calls_data else None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            model=self.get_model_name(),
            created=int(time.time()),
        )

    def chat_complete(
        self,
        messages: List[Dict[str, Any]],
        stream: Optional[bool] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        validate_tool_calls: bool = False,
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Send a chat completion request.

        Args:
            messages: List of messages in the conversation. Messages can include
                tool call results with role="tool" and tool_call_id.
            stream: Whether to stream the response. If None, uses the instance's
                streaming setting.
            tools: List of tools the model can call. If None, uses instance tools.
            tool_choice: Controls tool usage. Values:
                - "auto": Model decides whether to call tools (default)
                - "required": Model must call at least one tool
                - "none": Model cannot call tools
                - {"type": "function", "function": {"name": "..."}}: Force specific tool
            parallel_tool_calls: Whether to allow multiple tool calls in one response.
                Note: Vertex AI doesn't have a direct parallel_tool_calls parameter,
                this is included for interface consistency but may not affect behavior.
            validate_tool_calls: If True, validate tool call arguments against the
                tool's JSON schema. Raises ToolCallValidationError on validation
                failure. Requires jsonschema package.

        Returns:
            Either a ChatCompletion or a Generator yielding ChatCompletionChunks
            if streaming. When the model calls tools, the response message will
            have tool_calls populated.
        """
        # Warn if validate_tool_calls is used with streaming
        self._warn_if_validate_with_streaming(validate_tool_calls, stream)

        should_stream = stream if stream is not None else self.streaming

        # Resolve tool configuration
        resolved_tools = self._resolve_tools(tools)
        resolved_tool_choice = self._resolve_tool_choice(tool_choice)
        # Note: parallel_tool_calls is resolved but Vertex AI doesn't support it directly
        _ = self._resolve_parallel_tool_calls(parallel_tool_calls)

        formatted_messages, system_instruction = self._format_messages(messages)

        # Prepare request payload
        payload: Dict[str, Any] = {
            "contents": formatted_messages,
        }

        # Add generation config if provided
        generation_config = self._create_generation_config()
        if generation_config:
            payload["generationConfig"] = generation_config

        if system_instruction:
            payload["system_instruction"] = system_instruction

        # Add tools if provided
        if resolved_tools:
            payload["tools"] = self._convert_tools_to_vertex(resolved_tools)

        # Add tool_config if tool_choice is specified
        if resolved_tool_choice is not None:
            tool_config = self._convert_tool_choice_to_vertex(resolved_tool_choice)
            if tool_config:
                payload["tool_config"] = tool_config

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
                chunk = self._normalize_chunk(response_data)
                if chunk:
                    yield chunk

            return generate()

        response_data = response.json()
        result = self._normalize_response(response_data)

        # Validate tool calls if requested
        if validate_tool_calls and resolved_tools:
            for choice in result.choices:
                if choice.message.tool_calls:
                    _validate_tool_calls(choice.message.tool_calls, resolved_tools)

        return result

    def _normalize_response(self, response_data: Dict[str, Any]) -> ChatCompletion:
        """Normalize Vertex AI response to our format.

        Handles both text content and functionCall parts from Vertex AI responses.
        """
        candidate = response_data["candidates"][0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        # Extract text content and tool calls from parts
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []

        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                # Convert Vertex AI functionCall to Esperanto ToolCall
                fc = part["functionCall"]
                func_args = fc.get("args", {})
                # Convert args dict to JSON string (Esperanto uses JSON string)
                arguments_str = json.dumps(func_args) if isinstance(func_args, dict) else str(func_args)

                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:12]}",  # Vertex AI doesn't provide ID
                        type="function",
                        function=FunctionCall(
                            name=fc.get("name", ""),
                            arguments=arguments_str,
                        ),
                    )
                )

        # Combine text parts
        text_content = "".join(text_parts) if text_parts else None

        # Map Vertex AI finishReason to standard finish_reason
        finish_reason = "stop"
        if "finishReason" in candidate:
            vertex_reason = candidate["finishReason"].upper()
            if vertex_reason == "STOP":
                finish_reason = "stop"
            elif vertex_reason == "MAX_TOKENS":
                finish_reason = "length"
            elif vertex_reason in ("SAFETY", "RECITATION", "OTHER"):
                finish_reason = "stop"
            else:
                finish_reason = vertex_reason.lower()

        # If we have tool calls, set finish_reason appropriately
        if tool_calls:
            finish_reason = "tool_calls"

        return ChatCompletion(
            id=str(uuid.uuid4()),
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=text_content,
                        tool_calls=tool_calls if tool_calls else None,
                    ),
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
        messages: List[Dict[str, Any]],
        stream: Optional[bool] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        validate_tool_calls: bool = False,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Send an async chat completion request.

        Args:
            messages: List of messages in the conversation. Messages can include
                tool call results with role="tool" and tool_call_id.
            stream: Whether to stream the response. If None, uses the instance's
                streaming setting.
            tools: List of tools the model can call. If None, uses instance tools.
            tool_choice: Controls tool usage. Values:
                - "auto": Model decides whether to call tools (default)
                - "required": Model must call at least one tool
                - "none": Model cannot call tools
                - {"type": "function", "function": {"name": "..."}}: Force specific tool
            parallel_tool_calls: Whether to allow multiple tool calls in one response.
                Note: Vertex AI doesn't have a direct parallel_tool_calls parameter,
                this is included for interface consistency but may not affect behavior.
            validate_tool_calls: If True, validate tool call arguments against the
                tool's JSON schema. Raises ToolCallValidationError on validation
                failure. Requires jsonschema package.

        Returns:
            Either a ChatCompletion or an AsyncGenerator yielding ChatCompletionChunks
            if streaming. When the model calls tools, the response message will
            have tool_calls populated.
        """
        # Warn if validate_tool_calls is used with streaming
        self._warn_if_validate_with_streaming(validate_tool_calls, stream)

        should_stream = stream if stream is not None else self.streaming

        # Resolve tool configuration
        resolved_tools = self._resolve_tools(tools)
        resolved_tool_choice = self._resolve_tool_choice(tool_choice)
        # Note: parallel_tool_calls is resolved but Vertex AI doesn't support it directly
        _ = self._resolve_parallel_tool_calls(parallel_tool_calls)

        formatted_messages, system_instruction = self._format_messages(messages)

        # Prepare request payload
        payload: Dict[str, Any] = {
            "contents": formatted_messages,
        }

        # Add generation config if provided
        generation_config = self._create_generation_config()
        if generation_config:
            payload["generationConfig"] = generation_config

        if system_instruction:
            payload["system_instruction"] = system_instruction

        # Add tools if provided
        if resolved_tools:
            payload["tools"] = self._convert_tools_to_vertex(resolved_tools)

        # Add tool_config if tool_choice is specified
        if resolved_tool_choice is not None:
            tool_config = self._convert_tool_choice_to_vertex(resolved_tool_choice)
            if tool_config:
                payload["tool_config"] = tool_config

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
                chunk = self._normalize_chunk(response_data)
                if chunk:
                    yield chunk

            return generate()

        response_data = response.json()
        result = self._normalize_response(response_data)

        # Validate tool calls if requested
        if validate_tool_calls and resolved_tools:
            for choice in result.choices:
                if choice.message.tool_calls:
                    _validate_tool_calls(choice.message.tool_calls, resolved_tools)

        return result

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