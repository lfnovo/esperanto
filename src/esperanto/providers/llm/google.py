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
        base_host = os.getenv("GEMINI_API_BASE_URL") or "https://generativelanguage.googleapis.com"
        self.base_url = f"{base_host}/v1beta"

        # Initialize HTTP clients with configurable timeout
        self._create_http_clients()
        
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

    def _get_models(self) -> List[Model]:
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
                )
                for model in models_data.get("models", [])
            ]
        except Exception:
            # Fallback to known models if API call fails
            return [
                Model(id="gemini-2.0-flash", owned_by="Google", context_window=1000000),
                Model(id="gemini-1.5-pro", owned_by="Google", context_window=2000000),
                Model(id="gemini-1.5-flash", owned_by="Google", context_window=1000000),
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
                google_api_key=self.api_key,
            )
        return self._langchain_model

    def _format_messages(self, messages: List[Dict[str, Any]]) -> tuple:
        """Return (formatted_messages, system_instruction) tuple.

        Converts messages to Google format, including handling:
        - System messages (extracted as system_instruction)
        - Tool result messages (converted to functionResponse format)
        - Assistant messages with tool_calls (converted to functionCall format)

        Args:
            messages: List of messages in Esperanto/OpenAI format.

        Returns:
            Tuple of (formatted_messages, system_instruction) for Google API.
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
                # Convert tool result to Google's functionResponse format
                tool_call_id = msg.get("tool_call_id", "")
                # Try to parse content as JSON, otherwise use as string
                # Google's response field MUST be a Struct (dict), not a primitive
                try:
                    response_content = json.loads(content) if content else {}
                except (json.JSONDecodeError, TypeError):
                    response_content = {"result": content}

                # Ensure response is always a dict (Google requires Struct)
                if not isinstance(response_content, dict):
                    response_content = {"result": response_content}

                formatted.append({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": tool_call_id,  # Google uses name, we use the tool_call_id
                            "response": response_content,
                        }
                    }]
                })

            elif role == "assistant" and msg.get("tool_calls"):
                # Convert assistant tool_calls to Google's functionCall format
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

                    # Parse arguments JSON to dict (Google expects dict, not JSON string)
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

    def _clean_schema_for_google(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Remove fields from JSON schema that Google doesn't support.

        Google's API doesn't support certain JSON Schema fields like
        'additionalProperties', 'title', '$schema', etc.

        Args:
            schema: JSON Schema dict.

        Returns:
            Cleaned schema dict with unsupported fields removed.
        """
        if not isinstance(schema, dict):
            return schema

        # Fields Google doesn't support in tool parameters
        unsupported_fields = {
            "additionalProperties",
            "title",
            "$schema",
            "default",
        }

        cleaned = {}
        for key, value in schema.items():
            if key in unsupported_fields:
                continue
            elif key == "properties" and isinstance(value, dict):
                # Recursively clean nested properties
                cleaned[key] = {
                    k: self._clean_schema_for_google(v)
                    for k, v in value.items()
                }
            elif key == "items" and isinstance(value, dict):
                # Clean array item schemas
                cleaned[key] = self._clean_schema_for_google(value)
            else:
                cleaned[key] = value

        return cleaned

    def _convert_tools_to_google(
        self, tools: Optional[List[Tool]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert Esperanto tools to Google format.

        Google/Gemini uses a different structure with function_declarations
        wrapped in a tools array.

        Args:
            tools: List of Esperanto Tool objects.

        Returns:
            List of tools in Google API format, or None if no tools provided.
        """
        if not tools:
            return None
        return [{
            "function_declarations": [
                {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": self._clean_schema_for_google(
                        tool.function.parameters or {}
                    ),
                }
                for tool in tools
            ]
        }]

    def _convert_tool_choice_to_google(
        self, tool_choice: Optional[Union[str, Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        """Convert tool_choice to Google format.

        Args:
            tool_choice: The tool choice setting (Esperanto/OpenAI format).

        Returns:
            Tool config in Google format, or None if not configured.
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
                Note: Google/Gemini doesn't have a direct parallel_tool_calls parameter,
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
        # Note: parallel_tool_calls is resolved but Google doesn't support it directly
        _ = self._resolve_parallel_tool_calls(parallel_tool_calls)

        formatted_messages, system_instruction = self._format_messages(messages)

        # Prepare request payload
        payload: Dict[str, Any] = {
            "contents": formatted_messages,
            "generationConfig": self._create_generation_config(),
        }

        if system_instruction:
            payload["system_instruction"] = system_instruction

        # Add tools if provided
        if resolved_tools:
            payload["tools"] = self._convert_tools_to_google(resolved_tools)

        # Add tool_config if tool_choice is specified
        if resolved_tool_choice is not None:
            tool_config = self._convert_tool_choice_to_google(resolved_tool_choice)
            if tool_config:
                payload["tool_config"] = tool_config

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
        result = self._normalize_response(response_data)

        # Validate tool calls if requested
        if validate_tool_calls and resolved_tools:
            for choice in result.choices:
                if choice.message.tool_calls:
                    _validate_tool_calls(choice.message.tool_calls, resolved_tools)

        return result

    def _normalize_response(self, response_data: Dict[str, Any]) -> ChatCompletion:
        """Normalize Google response to our format.

        Handles both text content and functionCall parts from Google responses.
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
                # Convert Google functionCall to Esperanto ToolCall
                fc = part["functionCall"]
                func_args = fc.get("args", {})
                # Convert args dict to JSON string (Esperanto uses JSON string)
                arguments_str = json.dumps(func_args) if isinstance(func_args, dict) else str(func_args)

                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:12]}",  # Google doesn't provide ID
                        type="function",
                        function=FunctionCall(
                            name=fc.get("name", ""),
                            arguments=arguments_str,
                        ),
                    )
                )

        # Combine text parts
        text_content = "".join(text_parts) if text_parts else None

        # Map Google finishReason to standard finish_reason
        finish_reason = "stop"
        if "finishReason" in candidate:
            google_reason = candidate["finishReason"].upper()
            if google_reason == "STOP":
                finish_reason = "stop"
            elif google_reason == "MAX_TOKENS":
                finish_reason = "length"
            elif google_reason in ("SAFETY", "RECITATION", "OTHER"):
                finish_reason = "stop"
            else:
                finish_reason = google_reason.lower()

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

    def _normalize_chunk(self, chunk_data: Dict[str, Any]) -> Optional[ChatCompletionChunk]:
        """Normalize Google stream chunk to our format.

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
            google_reason = candidate["finishReason"].upper()
            if google_reason == "STOP":
                finish_reason = "stop"
            elif google_reason == "MAX_TOKENS":
                finish_reason = "length"
            else:
                finish_reason = google_reason.lower()

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
                Note: Google/Gemini doesn't have a direct parallel_tool_calls parameter,
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
        # Note: parallel_tool_calls is resolved but Google doesn't support it directly
        _ = self._resolve_parallel_tool_calls(parallel_tool_calls)

        formatted_messages, system_instruction = self._format_messages(messages)

        # Prepare request payload
        payload: Dict[str, Any] = {
            "contents": formatted_messages,
            "generationConfig": self._create_generation_config(),
        }

        if system_instruction:
            payload["system_instruction"] = system_instruction

        # Add tools if provided
        if resolved_tools:
            payload["tools"] = self._convert_tools_to_google(resolved_tools)

        # Add tool_config if tool_choice is specified
        if resolved_tool_choice is not None:
            tool_config = self._convert_tool_choice_to_google(resolved_tool_choice)
            if tool_config:
                payload["tool_config"] = tool_config

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
        result = self._normalize_response(response_data)

        # Validate tool calls if requested
        if validate_tool_calls and resolved_tools:
            for choice in result.choices:
                if choice.message.tool_calls:
                    _validate_tool_calls(choice.message.tool_calls, resolved_tools)

        return result
