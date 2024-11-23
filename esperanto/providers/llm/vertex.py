import os
from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

from esperanto.base.types import LanguageModel


class VertexAILanguageModel(LanguageModel):
    """Vertex AI language model implementation."""

    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any] = {},
    ):
        if not model_name:
            raise ValueError(
                "model_name must be specified for Vertex AI language model"
            )
        super().__init__(model_name, config)
        self.max_tokens = self.config.get("max_tokens", 850)
        self.temperature = self.config.get("temperature", 1.0)
        self.streaming = self.config.get("streaming", True)
        self.top_p = self.config.get("top_p", 0.9)

        # Handle project and location
        self.project = self.config.get("project") or os.environ.get(
            "VERTEX_PROJECT", "no-project"
        )
        self.location = self.config.get("location") or os.environ.get(
            "VERTEX_LOCATION", "us-central1"
        )

    @property
    def provider(self) -> str:
        return "vertex"

    def validate_config(self) -> None:
        """Validate Vertex AI configuration."""
        if not self.model_name:
            raise ValueError(
                "model_name must be specified for Vertex AI language model"
            )
        if not self.project:
            raise ValueError(
                "project must be specified in config or VERTEX_PROJECT environment variable"
            )
        if not self.location:
            raise ValueError(
                "location must be specified in config or VERTEX_LOCATION environment variable"
            )

    def to_langchain(self) -> BaseChatModel:
        """Convert to a LangChain chat model."""
        return ChatVertexAI(
            model_name=self.model_name,
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            project=self.project,
            location=self.location,
            streaming=self.streaming,
        )


class VertexAnthropicLanguageModel(LanguageModel):
    """Vertex AI Anthropic language model implementation."""

    def __init__(
        self,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        if not model_name:
            raise ValueError(
                "model_name must be specified for Vertex Anthropic language model"
            )
        super().__init__(model_name, config)
        self.max_tokens = self.config.get("max_tokens", 850)
        self.temperature = self.config.get("temperature", 1.0)
        self.streaming = self.config.get("streaming", True)
        self.top_p = self.config.get("top_p", 0.9)

        # Handle project and location
        self.project = self.config.get("project") or os.environ.get(
            "VERTEX_PROJECT", "no-project"
        )
        self.location = self.config.get("location") or os.environ.get(
            "VERTEX_LOCATION", "us-central1"
        )

    @property
    def provider(self) -> str:
        return "vertex_anthropic"

    def validate_config(self) -> None:
        """Validate Vertex Anthropic configuration."""
        if not self.model_name:
            raise ValueError(
                "model_name must be specified for Vertex Anthropic language model"
            )
        if not self.project:
            raise ValueError(
                "project must be specified in config or VERTEX_PROJECT environment variable"
            )
        if not self.location:
            raise ValueError(
                "location must be specified in config or VERTEX_LOCATION environment variable"
            )

    def to_langchain(self) -> BaseChatModel:
        """Convert to a LangChain chat model."""
        return ChatAnthropicVertex(
            model_name=self.model_name,
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            project=self.project,
            location=self.location,
            streaming=self.streaming,
        )
