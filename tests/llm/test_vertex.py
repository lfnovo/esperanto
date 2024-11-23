"""Tests for Vertex AI language model implementations."""

import os

import pytest
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

from esperanto.providers.llm import (
    VertexAILanguageModel,
    VertexAnthropicLanguageModel,
)


@pytest.fixture
def vertex_model():
    """Vertex AI language model fixture."""
    return VertexAILanguageModel(
        model_name="gemini-pro",
        config={
            "temperature": 0.7,
            "project": "test-project",
            "location": "us-central1",
        },
    )


@pytest.fixture
def vertex_anthropic_model():
    """Vertex AI Anthropic language model fixture."""
    return VertexAnthropicLanguageModel(
        model_name="claude-3-opus",
        config={
            "temperature": 0.7,
            "project": "test-project",
            "location": "us-central1",
        },
    )


class TestVertexAILanguageModel:
    """Test suite for Vertex AI language model."""

    async def test_initialization(self, vertex_model):
        """Test model initialization with valid config."""
        assert vertex_model.model_name == "gemini-pro"
        assert vertex_model.temperature == 0.7
        assert vertex_model.streaming is True
        assert vertex_model.top_p == 0.9
        assert vertex_model.max_tokens == 850
        assert vertex_model.project == "test-project"
        assert vertex_model.location == "us-central1"

    async def test_initialization_invalid_config(self):
        """Test model initialization with invalid config."""
        with pytest.raises(ValueError):
            VertexAILanguageModel(model_name="")

    async def test_to_langchain(self, vertex_model):
        """Test conversion to LangChain model."""
        langchain_model = vertex_model.to_langchain()
        assert isinstance(langchain_model, ChatVertexAI)
        assert langchain_model.model_name == "gemini-pro"
        assert langchain_model.temperature == 0.7
        assert langchain_model.streaming is True
        assert langchain_model.top_p == 0.9
        assert langchain_model.max_output_tokens == 850
        assert langchain_model.project == "test-project"
        assert langchain_model.location == "us-central1"

    async def test_validate_config_success(self, vertex_model):
        """Test config validation with valid config."""
        vertex_model.validate_config()  # Should not raise any exception

    async def test_validate_config_failure(self):
        """Test config validation with invalid config."""
        with pytest.raises(ValueError, match="model_name must be specified for Vertex AI language model"):
            VertexAILanguageModel(model_name="")

    async def test_provider_name(self, vertex_model):
        """Test provider name is correct."""
        assert vertex_model.provider == "vertex"

    async def test_model_parameters(self, vertex_model):
        """Test model parameters are correctly set."""
        assert vertex_model.model_name == "gemini-pro"
        assert vertex_model.temperature == 0.7
        assert vertex_model.streaming is True
        assert vertex_model.top_p == 0.9
        assert vertex_model.max_tokens == 850
        assert vertex_model.project == "test-project"
        assert vertex_model.location == "us-central1"

    async def test_project_validation(self):
        """Test project validation."""
        # Temporarily unset VERTEX_PROJECT
        original_project = os.environ.get("VERTEX_PROJECT")
        if original_project:
            del os.environ["VERTEX_PROJECT"]
        
        try:
            model = VertexAILanguageModel(
                model_name="gemini-pro",
                config={"project": None}
            )
            assert model.project == "no-project"
        finally:
            if original_project:
                os.environ["VERTEX_PROJECT"] = original_project

    async def test_location_validation(self):
        """Test location validation."""
        # Temporarily unset VERTEX_LOCATION
        original_location = os.environ.get("VERTEX_LOCATION")
        if original_location:
            del os.environ["VERTEX_LOCATION"]
        
        try:
            model = VertexAILanguageModel(
                model_name="gemini-pro",
                config={"location": None}
            )
            assert model.location == "us-central1"
        finally:
            if original_location:
                os.environ["VERTEX_LOCATION"] = original_location


class TestVertexAnthropicLanguageModel:
    """Test suite for Vertex Anthropic language model."""

    async def test_initialization(self, vertex_anthropic_model):
        """Test model initialization with valid config."""
        assert vertex_anthropic_model.model_name == "claude-3-opus"
        assert vertex_anthropic_model.temperature == 0.7
        assert vertex_anthropic_model.streaming is True
        assert vertex_anthropic_model.top_p == 0.9
        assert vertex_anthropic_model.max_tokens == 850
        assert vertex_anthropic_model.project == "test-project"
        assert vertex_anthropic_model.location == "us-central1"

    async def test_initialization_invalid_config(self):
        """Test model initialization with invalid config."""
        with pytest.raises(ValueError):
            VertexAnthropicLanguageModel(model_name="")

    async def test_to_langchain(self, vertex_anthropic_model):
        """Test conversion to LangChain model."""
        langchain_model = vertex_anthropic_model.to_langchain()
        assert isinstance(langchain_model, ChatAnthropicVertex)
        assert langchain_model.model_name == "claude-3-opus"
        assert langchain_model.temperature == 0.7
        assert langchain_model.streaming is True
        assert langchain_model.top_p == 0.9
        assert langchain_model.max_output_tokens == 850
        assert langchain_model.project == "test-project"
        assert langchain_model.location == "us-central1"

    async def test_validate_config_success(self, vertex_anthropic_model):
        """Test config validation with valid config."""
        vertex_anthropic_model.validate_config()  # Should not raise any exception

    async def test_validate_config_failure(self):
        """Test config validation with invalid config."""
        with pytest.raises(ValueError, match="model_name must be specified for Vertex Anthropic language model"):
            VertexAnthropicLanguageModel(model_name="")

    async def test_provider_name(self, vertex_anthropic_model):
        """Test provider name is correct."""
        assert vertex_anthropic_model.provider == "vertex_anthropic"

    async def test_model_parameters(self, vertex_anthropic_model):
        """Test model parameters are correctly set."""
        assert vertex_anthropic_model.model_name == "claude-3-opus"
        assert vertex_anthropic_model.temperature == 0.7
        assert vertex_anthropic_model.streaming is True
        assert vertex_anthropic_model.top_p == 0.9
        assert vertex_anthropic_model.max_tokens == 850
        assert vertex_anthropic_model.project == "test-project"
        assert vertex_anthropic_model.location == "us-central1"

    async def test_project_validation(self):
        """Test project validation."""
        # Temporarily unset VERTEX_PROJECT
        original_project = os.environ.get("VERTEX_PROJECT")
        if original_project:
            del os.environ["VERTEX_PROJECT"]
        
        try:
            model = VertexAnthropicLanguageModel(
                model_name="claude-3-opus",
                config={"project": None}
            )
            assert model.project == "no-project"
        finally:
            if original_project:
                os.environ["VERTEX_PROJECT"] = original_project

    async def test_location_validation(self):
        """Test location validation."""
        # Temporarily unset VERTEX_LOCATION
        original_location = os.environ.get("VERTEX_LOCATION")
        if original_location:
            del os.environ["VERTEX_LOCATION"]
        
        try:
            model = VertexAnthropicLanguageModel(
                model_name="claude-3-opus",
                config={"location": None}
            )
            assert model.location == "us-central1"
        finally:
            if original_location:
                os.environ["VERTEX_LOCATION"] = original_location
