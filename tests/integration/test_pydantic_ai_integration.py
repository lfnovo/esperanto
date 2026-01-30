"""Integration tests for Pydantic AI integration.

These tests verify that Esperanto models work correctly with Pydantic AI agents.
"""

import pytest
from pydantic_ai import Agent
from pydantic_ai.models import Model

from esperanto import AIFactory
from esperanto.integrations.pydantic_ai import EsperantoPydanticModel
from esperanto.providers.llm.anthropic import AnthropicLanguageModel
from esperanto.providers.llm.openai import OpenAILanguageModel


class TestBasicAgentCreation:
    """Test basic agent creation with various providers."""

    def test_openai_to_pydantic_ai(self):
        """Test OpenAI model conversion to Pydantic AI."""
        model = OpenAILanguageModel(
            api_key="test-key",
            model_name="gpt-4o",
            temperature=0.7,
            max_tokens=100,
        )

        pydantic_model = model.to_pydantic_ai()

        assert isinstance(pydantic_model, Model)
        assert isinstance(pydantic_model, EsperantoPydanticModel)
        assert pydantic_model.model_name == "gpt-4o"
        assert pydantic_model.system == "openai"

    def test_anthropic_to_pydantic_ai(self):
        """Test Anthropic model conversion to Pydantic AI."""
        model = AnthropicLanguageModel(
            api_key="test-key",
            model_name="claude-sonnet-4-20250514",
            temperature=0.7,
            max_tokens=100,
        )

        pydantic_model = model.to_pydantic_ai()

        assert isinstance(pydantic_model, Model)
        assert pydantic_model.model_name == "claude-sonnet-4-20250514"
        assert pydantic_model.system == "anthropic"

    def test_factory_created_model_to_pydantic_ai(self):
        """Test AIFactory-created model conversion."""
        model = AIFactory.create_language(
            "openai",
            "gpt-4o-mini",
            config={"api_key": "test-key"}
        )

        pydantic_model = model.to_pydantic_ai()

        assert isinstance(pydantic_model, Model)
        assert pydantic_model.model_name == "gpt-4o-mini"

    def test_agent_creation_with_esperanto_model(self):
        """Test that an Agent can be created with an Esperanto model."""
        model = AIFactory.create_language(
            "openai",
            "gpt-4o",
            config={"api_key": "test-key"}
        )

        # This should not raise
        agent = Agent(model.to_pydantic_ai())

        assert agent is not None


class TestProviderSwitching:
    """Test provider switching capability."""

    def test_same_code_different_providers(self):
        """Test that the same code works with different providers."""
        providers = [
            ("openai", "gpt-4o"),
            ("anthropic", "claude-sonnet-4-20250514"),
            ("groq", "llama-3.3-70b-versatile"),
        ]

        for provider, model_name in providers:
            model = AIFactory.create_language(
                provider,
                model_name,
                config={"api_key": "test-key"}
            )
            pydantic_model = model.to_pydantic_ai()

            # Verify basic properties
            assert pydantic_model.model_name == model_name
            assert pydantic_model.system == provider

            # Verify Agent can be created
            agent = Agent(pydantic_model)
            assert agent is not None


class TestConfigurationPreservation:
    """Test that Esperanto configuration is preserved."""

    def test_model_name_preserved(self):
        """Test that model name is preserved through conversion."""
        model = OpenAILanguageModel(
            api_key="test-key",
            model_name="gpt-4o-mini",
        )

        pydantic_model = model.to_pydantic_ai()

        assert pydantic_model.model_name == "gpt-4o-mini"

    def test_provider_preserved(self):
        """Test that provider is preserved through conversion."""
        model = AnthropicLanguageModel(
            api_key="test-key",
            model_name="claude-sonnet-4-20250514",
        )

        pydantic_model = model.to_pydantic_ai()

        assert pydantic_model.system == "anthropic"

    def test_esperanto_model_reference_preserved(self):
        """Test that the Esperanto model reference is preserved."""
        model = OpenAILanguageModel(
            api_key="test-key",
            model_name="gpt-4o",
            temperature=0.5,
            max_tokens=200,
        )

        pydantic_model = model.to_pydantic_ai()

        # The adapter should have access to the original model
        assert pydantic_model._esperanto_model is model
        assert pydantic_model._esperanto_model.temperature == 0.5
        assert pydantic_model._esperanto_model.max_tokens == 200


class TestImportErrorHandling:
    """Test ImportError handling."""

    def test_to_pydantic_ai_method_exists(self):
        """Test that to_pydantic_ai method exists on all language models."""
        model = OpenAILanguageModel(
            api_key="test-key",
            model_name="gpt-4o",
        )

        assert hasattr(model, "to_pydantic_ai")
        assert callable(model.to_pydantic_ai)


class TestMultipleAdapterInstances:
    """Test creating multiple adapter instances."""

    def test_multiple_adapters_same_model(self):
        """Test creating multiple adapters from the same model."""
        model = OpenAILanguageModel(
            api_key="test-key",
            model_name="gpt-4o",
        )

        adapter1 = model.to_pydantic_ai()
        adapter2 = model.to_pydantic_ai()

        # Should be different instances
        assert adapter1 is not adapter2
        # But same underlying model
        assert adapter1._esperanto_model is adapter2._esperanto_model

    def test_multiple_adapters_different_models(self):
        """Test creating adapters from different models."""
        model1 = OpenAILanguageModel(
            api_key="test-key",
            model_name="gpt-4o",
        )
        model2 = AnthropicLanguageModel(
            api_key="test-key",
            model_name="claude-sonnet-4-20250514",
        )

        adapter1 = model1.to_pydantic_ai()
        adapter2 = model2.to_pydantic_ai()

        assert adapter1.model_name == "gpt-4o"
        assert adapter2.model_name == "claude-sonnet-4-20250514"
        assert adapter1.system == "openai"
        assert adapter2.system == "anthropic"
