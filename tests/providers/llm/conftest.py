import pytest
from unittest.mock import MagicMock, patch

from esperanto.providers.llm.gemini import GeminiLanguageModel
from esperanto.providers.llm.openai import OpenAILanguageModel


@pytest.fixture
def openai_model():
    with patch("openai.OpenAI") as mock_openai, \
         patch("openai.AsyncOpenAI") as mock_async_openai:
        model = OpenAILanguageModel(api_key="test-key")
        yield model


@pytest.fixture
def gemini_model():
    with patch("google.generativeai.configure") as mock_configure, \
         patch("google.generativeai.GenerativeModel") as mock_model:
        # Create a mock instance
        mock_instance = MagicMock()
        mock_model.return_value = mock_instance
        
        # Initialize model with test key
        model = GeminiLanguageModel(api_key="test-key")
        
        yield model
