"""Tests for the AIFactory class."""

import pytest
from unittest.mock import patch, MagicMock

from esperanto.factory import AIFactory


class TestAIFactoryCaching:
    """Test the caching functionality of AIFactory."""

    def test_language_model_caching(self):
        """Test that language models are cached properly."""
        # Clear the cache to ensure a clean state
        AIFactory.clear_cache()

        # Create a model
        with patch("esperanto.factory.AIFactory._import_provider_class") as mock_import:
            # Create a mock provider class that returns a new instance each time it's called
            mock_provider_class = MagicMock()
            mock_provider_class.side_effect = lambda **kwargs: MagicMock()
            mock_import.return_value = mock_provider_class
            
            # First call should create a new instance
            model1 = AIFactory.create_language("openai", "gpt-4", {"temperature": 0.7})
            
            # Second call with same parameters should return the cached instance
            model2 = AIFactory.create_language("openai", "gpt-4", {"temperature": 0.7})
            
            # Different parameters should create a new instance
            model3 = AIFactory.create_language("openai", "gpt-4", {"temperature": 0.9})
            
            # Different model name should create a new instance
            model4 = AIFactory.create_language("openai", "gpt-3.5-turbo", {"temperature": 0.7})

            # Verify that mock_provider_class was called the correct number of times
            assert mock_provider_class.call_count == 3
            
            # Verify that the same instance is returned for identical parameters
            assert model1 is model2
            
            # Verify that different instances are returned for different parameters
            assert model1 is not model3
            assert model1 is not model4
            assert model3 is not model4

    def test_embedding_model_caching(self):
        """Test that embedding models are cached properly."""
        # Clear the cache to ensure a clean state
        AIFactory.clear_cache()

        # Create a model
        with patch("esperanto.factory.AIFactory._import_provider_class") as mock_import:
            # Create a mock provider class that returns a new instance each time it's called
            mock_provider_class = MagicMock()
            mock_provider_class.side_effect = lambda **kwargs: MagicMock()
            mock_import.return_value = mock_provider_class
            
            # First call should create a new instance
            model1 = AIFactory.create_embedding("openai", "text-embedding-3-small", {"dimensions": 1536})
            
            # Second call with same parameters should return the cached instance
            model2 = AIFactory.create_embedding("openai", "text-embedding-3-small", {"dimensions": 1536})
            
            # Different parameters should create a new instance
            model3 = AIFactory.create_embedding("openai", "text-embedding-3-small", {"dimensions": 768})

            # Verify that mock_provider_class was called the correct number of times
            assert mock_provider_class.call_count == 2
            
            # Verify that the same instance is returned for identical parameters
            assert model1 is model2
            
            # Verify that different instances are returned for different parameters
            assert model1 is not model3

    def test_speech_to_text_model_caching(self):
        """Test that speech-to-text models are cached properly."""
        # Clear the cache to ensure a clean state
        AIFactory.clear_cache()

        # Create a model
        with patch("esperanto.factory.AIFactory._import_provider_class") as mock_import:
            # Create a mock provider class that returns a new instance each time it's called
            mock_provider_class = MagicMock()
            mock_provider_class.side_effect = lambda **kwargs: MagicMock()
            mock_import.return_value = mock_provider_class
            
            # First call should create a new instance
            model1 = AIFactory.create_speech_to_text("openai", "whisper-1")
            
            # Second call with same parameters should return the cached instance
            model2 = AIFactory.create_speech_to_text("openai", "whisper-1")
            
            # Different model name should create a new instance
            model3 = AIFactory.create_speech_to_text("openai", "whisper-2")

            # Verify that mock_provider_class was called the correct number of times
            assert mock_provider_class.call_count == 2
            
            # Verify that the same instance is returned for identical parameters
            assert model1 is model2
            
            # Verify that different instances are returned for different parameters
            assert model1 is not model3

    def test_text_to_speech_model_caching(self):
        """Test that text-to-speech models are cached properly."""
        # Clear the cache to ensure a clean state
        AIFactory.clear_cache()

        # Create a model
        with patch("esperanto.factory.AIFactory._import_provider_class") as mock_import:
            # Create a mock provider class that returns a new instance each time it's called
            mock_provider_class = MagicMock()
            mock_provider_class.side_effect = lambda **kwargs: MagicMock()
            mock_import.return_value = mock_provider_class
            
            # First call should create a new instance
            model1 = AIFactory.create_text_to_speech("openai", "tts-1", api_key="test-key")
            
            # Second call with same parameters should return the cached instance
            model2 = AIFactory.create_text_to_speech("openai", "tts-1", api_key="test-key")
            
            # Different api_key should create a new instance
            model3 = AIFactory.create_text_to_speech("openai", "tts-1", api_key="different-key")

            # Verify that mock_provider_class was called the correct number of times
            assert mock_provider_class.call_count == 2
            
            # Verify that the same instance is returned for identical parameters
            assert model1 is model2
            
            # Verify that different instances are returned for different parameters
            assert model1 is not model3

    def test_clear_cache(self):
        """Test that clear_cache removes all cached instances."""
        # Clear the cache to ensure a clean state
        AIFactory.clear_cache()

        # Patch the import_provider_class method
        with patch("esperanto.factory.AIFactory._import_provider_class") as mock_import:
            # Create a mock provider class that returns a new instance each time it's called
            mock_provider_class = MagicMock()
            mock_provider_class.side_effect = lambda **kwargs: MagicMock()
            mock_import.return_value = mock_provider_class
            
            # Create instances of different model types
            language_model = AIFactory.create_language("openai", "gpt-4")
            embedding_model = AIFactory.create_embedding("openai", "text-embedding-3-small")
            
            # Verify models are cached
            assert AIFactory.create_language("openai", "gpt-4") is language_model
            assert AIFactory.create_embedding("openai", "text-embedding-3-small") is embedding_model
            
            # Clear the cache
            AIFactory.clear_cache()
            
            # Create new instances
            new_language_model = AIFactory.create_language("openai", "gpt-4")
            new_embedding_model = AIFactory.create_embedding("openai", "text-embedding-3-small")
            
            # Verify that new instances are different from the original ones
            assert new_language_model is not language_model
            assert new_embedding_model is not embedding_model

            # Verify that mock_provider_class was called the correct number of times (4 times)
            assert mock_provider_class.call_count == 4

    def test_cache_key_generation(self):
        """Test that cache keys are generated correctly."""
        # Test with different configurations
        key1 = AIFactory._generate_cache_key("language", "openai", "gpt-4", {"temperature": 0.7})
        key2 = AIFactory._generate_cache_key("language", "openai", "gpt-4", {"temperature": 0.7})
        key3 = AIFactory._generate_cache_key("language", "openai", "gpt-4", {"temperature": 0.9})
        
        # Same config should generate the same key
        assert key1 == key2
        
        # Different config should generate different keys
        assert key1 != key3
        
        # Test with None values
        key4 = AIFactory._generate_cache_key("speech_to_text", "openai", None)
        key5 = AIFactory._generate_cache_key("speech_to_text", "openai", None)
        
        # Same config with None values should generate the same key
        assert key4 == key5
        
        # Test with ordered vs unordered dict
        key6 = AIFactory._generate_cache_key("language", "openai", "gpt-4", {"a": 1, "b": 2})
        key7 = AIFactory._generate_cache_key("language", "openai", "gpt-4", {"b": 2, "a": 1})
        
        # Different order in dict should still generate the same key
        assert key6 == key7
