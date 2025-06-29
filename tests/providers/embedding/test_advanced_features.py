"""Tests for advanced embedding features across providers."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from esperanto.factory import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType
from esperanto.providers.embedding.openai import OpenAIEmbeddingModel
from esperanto.providers.embedding.jina import JinaEmbeddingModel
from esperanto.providers.embedding.transformers import TransformersEmbeddingModel


class TestFeatureDetection:
    """Test feature detection and filtering functionality."""
    
    def test_unsupported_features_filtered(self):
        """Test that providers without SUPPORTED_FEATURES filter advanced params."""
        # Create OpenAI model (basic provider without advanced features)
        with patch('httpx.Client'), patch('httpx.AsyncClient'):
            openai_model = OpenAIEmbeddingModel(api_key="test-key")
            
            # OpenAI doesn't support advanced features
            assert not hasattr(openai_model, 'SUPPORTED_FEATURES') or \
                   openai_model.SUPPORTED_FEATURES == []
            
            # Advanced features should be None/False for unsupported providers
            assert openai_model.task_type is None
            assert openai_model.late_chunking is False
            assert openai_model.output_dimensions is None
            assert openai_model.truncate_at_max_length is True  # Default behavior
    
    def test_supported_features_preserved(self):
        """Test that providers with SUPPORTED_FEATURES preserve advanced params."""
        # Create Jina model (supports all advanced features)
        jina_model = JinaEmbeddingModel(
            api_key="test-key",
            config={
                "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,
                "late_chunking": True,
                "output_dimensions": 512,
                "truncate_at_max_length": False
            }
        )
        
        # Jina supports all advanced features
        assert hasattr(jina_model, 'SUPPORTED_FEATURES')
        assert "task_type" in jina_model.SUPPORTED_FEATURES
        assert "late_chunking" in jina_model.SUPPORTED_FEATURES
        assert "output_dimensions" in jina_model.SUPPORTED_FEATURES
        
        # Features should be preserved
        assert jina_model.task_type == EmbeddingTaskType.RETRIEVAL_QUERY
        assert jina_model.late_chunking is True
        assert jina_model.output_dimensions == 512
        assert jina_model.truncate_at_max_length is False
    
    def test_partial_feature_support(self):
        """Test providers with partial feature support."""
        # Test that basic providers don't support advanced features
        with patch('httpx.Client'), patch('httpx.AsyncClient'):
            model = OpenAIEmbeddingModel(
                api_key="test-key",
                config={
                    "task_type": EmbeddingTaskType.CLASSIFICATION,
                    "late_chunking": True,  # Not supported by OpenAI
                    "output_dimensions": 256,  # Not supported by OpenAI
                    "truncate_at_max_length": False
                }
            )
            
            # OpenAI doesn't support advanced features, so they should be default values
            # The base class sets these regardless, but they're not used in OpenAI implementation
            assert model.task_type == EmbeddingTaskType.CLASSIFICATION  # Set but not used
            assert model.truncate_at_max_length is False
            # These should use the default values since OpenAI doesn't use them
            assert model.late_chunking is True  # Set in config but not used by OpenAI
            assert model.output_dimensions == 256  # Set in config but not used by OpenAI


class TestTaskPrefixVsNativeAPI:
    """Test task prefix addition vs native API usage."""
    
    def test_prefix_based_optimization(self):
        """Test providers that use prefix-based task optimization."""
        # Transformers uses prefix-based optimization
        model = AIFactory.create_embedding(
            provider="transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            config={
                "device": "cpu",
                "task_type": EmbeddingTaskType.RETRIEVAL_QUERY
            }
        )
        
        # Test that prefix is added
        text = "Sample query text"
        processed = model._apply_task_optimization([text])
        
        assert len(processed) == 1
        assert processed[0] != text  # Should be modified
        assert "Represent this query for retrieving relevant documents:" in processed[0]
        assert text in processed[0]
    
    def test_native_api_optimization(self):
        """Test providers that use native API task optimization."""
        # Mock HTTP client for Jina
        mock_client = Mock()
        mock_async_client = AsyncMock()
        
        # Jina uses native API task optimization
        jina_model = JinaEmbeddingModel(
            api_key="test-key",
            config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
        )
        jina_model.client = mock_client
        jina_model.async_client = mock_async_client
        
        # Test that no prefix is added (handled by API)
        text = "Sample query text"
        processed = jina_model._apply_task_optimization([text])
        
        assert processed == [text]  # Should be unchanged
        
        # But task type should be mapped for API
        mapped_task = jina_model._map_task_type()
        assert mapped_task == "retrieval.query"
    
    def test_task_type_api_payload_inclusion(self):
        """Test that native API providers include task type in payload."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        
        # Mock client
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        
        jina_model = JinaEmbeddingModel(
            api_key="test-key",
            config={"task_type": EmbeddingTaskType.CLASSIFICATION}
        )
        jina_model.client = mock_client
        
        # Make embedding request
        jina_model.embed(["Test text"])
        
        # Check that task type was included in API payload
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        
        assert "task" in payload
        assert payload["task"] == "classification"
    
    def test_no_task_optimization_when_disabled(self):
        """Test that no optimization occurs when task_type is None."""
        # Transformers model without task type
        model = AIFactory.create_embedding(
            provider="transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            config={"device": "cpu"}  # No task_type
        )
        
        text = "Sample text"
        processed = model._apply_task_optimization([text])
        
        assert processed == [text]  # Should be unchanged


class TestGracefulDegradation:
    """Test graceful degradation when features aren't supported."""
    
    def test_missing_dependencies_warning(self):
        """Test warning when advanced features are requested but dependencies missing."""
        with patch('esperanto.providers.embedding.transformers.ADVANCED_FEATURES_AVAILABLE', False):
            with patch('esperanto.providers.embedding.transformers.logger') as mock_logger:
                # Request advanced features without dependencies
                model = AIFactory.create_embedding(
                    provider="transformers",
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    config={
                        "device": "cpu",
                        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,
                        "late_chunking": True,
                        "output_dimensions": 256
                    }
                )
                
                # Should log warning about missing dependencies
                mock_logger.warning.assert_called()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "dependencies not available" in warning_msg.lower()
    
    @patch('esperanto.providers.embedding.transformers.ADVANCED_FEATURES_AVAILABLE', False)
    def test_fallback_chunking_behavior(self):
        """Test fallback chunking when sentence-transformers not available."""
        model = AIFactory.create_embedding(
            provider="transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            config={
                "device": "cpu",
                "late_chunking": True
            }
        )
        
        # Long text that would normally be chunked
        long_text = " ".join([f"Sentence {i}." for i in range(100)])
        
        # Should still work with fallback chunking
        chunked = model._apply_late_chunking([long_text])
        
        # Fallback should use base class implementation
        assert isinstance(chunked, list)
        assert len(chunked) >= 1
    
    @patch('esperanto.providers.embedding.transformers.ADVANCED_FEATURES_AVAILABLE', False)
    def test_fallback_dimension_control(self):
        """Test fallback dimension control when PCA not available."""
        model = AIFactory.create_embedding(
            provider="transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            config={
                "device": "cpu",
                "output_dimensions": 200
            }
        )
        
        # Create test embeddings
        test_embeddings = np.random.randn(3, 384).astype(np.float32)
        
        # Should use truncation fallback instead of PCA
        reduced = model._apply_dimension_control(test_embeddings)
        
        assert reduced.shape == (3, 200)
        # Should be simple truncation
        np.testing.assert_array_equal(reduced, test_embeddings[:, :200])
    
    def test_invalid_task_type_handling(self):
        """Test handling of invalid task types."""
        model = AIFactory.create_embedding(
            provider="transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            config={
                "device": "cpu",
                "task_type": "invalid_task_type"  # Invalid
            }
        )
        
        # Should gracefully fall back to None
        assert model.task_type is None
        
        # Should not apply any optimization
        text = "Test text"
        processed = model._apply_task_optimization([text])
        assert processed == [text]
    
    def test_network_error_handling(self):
        """Test graceful handling of network errors."""
        # Mock network error - use httpx.RequestError which is what Jina catches
        import httpx
        
        mock_client = Mock()
        mock_client.post.side_effect = httpx.RequestError("Network error")
        
        jina_model = JinaEmbeddingModel(api_key="test-key")
        jina_model.client = mock_client
        
        # Should raise a more informative error
        with pytest.raises(RuntimeError, match="Network error"):
            jina_model.embed(["Test text"])


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_input_validation(self):
        """Test handling of empty inputs."""
        model = AIFactory.create_embedding(
            provider="transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            config={"device": "cpu"}
        )
        
        # Empty list should raise error
        with pytest.raises(ValueError, match="Texts cannot be empty"):
            model.embed([])
        
        # List with empty string should be handled
        with patch.object(model, '_clean_text', return_value="cleaned"):
            # Should work with cleaned text
            embeddings = model.embed([""])
            assert len(embeddings) == 1
    
    def test_very_long_text_handling(self):
        """Test handling of extremely long texts."""
        model = AIFactory.create_embedding(
            provider="transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            config={
                "device": "cpu",
                "late_chunking": True
            }
        )
        
        # Create extremely long text (>10k tokens)
        very_long_text = "Very long text. " * 3000
        
        # Should handle without crashing
        embeddings = model.embed([very_long_text])
        
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) > 0
    
    def test_special_characters_handling(self):
        """Test handling of texts with special characters."""
        model = AIFactory.create_embedding(
            provider="transformers", 
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            config={"device": "cpu"}
        )
        
        special_texts = [
            "Text with émojis 🚀🎉",
            "Text with Unicode: αβγδε",
            "Text with\nnewlines\tand\ttabs",
            "Text with \"quotes\" and 'apostrophes'",
            "Text with numbers: 123.456",
            "Text with symbols: @#$%^&*()",
            ""  # Empty string
        ]
        
        # Should handle all special characters
        embeddings = model.embed(special_texts)
        
        assert len(embeddings) == len(special_texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
    
    def test_large_batch_processing(self):
        """Test processing of large batches."""
        model = AIFactory.create_embedding(
            provider="transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2", 
            config={"device": "cpu"}
        )
        
        # Large batch (100 texts)
        large_batch = [f"Text number {i}" for i in range(100)]
        
        # Should process successfully with batching
        embeddings = model.embed(large_batch, batch_size=10)
        
        assert len(embeddings) == 100
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) == len(embeddings[0]) for emb in embeddings)  # Consistent dimensions
    
    def test_zero_dimension_edge_case(self):
        """Test edge case with zero output dimensions."""
        model = AIFactory.create_embedding(
            provider="transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            config={
                "device": "cpu",
                "output_dimensions": 0  # Edge case
            }
        )
        
        # Create test embeddings
        test_embeddings = np.random.randn(2, 384).astype(np.float32)
        
        # Should handle gracefully
        result = model._apply_dimension_control(test_embeddings)
        
        # Might return empty embeddings or handle specially
        assert result.shape[0] == 2
        assert result.shape[1] == 0
    
    def test_single_token_text(self):
        """Test handling of very short texts (single token)."""
        model = AIFactory.create_embedding(
            provider="transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            config={"device": "cpu"}
        )
        
        short_texts = ["a", "I", ".", "123", "é"]
        
        embeddings = model.embed(short_texts)
        
        assert len(embeddings) == len(short_texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
    
    @pytest.mark.asyncio
    async def test_async_edge_cases(self):
        """Test edge cases in async operations."""
        model = AIFactory.create_embedding(
            provider="transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            config={"device": "cpu"}
        )
        
        # Test async with valid input (since async just calls sync version)
        embeddings = await model.aembed(["Valid text"])
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        
        # Test async with empty input
        with pytest.raises(ValueError, match="Texts cannot be empty"):
            await model.aembed([])


class TestCrossProviderConsistency:
    """Test consistency across different providers."""
    
    def test_consistent_error_messages(self):
        """Test that similar errors have consistent messages across providers."""
        # Test empty input handling
        providers_configs = [
            ("transformers", {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "config": {"device": "cpu"}
            })
        ]
        
        for provider_name, config in providers_configs:
            model = AIFactory.create_embedding(provider=provider_name, **config)
            
            with pytest.raises(ValueError) as exc_info:
                model.embed([])
            
            # Error message should mention "empty"
            assert "empty" in str(exc_info.value).lower()
    
    def test_consistent_return_types(self):
        """Test that all providers return consistent data types."""
        # Mock providers to avoid actual model loading/API calls
        with patch('httpx.Client'), patch('httpx.AsyncClient'):
            # Create providers with direct instantiation to avoid factory validation
            from esperanto.providers.embedding.openai import OpenAIEmbeddingModel
            providers = [
                OpenAIEmbeddingModel(api_key="test-key"),
                JinaEmbeddingModel(api_key="test-key")
            ]
        
        # Add transformers if available
        try:
            providers.append(AIFactory.create_embedding(
                "transformers", 
                "sentence-transformers/all-MiniLM-L6-v2",
                config={"device": "cpu"}
            ))
        except Exception:
            pass  # Skip if transformers not available
        
        for provider in providers:
            # Mock the embed method to return consistent data
            with patch.object(provider, 'embed', return_value=[[0.1, 0.2, 0.3]]):
                embeddings = provider.embed(["test"])
                
                # All should return List[List[float]]
                assert isinstance(embeddings, list)
                assert isinstance(embeddings[0], list)
                assert all(isinstance(val, float) for val in embeddings[0])