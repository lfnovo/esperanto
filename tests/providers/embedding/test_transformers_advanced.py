"""Tests for the Transformers embedding model provider with advanced features."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from esperanto.factory import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType


@pytest.fixture
def basic_model():
    """Create a basic test model instance."""
    return AIFactory.create_embedding(
        provider="transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        config={
            "device": "cpu",  # Force CPU for testing
        },
    )


@pytest.fixture
def advanced_model():
    """Create an advanced test model instance with all features enabled."""
    return AIFactory.create_embedding(
        provider="transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        config={
            "device": "cpu",
            "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,
            "late_chunking": True,
            "output_dimensions": 256,
            "truncate_at_max_length": True,
        },
    )


@pytest.fixture
def qwen_model():
    """Create a Qwen3 model instance for testing large context handling."""
    return AIFactory.create_embedding(
        provider="transformers",
        model_name="Qwen/Qwen3-Embedding-0.6B",
        config={
            "device": "cpu",
            "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
            "late_chunking": True,
        },
    )


class TestBasicFunctionality:
    """Test basic functionality without advanced features."""

    def test_initialization(self, basic_model):
        """Test model initialization."""
        assert basic_model.provider == "transformers"
        assert "all-MiniLM-L6-v2" in basic_model.get_model_name()
        assert basic_model.device == "cpu"

    def test_embed_basic(self, basic_model):
        """Test basic embedding generation."""
        texts = ["Hello world", "This is a test"]
        embeddings = basic_model.embed(texts)

        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(isinstance(val, float) for emb in embeddings for val in emb)
        # MiniLM has 384-dimensional embeddings
        assert all(len(emb) == 384 for emb in embeddings)

    @pytest.mark.asyncio
    async def test_aembed_basic(self, basic_model):
        """Test async embedding generation."""
        texts = ["Hello world", "This is a test"]
        embeddings = await basic_model.aembed(texts)

        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) == 384 for emb in embeddings)


class TestTaskOptimization:
    """Test task-specific optimization features."""

    def test_task_optimization_enabled(self, advanced_model):
        """Test that task optimization is applied."""
        text = "Machine learning is fascinating"
        
        # Get processed text via the internal method
        processed = advanced_model._apply_task_optimization([text])
        
        assert len(processed) == 1
        assert processed[0].startswith("Represent this query for retrieving relevant documents:")
        assert text in processed[0]

    def test_different_task_types(self, basic_model):
        """Test different task type optimizations."""
        text = "Sample text"
        
        # Test different task types
        task_tests = [
            (EmbeddingTaskType.CLASSIFICATION, "Represent this text for classification:"),
            (EmbeddingTaskType.CLUSTERING, "Represent this text for clustering:"),
            (EmbeddingTaskType.SIMILARITY, "Represent this text for semantic similarity:"),
            (EmbeddingTaskType.CODE_RETRIEVAL, "Represent this code for search:"),
        ]
        
        for task_type, expected_prefix in task_tests:
            basic_model.task_type = task_type
            processed = basic_model._apply_task_optimization([text])
            assert processed[0].startswith(expected_prefix)
            assert text in processed[0]

    def test_no_task_optimization_default(self, basic_model):
        """Test that no optimization is applied by default."""
        text = "Sample text"
        # Ensure no task type is set
        basic_model.task_type = None
        processed = basic_model._apply_task_optimization([text])
        
        assert processed == [text]  # No change

    def test_task_optimization_in_embedding(self, advanced_model):
        """Test that task optimization affects actual embeddings."""
        text = "Machine learning research"
        
        # Create model without task optimization
        basic_model = AIFactory.create_embedding(
            provider="transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            config={"device": "cpu"},
        )
        
        # Get embeddings with and without task optimization
        basic_embedding = basic_model.embed([text])
        optimized_embedding = advanced_model.embed([text])
        
        # They should be different due to the prefix
        assert basic_embedding != optimized_embedding


class TestLateChunking:
    """Test late chunking functionality."""

    def test_no_chunking_short_text(self, advanced_model):
        """Test that short texts are not chunked."""
        short_text = "This is a short text that doesn't need chunking."
        
        chunked = advanced_model._apply_late_chunking([short_text])
        assert len(chunked) == 1
        assert chunked[0] == short_text

    def test_chunking_long_text(self, advanced_model):
        """Test that long texts are chunked."""
        # Create a long text that should be chunked
        long_text = " ".join([f"Sentence {i} with some content." for i in range(100)])
        
        chunked = advanced_model._apply_late_chunking([long_text])
        
        # Should be chunked into multiple pieces
        assert len(chunked) > 1
        assert all(isinstance(chunk, str) for chunk in chunked)
        
        # Each chunk should be reasonably sized
        for chunk in chunked:
            estimated_tokens = len(chunk) // 4
            assert estimated_tokens <= advanced_model._max_chunk_tokens * 1.2  # Allow some variance

    def test_sentence_splitting(self, advanced_model):
        """Test sentence-based splitting logic."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        
        sentences = advanced_model._split_into_sentences(text)
        
        assert len(sentences) == 4
        assert "First sentence." in sentences[0]
        assert "Second sentence!" in sentences[1]
        assert "Third sentence?" in sentences[2]
        assert "Fourth sentence." in sentences[3]

    @patch('esperanto.providers.embedding.transformers.ADVANCED_FEATURES_AVAILABLE', False)
    def test_fallback_chunking_without_dependencies(self, advanced_model):
        """Test that chunking works even without optional dependencies."""
        long_text = " ".join([f"Sentence {i}." for i in range(50)])
        
        chunked = advanced_model._apply_late_chunking([long_text])
        
        # Should still chunk, just with simpler logic
        assert len(chunked) >= 1


class TestDimensionControl:
    """Test output dimension control functionality."""

    def test_dimension_reduction(self, basic_model):
        """Test PCA-based dimension reduction."""
        basic_model.output_dimensions = 128
        
        # Create some test embeddings
        test_embeddings = np.random.randn(5, 384).astype(np.float32)
        
        reduced = basic_model._apply_dimension_control(test_embeddings)
        
        assert reduced.shape == (5, 128)
        # PCA typically returns float64, but could be float32 depending on implementation
        assert reduced.dtype in [np.float32, np.float64]

    def test_dimension_expansion(self, basic_model):
        """Test zero-padding dimension expansion."""
        basic_model.output_dimensions = 512
        
        # Create some test embeddings
        test_embeddings = np.random.randn(3, 384).astype(np.float32)
        
        expanded = basic_model._apply_dimension_control(test_embeddings)
        
        assert expanded.shape == (3, 512)
        # First 384 dimensions should be preserved, rest should be zeros
        np.testing.assert_array_equal(expanded[:, :384], test_embeddings)
        np.testing.assert_array_equal(expanded[:, 384:], np.zeros((3, 128)))

    def test_no_dimension_change(self, basic_model):
        """Test that embeddings are unchanged when target dimension equals current."""
        basic_model.output_dimensions = 384  # Set to match embedding dimension
        test_embeddings = np.random.randn(2, 384).astype(np.float32)
        
        unchanged = basic_model._apply_dimension_control(test_embeddings)
        
        np.testing.assert_array_equal(unchanged, test_embeddings)

    def test_dimension_control_in_embedding(self, basic_model):
        """Test dimension control in actual embedding generation."""
        basic_model.output_dimensions = 256
        
        texts = ["Test text for dimension control"]
        embeddings = basic_model.embed(texts)
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 256

    @patch('esperanto.providers.embedding.transformers.ADVANCED_FEATURES_AVAILABLE', False)
    def test_dimension_control_fallback(self, basic_model):
        """Test dimension control fallback when PCA is not available."""
        basic_model.output_dimensions = 200
        
        # Create test embeddings
        test_embeddings = np.random.randn(2, 384).astype(np.float32)
        
        # Should use truncation fallback
        reduced = basic_model._reduce_dimensions(test_embeddings, 200)
        
        assert reduced.shape == (2, 200)
        # Should be simple truncation
        np.testing.assert_array_equal(reduced, test_embeddings[:, :200])


class TestIntegration:
    """Test integration of all advanced features."""

    def test_full_pipeline(self, advanced_model):
        """Test the complete preprocessing pipeline."""
        texts = [
            "What is machine learning?",
            "A very long document that contains multiple sentences. It should demonstrate the late chunking functionality. Each sentence adds more content to trigger chunking behavior. This continues with even more text to ensure the chunking algorithm activates properly."
        ]
        
        embeddings = advanced_model.embed(texts)
        
        # Should still return one embedding per input text
        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        
        # Dimensions should be controlled
        assert all(len(emb) == 256 for emb in embeddings)  # output_dimensions=256

    def test_qwen_model_configuration(self, qwen_model):
        """Test Qwen3 model-specific configuration."""
        # Qwen3 should have larger token limits
        assert qwen_model._max_chunk_tokens == 8192
        
        # Should handle task optimization
        text = "Research document content"
        processed = qwen_model._apply_task_optimization([text])
        assert processed[0].startswith("Represent this document for retrieval:")

    def test_config_parameter_extraction(self):
        """Test that config parameters are properly extracted."""
        model = AIFactory.create_embedding(
            provider="transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            config={
                "device": "cpu",
                "task_type": "retrieval_query",  # String version
                "late_chunking": True,
                "output_dimensions": 300,
                "truncate_at_max_length": False,
            },
        )
        
        assert model.task_type == EmbeddingTaskType.RETRIEVAL_QUERY
        assert model.late_chunking is True
        assert model.output_dimensions == 300
        assert model.truncate_at_max_length is False

    def test_backward_compatibility(self, basic_model):
        """Test that existing code without advanced features still works."""
        texts = ["Simple test", "Another test"]
        embeddings = basic_model.embed(texts)
        
        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(isinstance(val, float) for emb in embeddings for val in emb)
        # All embeddings should have same dimension
        embedding_dim = len(embeddings[0])
        assert all(len(emb) == embedding_dim for emb in embeddings)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_texts(self, advanced_model):
        """Test error handling for empty text lists."""
        with pytest.raises(ValueError, match="Texts cannot be empty"):
            advanced_model.embed([])

    def test_invalid_task_type(self):
        """Test handling of invalid task types."""
        model = AIFactory.create_embedding(
            provider="transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            config={
                "device": "cpu",
                "task_type": "invalid_task_type",
            },
        )
        
        # Should fall back to no task optimization
        assert model.task_type is None

    def test_missing_optional_dependencies_warning(self):
        """Test warning when advanced features are requested but dependencies missing."""
        with patch('esperanto.providers.embedding.transformers.ADVANCED_FEATURES_AVAILABLE', False):
            with patch('esperanto.providers.embedding.transformers.logger') as mock_logger:
                AIFactory.create_embedding(
                    provider="transformers",
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    config={
                        "device": "cpu",
                        "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,
                        "late_chunking": True,
                    },
                )
                
                # Should log a warning about missing dependencies
                mock_logger.warning.assert_called_once()
                assert "dependencies not available" in mock_logger.warning.call_args[0][0]


class TestPerformance:
    """Test performance characteristics."""

    def test_batch_processing(self, advanced_model):
        """Test that batch processing works with advanced features."""
        texts = [f"Test text number {i}" for i in range(10)]
        
        embeddings = advanced_model.embed(texts, batch_size=3)
        
        assert len(embeddings) == len(texts)
        assert all(len(emb) == 256 for emb in embeddings)

    def test_large_text_handling(self, advanced_model):
        """Test handling of very large texts."""
        large_text = "Long text. " * 1000  # Very long text
        
        embeddings = advanced_model.embed([large_text])
        
        # Should still return one embedding
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 256