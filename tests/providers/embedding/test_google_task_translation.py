"""Tests for Google embedding task type translation functionality."""

import pytest
from unittest.mock import Mock, patch

from esperanto.common_types.task_type import EmbeddingTaskType
from esperanto.providers.embedding.google import GoogleEmbeddingModel


@pytest.fixture
def mock_google_embedding_response():
    """Mock Google API embedding response."""
    return {
        "embedding": {
            "values": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
    }


@pytest.fixture
def google_model():
    """Create GoogleEmbeddingModel instance for testing."""
    return GoogleEmbeddingModel(api_key="test-key", model_name="text-embedding-004")


class TestGeminiTaskMapping:
    """Test task type mapping from universal enum to Gemini API values."""

    def test_direct_mappings(self, google_model):
        """Test direct one-to-one task type mappings."""
        # Test retrieval tasks
        google_model.task_type = EmbeddingTaskType.RETRIEVAL_QUERY
        assert google_model._get_task_type_param() == "RETRIEVAL_QUERY"
        
        google_model.task_type = EmbeddingTaskType.RETRIEVAL_DOCUMENT
        assert google_model._get_task_type_param() == "RETRIEVAL_DOCUMENT"
        
        # Test classification and clustering
        google_model.task_type = EmbeddingTaskType.CLASSIFICATION
        assert google_model._get_task_type_param() == "CLASSIFICATION"
        
        google_model.task_type = EmbeddingTaskType.CLUSTERING
        assert google_model._get_task_type_param() == "CLUSTERING"
        
        # Test new task types
        google_model.task_type = EmbeddingTaskType.QUESTION_ANSWERING
        assert google_model._get_task_type_param() == "QUESTION_ANSWERING"
        
        google_model.task_type = EmbeddingTaskType.FACT_VERIFICATION
        assert google_model._get_task_type_param() == "FACT_VERIFICATION"

    def test_universal_to_gemini_translations(self, google_model):
        """Test universal task types that map to different Gemini values."""
        # Universal SIMILARITY maps to Gemini SEMANTIC_SIMILARITY
        google_model.task_type = EmbeddingTaskType.SIMILARITY
        assert google_model._get_task_type_param() == "SEMANTIC_SIMILARITY"
        
        # Universal CODE_RETRIEVAL maps to Gemini CODE_RETRIEVAL_QUERY
        google_model.task_type = EmbeddingTaskType.CODE_RETRIEVAL
        assert google_model._get_task_type_param() == "CODE_RETRIEVAL_QUERY"

    def test_default_and_none_behavior(self, google_model):
        """Test fallback behavior for DEFAULT and None task types."""
        # DEFAULT task type maps to None (no task optimization)
        google_model.task_type = EmbeddingTaskType.DEFAULT
        assert google_model._get_task_type_param() is None
        
        # No task type set
        google_model.task_type = None
        assert google_model._get_task_type_param() is None

    def test_complete_mapping_coverage(self, google_model):
        """Test that all enum values have mappings defined."""
        for task_type in EmbeddingTaskType:
            google_model.task_type = task_type
            result = google_model._get_task_type_param()
            # All task types should have a mapping (even if None for DEFAULT)
            assert task_type in GoogleEmbeddingModel.GEMINI_TASK_MAPPING


class TestNativeTaskOptimization:
    """Test native task optimization in embed/aembed methods."""

    @patch('httpx.Client.post')
    def test_native_task_type_in_payload(self, mock_post, google_model, mock_google_embedding_response):
        """Test that native task types are included in API payload."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_google_embedding_response
        
        # Configure model with task type
        google_model.task_type = EmbeddingTaskType.RETRIEVAL_QUERY
        
        # Make embedding call
        google_model.embed(["test text"])
        
        # Verify API call includes task_type parameter
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert "task_type" in payload
        assert payload["task_type"] == "RETRIEVAL_QUERY"

    @patch('httpx.AsyncClient.post')
    async def test_async_native_task_type_in_payload(self, mock_post, google_model, mock_google_embedding_response):
        """Test that native task types are included in async API payload."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_google_embedding_response
        mock_post.return_value = mock_response
        
        # Configure model with task type
        google_model.task_type = EmbeddingTaskType.CLASSIFICATION
        
        # Make async embedding call
        await google_model.aembed(["test text"])
        
        # Verify async API call includes task_type parameter
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert "task_type" in payload
        assert payload["task_type"] == "CLASSIFICATION"

    @patch('httpx.Client.post')
    def test_no_task_type_when_default(self, mock_post, google_model, mock_google_embedding_response):
        """Test that DEFAULT task type doesn't add task_type parameter."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_google_embedding_response
        
        # Configure model with DEFAULT task type
        google_model.task_type = EmbeddingTaskType.DEFAULT
        
        # Make embedding call
        google_model.embed(["test text"])
        
        # Verify API call doesn't include task_type parameter
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert "task_type" not in payload

    @patch('httpx.Client.post')
    def test_no_task_type_when_none(self, mock_post, google_model, mock_google_embedding_response):
        """Test that None task type doesn't add task_type parameter."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_google_embedding_response
        
        # Configure model with no task type
        google_model.task_type = None
        
        # Make embedding call
        google_model.embed(["test text"])
        
        # Verify API call doesn't include task_type parameter
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert "task_type" not in payload


class TestGeminiTaskTypeIntegration:
    """Test integration of task types with model configuration."""

    def test_task_type_from_config(self):
        """Test task type configuration through config parameter."""
        model = GoogleEmbeddingModel(
            api_key="test-key",
            model_name="text-embedding-004",
            config={"task_type": EmbeddingTaskType.RETRIEVAL_QUERY}
        )
        
        assert model.task_type == EmbeddingTaskType.RETRIEVAL_QUERY
        assert model._get_task_type_param() == "RETRIEVAL_QUERY"

    def test_task_type_string_conversion(self):
        """Test task type string value conversion to enum."""
        model = GoogleEmbeddingModel(
            api_key="test-key",
            model_name="text-embedding-004",
            config={"task_type": "retrieval.query"}
        )
        
        assert model.task_type == EmbeddingTaskType.RETRIEVAL_QUERY
        assert model._get_task_type_param() == "RETRIEVAL_QUERY"

    @patch('httpx.Client.post')
    def test_all_gemini_task_types_work(self, mock_post, mock_google_embedding_response):
        """Test that all Gemini-supported task types work end-to-end."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_google_embedding_response
        
        # Test each task type that has a Gemini mapping
        gemini_supported_tasks = [
            (EmbeddingTaskType.RETRIEVAL_QUERY, "RETRIEVAL_QUERY"),
            (EmbeddingTaskType.RETRIEVAL_DOCUMENT, "RETRIEVAL_DOCUMENT"),
            (EmbeddingTaskType.CLASSIFICATION, "CLASSIFICATION"),
            (EmbeddingTaskType.CLUSTERING, "CLUSTERING"),
            (EmbeddingTaskType.SIMILARITY, "SEMANTIC_SIMILARITY"),
            (EmbeddingTaskType.CODE_RETRIEVAL, "CODE_RETRIEVAL_QUERY"),
            (EmbeddingTaskType.QUESTION_ANSWERING, "QUESTION_ANSWERING"),
            (EmbeddingTaskType.FACT_VERIFICATION, "FACT_VERIFICATION"),
        ]
        
        for task_type, expected_gemini_value in gemini_supported_tasks:
            model = GoogleEmbeddingModel(
                api_key="test-key",
                model_name="text-embedding-004",
                config={"task_type": task_type}
            )
            
            # Make embedding call
            model.embed(["test text"])
            
            # Verify correct Gemini task type was sent
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert payload["task_type"] == expected_gemini_value


class TestBackwardsCompatibility:
    """Test that existing code continues to work unchanged."""

    @patch('httpx.Client.post')
    def test_no_config_still_works(self, mock_post, mock_google_embedding_response):
        """Test that models without task configuration still work."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_google_embedding_response
        
        # Create model without any task configuration
        model = GoogleEmbeddingModel(api_key="test-key", model_name="text-embedding-004")
        
        # Should work without errors
        result = model.embed(["test text"])
        assert len(result) == 1
        assert len(result[0]) == 5  # Mock has 5 dimensions
        
        # Verify no task_type parameter was sent
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert "task_type" not in payload

    @patch('httpx.AsyncClient.post')
    async def test_async_no_config_still_works(self, mock_post, mock_google_embedding_response):
        """Test that async models without task configuration still work."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_google_embedding_response
        mock_post.return_value = mock_response
        
        # Create model without any task configuration
        model = GoogleEmbeddingModel(api_key="test-key", model_name="text-embedding-004")
        
        # Should work without errors
        result = await model.aembed(["test text"])
        assert len(result) == 1
        assert len(result[0]) == 5  # Mock has 5 dimensions
        
        # Verify no task_type parameter was sent
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert "task_type" not in payload


class TestErrorCases:
    """Test error handling and edge cases."""

    def test_invalid_task_type_graceful_handling(self):
        """Test that invalid task types are handled gracefully."""
        # This should not raise an error (base class handles conversion)
        model = GoogleEmbeddingModel(
            api_key="test-key",
            model_name="text-embedding-004",
            config={"task_type": "invalid_task_type"}
        )
        
        # Invalid task type should result in None
        assert model.task_type is None
        assert model._get_task_type_param() is None

    @patch('httpx.Client.post')
    def test_multiple_texts_with_task_types(self, mock_post, mock_google_embedding_response):
        """Test multiple texts with task type optimization."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_google_embedding_response
        
        model = GoogleEmbeddingModel(
            api_key="test-key",
            model_name="text-embedding-004",
            config={"task_type": EmbeddingTaskType.SIMILARITY}
        )
        
        # Test multiple texts
        texts = ["text one", "text two", "text three"]
        result = model.embed(texts)
        
        # Should make 3 API calls (one per text)
        assert mock_post.call_count == 3
        assert len(result) == 3
        
        # Each call should include the task_type
        for call in mock_post.call_args_list:
            payload = call[1]["json"]
            assert payload["task_type"] == "SEMANTIC_SIMILARITY"