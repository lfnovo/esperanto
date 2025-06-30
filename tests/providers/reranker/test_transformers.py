"""Basic test for Transformers reranker provider."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

from esperanto.providers.reranker.transformers import TransformersRerankerModel
from esperanto.common_types.reranker import RerankResponse, RerankResult


class TestTransformersReranker:
    """Test cases for Transformers reranker provider."""

    @patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True)
    @patch('esperanto.providers.reranker.transformers.torch')
    @patch('esperanto.providers.reranker.transformers.AutoTokenizer')
    @patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM')
    def test_initialization_with_qwen_setup(self, mock_model_class, mock_tokenizer_class, mock_torch):
        """Test that the model initializes correctly with Qwen-specific setup."""
        # Mock torch components
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.float32 = torch.float32
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.convert_tokens_to_ids.side_effect = lambda token: 100 if token == "yes" else 200
        mock_tokenizer.encode.side_effect = lambda text, **kwargs: [1, 2, 3]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Create reranker instance
        reranker = TransformersRerankerModel(
            model_name="Qwen/Qwen3-Reranker-4B",
            api_key=None,
            config={}
        )
        
        # Verify initialization
        assert reranker.device == "cpu"
        assert reranker.tokenizer == mock_tokenizer
        assert reranker.model == mock_model
        
        # Verify Qwen-specific setup
        assert hasattr(reranker, 'token_true_id')
        assert hasattr(reranker, 'token_false_id')
        assert hasattr(reranker, 'prefix')
        assert hasattr(reranker, 'suffix')
        assert reranker.token_true_id == 100
        assert reranker.token_false_id == 200

    @patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True)
    def test_rerank_functionality_with_mock_score_method(self):
        """Test the reranking functionality by mocking the internal scoring method."""
        with patch('esperanto.providers.reranker.transformers.torch'), \
             patch('esperanto.providers.reranker.transformers.AutoTokenizer'), \
             patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM'):
            
            mock_tokenizer = Mock()
            mock_tokenizer.convert_tokens_to_ids.side_effect = lambda token: 100 if token == "yes" else 200
            mock_tokenizer.encode.side_effect = lambda text, **kwargs: [1, 2, 3]
            
            with patch('esperanto.providers.reranker.transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
                 patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM.from_pretrained'):
                
                reranker = TransformersRerankerModel(
                    model_name="Qwen/Qwen3-Reranker-4B",
                    api_key=None,
                    config={}
                )
                
                # Mock the internal scoring method to return predictable scores
                with patch.object(reranker, '_score_all_pairs', return_value=[0.8, 0.3, 0.6]):
                    query = "What is machine learning?"
                    documents = [
                        "Machine learning is a subset of artificial intelligence.",
                        "The weather is nice today.",
                        "Python is a programming language used in ML."
                    ]
                    
                    result = reranker.rerank(query, documents, top_k=2)
                    
                    # Verify result structure
                    assert isinstance(result, RerankResponse)
                    assert result.model == "Qwen/Qwen3-Reranker-4B"
                    assert len(result.results) == 2  # Respects top_k
                    assert all(isinstance(r, RerankResult) for r in result.results)
                    
                    # Verify results are sorted by score (highest first)
                    scores = [r.relevance_score for r in result.results]
                    assert scores == sorted(scores, reverse=True)
                    
                    # Verify the highest scoring document comes first
                    assert result.results[0].document == documents[0]  # Should be doc with score 0.8
                    assert result.results[1].document == documents[2]  # Should be doc with score 0.6

    def test_format_instruction(self):
        """Test the Qwen instruction formatting."""
        with patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True), \
             patch('esperanto.providers.reranker.transformers.torch'), \
             patch('esperanto.providers.reranker.transformers.AutoTokenizer'), \
             patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM'):
            
            mock_tokenizer = Mock()
            mock_tokenizer.convert_tokens_to_ids.side_effect = lambda token: 100 if token == "yes" else 200
            mock_tokenizer.encode.side_effect = lambda text, **kwargs: [1, 2, 3]
            
            with patch('esperanto.providers.reranker.transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
                 patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM.from_pretrained'):
                
                reranker = TransformersRerankerModel(
                    model_name="Qwen/Qwen3-Reranker-4B",
                    api_key=None,
                    config={}
                )
                
                instruction = "Test instruction"
                query = "Test query"
                doc = "Test document"
                
                formatted = reranker._format_instruction(instruction, query, doc)
                
                expected = "<Instruct>: Test instruction\n<Query>: Test query\n<Document>: Test document"
                assert formatted == expected

    def test_format_instruction_default(self):
        """Test the Qwen instruction formatting with default instruction."""
        with patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True), \
             patch('esperanto.providers.reranker.transformers.torch'), \
             patch('esperanto.providers.reranker.transformers.AutoTokenizer'), \
             patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM'):
            
            mock_tokenizer = Mock()
            mock_tokenizer.convert_tokens_to_ids.side_effect = lambda token: 100 if token == "yes" else 200
            mock_tokenizer.encode.side_effect = lambda text, **kwargs: [1, 2, 3]
            
            with patch('esperanto.providers.reranker.transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
                 patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM.from_pretrained'):
                
                reranker = TransformersRerankerModel(
                    model_name="Qwen/Qwen3-Reranker-4B",
                    api_key=None,
                    config={}
                )
                
                query = "Test query"
                doc = "Test document"
                
                formatted = reranker._format_instruction(None, query, doc)
                
                expected = "<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: Test query\n<Document>: Test document"
                assert formatted == expected

    def test_provider_properties(self):
        """Test provider properties."""
        with patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True), \
             patch('esperanto.providers.reranker.transformers.torch'), \
             patch('esperanto.providers.reranker.transformers.AutoTokenizer'), \
             patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM'):
            
            mock_tokenizer = Mock()
            mock_tokenizer.convert_tokens_to_ids.side_effect = lambda token: 100 if token == "yes" else 200
            mock_tokenizer.encode.side_effect = lambda text, **kwargs: [1, 2, 3]
            
            with patch('esperanto.providers.reranker.transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
                 patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM.from_pretrained'):
                
                reranker = TransformersRerankerModel(
                    model_name="Qwen/Qwen3-Reranker-4B",
                    api_key=None,
                    config={}
                )
                
                assert reranker.provider == "transformers"
                assert len(reranker.models) > 0
                assert all(model.type == "reranker" for model in reranker.models)
                assert reranker._get_default_model() == "Qwen/Qwen3-Reranker-4B"

    def test_transformers_not_available(self):
        """Test handling when transformers library is not available."""
        with patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', False):
            with pytest.raises(ImportError, match="Transformers library not installed"):
                TransformersRerankerModel(
                    model_name="Qwen/Qwen3-Reranker-4B",
                    api_key=None,
                    config={}
                )