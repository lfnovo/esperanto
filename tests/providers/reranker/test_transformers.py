"""Comprehensive tests for universal Transformers reranker provider with 4-strategy architecture."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

from esperanto.providers.reranker.transformers import TransformersRerankerModel
from esperanto.common_types.reranker import RerankResponse, RerankResult


class TestTransformersRerankerStrategyDetection:
    """Test strategy detection logic."""

    def test_strategy_detection_qwen_models(self):
        """Test strategy detection for Qwen models (causal_lm)."""
        with patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True):
            reranker = object.__new__(TransformersRerankerModel)
            
            test_cases = [
                "Qwen/Qwen3-Reranker-4B",
                "Qwen/Qwen3-Reranker-0.6B",
                "Qwen/SomeOtherReranker"
            ]
            
            for model_name in test_cases:
                strategy = reranker._get_model_strategy(model_name)
                assert strategy == "causal_lm", f"Failed for {model_name}"

    def test_strategy_detection_cross_encoder_models(self):
        """Test strategy detection for CrossEncoder models (sentence_transformers)."""
        with patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True):
            reranker = object.__new__(TransformersRerankerModel)
            
            test_cases = [
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "cross-encoder/ms-marco-electra-base",
                "cross-encoder/anything-else"
            ]
            
            for model_name in test_cases:
                strategy = reranker._get_model_strategy(model_name)
                assert strategy == "sentence_transformers", f"Failed for {model_name}"

    def test_strategy_detection_baai_models(self):
        """Test strategy detection for BAAI models (sentence_transformers)."""
        with patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True):
            reranker = object.__new__(TransformersRerankerModel)
            
            test_cases = [
                "BAAI/bge-reranker-base",
                "BAAI/bge-reranker-large",
                "BAAI/bge-reranker-anything"
            ]
            
            for model_name in test_cases:
                strategy = reranker._get_model_strategy(model_name)
                assert strategy == "sentence_transformers", f"Failed for {model_name}"

    def test_strategy_detection_jina_models(self):
        """Test strategy detection for Jina models (sequence_classification)."""
        with patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True):
            reranker = object.__new__(TransformersRerankerModel)
            
            test_cases = [
                "jinaai/jina-reranker-v2-base-multilingual",
                "jinaai/jina-reranker-v1-base-en",
                "jinaai/jina-reranker-anything"
            ]
            
            for model_name in test_cases:
                strategy = reranker._get_model_strategy(model_name)
                assert strategy == "sequence_classification", f"Failed for {model_name}"

    def test_strategy_detection_mixedbread_models(self):
        """Test strategy detection for Mixedbread models (both v1 and v2)."""
        with patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True):
            reranker = object.__new__(TransformersRerankerModel)
            
            # v1 models -> sentence_transformers
            v1_cases = [
                "mixedbread-ai/mxbai-rerank-base-v1",
                "mixedbread-ai/mxbai-rerank-large-v1",
                "mixedbread-ai/mxbai-rerank-xsmall-v1"
            ]
            
            for model_name in v1_cases:
                strategy = reranker._get_model_strategy(model_name)
                assert strategy == "sentence_transformers", f"Failed for {model_name}"
            
            # v2 models -> mixedbread_v2
            v2_cases = [
                "mixedbread-ai/mxbai-rerank-base-v2",
                "mixedbread-ai/mxbai-rerank-large-v2"
            ]
            
            for model_name in v2_cases:
                strategy = reranker._get_model_strategy(model_name)
                assert strategy == "mixedbread_v2", f"Failed for {model_name}"

    def test_strategy_detection_unsupported_model(self):
        """Test that unsupported models raise appropriate error."""
        with patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True):
            reranker = object.__new__(TransformersRerankerModel)
            
            with pytest.raises(ValueError, match="Could not detect reranker strategy"):
                reranker._get_model_strategy("unsupported/random-model")


class TestTransformersRerankerCausalLMStrategy:
    """Test causal LM strategy (Qwen models)."""

    @patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True)
    @patch('esperanto.providers.reranker.transformers.torch')
    @patch('esperanto.providers.reranker.transformers.AutoTokenizer')
    @patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM')
    def test_causal_lm_initialization(self, mock_model_class, mock_tokenizer_class, mock_torch):
        """Test initialization for causal LM strategy."""
        # Mock torch components
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.float32 = torch.float32
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.convert_tokens_to_ids.side_effect = lambda token: 100 if token == "yes" else 200
        mock_tokenizer.decode.return_value = "decoded_text"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Create reranker instance
        reranker = TransformersRerankerModel(model_name="Qwen/Qwen3-Reranker-0.6B")
        
        # Verify strategy detection
        assert reranker.strategy == "causal_lm"
        
        # Verify Qwen-specific setup
        assert hasattr(reranker, 'token_true_id')
        assert hasattr(reranker, 'token_false_id')
        assert hasattr(reranker, 'prefix')
        assert hasattr(reranker, 'suffix')
        assert reranker.token_true_id == 100
        assert reranker.token_false_id == 200

    def test_format_instruction(self):
        """Test Qwen instruction formatting."""
        with patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True):
            reranker = object.__new__(TransformersRerankerModel)
            
            # Test with custom instruction
            formatted = reranker._format_instruction("Custom instruction", "test query", "test doc")
            expected = "<Instruct>: Custom instruction\n<Query>: test query\n<Document>: test doc"
            assert formatted == expected
            
            # Test with None instruction (default)
            formatted = reranker._format_instruction(None, "test query", "test doc")
            expected = "<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: test query\n<Document>: test doc"
            assert formatted == expected

    @patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True)
    def test_causal_lm_rerank_functionality(self):
        """Test reranking functionality for causal LM strategy."""
        with patch('esperanto.providers.reranker.transformers.torch'), \
             patch('esperanto.providers.reranker.transformers.AutoTokenizer'), \
             patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM'):
            
            mock_tokenizer = Mock()
            mock_tokenizer.convert_tokens_to_ids.side_effect = lambda token: 100 if token == "yes" else 200
            mock_tokenizer.decode.return_value = "decoded_text"
            
            with patch('esperanto.providers.reranker.transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
                 patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM.from_pretrained'):
                
                reranker = TransformersRerankerModel(model_name="Qwen/Qwen3-Reranker-0.6B")
                
                # Mock the scoring method
                with patch.object(reranker, '_score_all_pairs', return_value=[0.8, 0.3, 0.6]):
                    query = "What is machine learning?"
                    documents = [
                        "Machine learning is a subset of AI.",
                        "The weather is nice today.",
                        "Python is used in ML."
                    ]
                    
                    result = reranker.rerank(query, documents, top_k=2)
                    
                    assert isinstance(result, RerankResponse)
                    assert result.model == "Qwen/Qwen3-Reranker-0.6B"
                    assert len(result.results) == 2
                    assert result.results[0].document == documents[0]  # Highest score
                    assert result.results[1].document == documents[2]  # Second highest


class TestTransformersRerankerSentenceTransformersStrategy:
    """Test sentence transformers strategy (CrossEncoder, BAAI, Mixedbread v1)."""

    @patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True)
    @patch('esperanto.providers.reranker.transformers.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('esperanto.providers.reranker.transformers.CrossEncoder')
    def test_sentence_transformers_initialization(self, mock_cross_encoder):
        """Test initialization for sentence transformers strategy."""
        mock_model = Mock()
        mock_cross_encoder.return_value = mock_model
        
        reranker = TransformersRerankerModel(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        assert reranker.strategy == "sentence_transformers"
        assert reranker.model == mock_model
        mock_cross_encoder.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")

    @patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True)
    @patch('esperanto.providers.reranker.transformers.SENTENCE_TRANSFORMERS_AVAILABLE', False)
    def test_sentence_transformers_not_available(self):
        """Test error when sentence_transformers is not available."""
        with pytest.raises(ImportError, match="sentence-transformers library not available"):
            TransformersRerankerModel(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    @patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True)
    @patch('esperanto.providers.reranker.transformers.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('esperanto.providers.reranker.transformers.CrossEncoder')
    def test_sentence_transformers_rerank_functionality(self, mock_cross_encoder):
        """Test reranking functionality for sentence transformers strategy."""
        # Mock CrossEncoder.rank() method response
        mock_model = Mock()
        mock_model.rank.return_value = [
            {'corpus_id': 0, 'score': 0.8},
            {'corpus_id': 2, 'score': 0.6},
            {'corpus_id': 1, 'score': 0.3}
        ]
        mock_cross_encoder.return_value = mock_model
        
        reranker = TransformersRerankerModel(model_name="BAAI/bge-reranker-base")
        
        query = "What is machine learning?"
        documents = [
            "Machine learning is a subset of AI.",
            "The weather is nice today.",
            "Python is used in ML."
        ]
        
        result = reranker.rerank(query, documents, top_k=2)
        
        assert isinstance(result, RerankResponse)
        assert result.model == "BAAI/bge-reranker-base"
        assert len(result.results) == 2
        
        # Verify CrossEncoder.rank was called correctly
        mock_model.rank.assert_called_once_with(query, documents)


class TestTransformersRerankerSequenceClassificationStrategy:
    """Test sequence classification strategy (Jina models)."""

    @patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True)
    @patch('esperanto.providers.reranker.transformers.AutoModelForSequenceClassification')
    @patch('esperanto.providers.reranker.transformers.AutoTokenizer')
    def test_sequence_classification_initialization(self, mock_tokenizer_class, mock_model_class):
        """Test initialization for sequence classification strategy."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        reranker = TransformersRerankerModel(model_name="jinaai/jina-reranker-v2-base-multilingual")
        
        assert reranker.strategy == "sequence_classification"
        assert reranker.model == mock_model
        assert reranker.tokenizer == mock_tokenizer

    @patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True)
    @patch('esperanto.providers.reranker.transformers.AutoModelForSequenceClassification')
    @patch('esperanto.providers.reranker.transformers.AutoTokenizer')
    def test_sequence_classification_rerank_functionality(self, mock_tokenizer_class, mock_model_class):
        """Test reranking functionality for sequence classification strategy."""
        # Mock model and tokenizer
        mock_model = Mock()
        mock_model.compute_score.return_value = torch.tensor([0.8, 0.3, 0.6])
        mock_tokenizer = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        reranker = TransformersRerankerModel(model_name="jinaai/jina-reranker-v2-base-multilingual")
        
        query = "What is machine learning?"
        documents = [
            "Machine learning is a subset of AI.",
            "The weather is nice today.",
            "Python is used in ML."
        ]
        
        result = reranker.rerank(query, documents, top_k=2)
        
        assert isinstance(result, RerankResponse)
        assert result.model == "jinaai/jina-reranker-v2-base-multilingual"
        assert len(result.results) == 2
        
        # Verify compute_score was called with correct pairs
        expected_pairs = [[query, doc] for doc in documents]
        mock_model.compute_score.assert_called_once_with(expected_pairs, max_length=1024)


class TestTransformersRerankerMixedbreadV2Strategy:
    """Test mixedbread v2 strategy."""

    @patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True)
    @patch('esperanto.providers.reranker.transformers.MXBAI_AVAILABLE', True)
    def test_mixedbread_v2_initialization(self):
        """Test initialization for mixedbread v2 strategy."""
        with patch('esperanto.providers.reranker.transformers.MxbaiRerankV2') as mock_mxbai:
            mock_model = Mock()
            mock_mxbai.return_value = mock_model
            
            reranker = TransformersRerankerModel(model_name="mixedbread-ai/mxbai-rerank-base-v2")
            
            assert reranker.strategy == "mixedbread_v2"
            assert reranker.model == mock_model
            mock_mxbai.assert_called_once_with("mixedbread-ai/mxbai-rerank-base-v2")

    @patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True)
    def test_mixedbread_v2_not_available(self):
        """Test error when mxbai-rerank library is not available."""
        with patch('esperanto.providers.reranker.transformers.TransformersRerankerModel._load_mixedbread_v2_model') as mock_load:
            mock_load.side_effect = ImportError("mxbai-rerank library required")
            
            with pytest.raises(ImportError, match="mxbai-rerank library required"):
                TransformersRerankerModel(model_name="mixedbread-ai/mxbai-rerank-base-v2")

    @patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True)
    @patch('esperanto.providers.reranker.transformers.MXBAI_AVAILABLE', True)
    def test_mixedbread_v2_rerank_functionality(self):
        """Test reranking functionality for mixedbread v2 strategy."""
        with patch('esperanto.providers.reranker.transformers.MxbaiRerankV2') as mock_mxbai:
            # Mock RankResult objects with attributes
            mock_result_1 = Mock()
            mock_result_1.corpus_id = 0
            mock_result_1.score = 0.8
            
            mock_result_2 = Mock() 
            mock_result_2.corpus_id = 2
            mock_result_2.score = 0.6
            
            mock_result_3 = Mock()
            mock_result_3.corpus_id = 1
            mock_result_3.score = 0.3
            
            mock_model = Mock()
            mock_model.rank.return_value = [mock_result_1, mock_result_2, mock_result_3]
            mock_mxbai.return_value = mock_model
            
            reranker = TransformersRerankerModel(model_name="mixedbread-ai/mxbai-rerank-large-v2")
            
            query = "What is machine learning?"
            documents = [
                "Machine learning is a subset of AI.",
                "The weather is nice today.",
                "Python is used in ML."
            ]
            
            result = reranker.rerank(query, documents, top_k=2)
            
            assert isinstance(result, RerankResponse)
            assert result.model == "mixedbread-ai/mxbai-rerank-large-v2"
            assert len(result.results) == 2
            
            # Verify rank was called correctly
            mock_model.rank.assert_called_once_with(
                query, documents, return_documents=True, top_k=len(documents)
            )


class TestTransformersRerankerUniversalFunctionality:
    """Test universal functionality across all strategies."""

    @patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True)
    def test_models_property_includes_all_strategies(self):
        """Test that models property includes representatives from all strategies."""
        with patch('esperanto.providers.reranker.transformers.torch'), \
             patch('esperanto.providers.reranker.transformers.AutoTokenizer'), \
             patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM'):
            
            mock_tokenizer = Mock()
            mock_tokenizer.convert_tokens_to_ids.side_effect = lambda token: 100 if token == "yes" else 200
            mock_tokenizer.decode.return_value = "decoded_text"
            
            with patch('esperanto.providers.reranker.transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
                 patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM.from_pretrained'):
                
                reranker = TransformersRerankerModel(model_name="Qwen/Qwen3-Reranker-4B")
                models = reranker.models
                
                model_ids = [m.id for m in models]
                
                # Check representatives from each strategy
                assert "Qwen/Qwen3-Reranker-4B" in model_ids  # causal_lm
                assert "cross-encoder/ms-marco-MiniLM-L-6-v2" in model_ids  # sentence_transformers
                assert "BAAI/bge-reranker-base" in model_ids  # sentence_transformers
                assert "jinaai/jina-reranker-v2-base-multilingual" in model_ids  # sequence_classification
                assert "mixedbread-ai/mxbai-rerank-base-v2" in model_ids  # mixedbread_v2
                
                # Verify all are reranker type
                assert all(model.type == "reranker" for model in models)

    @patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True)
    def test_async_rerank_functionality(self):
        """Test async rerank method."""
        with patch('esperanto.providers.reranker.transformers.torch'), \
             patch('esperanto.providers.reranker.transformers.AutoTokenizer'), \
             patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM'):
            
            mock_tokenizer = Mock()
            mock_tokenizer.convert_tokens_to_ids.side_effect = lambda token: 100 if token == "yes" else 200
            mock_tokenizer.decode.return_value = "decoded_text"
            
            with patch('esperanto.providers.reranker.transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
                 patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM.from_pretrained'):
                
                reranker = TransformersRerankerModel(model_name="Qwen/Qwen3-Reranker-4B")
                
                # Mock the sync rerank method
                mock_result = RerankResponse(
                    results=[],
                    model="Qwen/Qwen3-Reranker-4B",
                    usage=None
                )
                
                with patch.object(reranker, 'rerank', return_value=mock_result) as mock_rerank:
                    import asyncio
                    
                    async def test_async():
                        result = await reranker.arerank("test", ["doc1", "doc2"])
                        return result
                    
                    result = asyncio.run(test_async())
                    
                    assert result == mock_result
                    mock_rerank.assert_called_once()

    def test_provider_property(self):
        """Test provider property."""
        with patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True):
            reranker = object.__new__(TransformersRerankerModel)
            assert reranker.provider == "transformers"

    def test_default_model(self):
        """Test default model property.""" 
        with patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True):
            reranker = object.__new__(TransformersRerankerModel)
            assert reranker._get_default_model() == "Qwen/Qwen3-Reranker-4B"

    def test_transformers_not_available(self):
        """Test handling when transformers library is not available."""
        with patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', False):
            with pytest.raises(ImportError, match="Transformers library not installed"):
                TransformersRerankerModel(model_name="any-model")

    @patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True)
    def test_langchain_integration(self):
        """Test LangChain integration works."""
        with patch('esperanto.providers.reranker.transformers.torch'), \
             patch('esperanto.providers.reranker.transformers.AutoTokenizer'), \
             patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM'):
            
            mock_tokenizer = Mock()
            mock_tokenizer.convert_tokens_to_ids.side_effect = lambda token: 100 if token == "yes" else 200
            mock_tokenizer.decode.return_value = "decoded_text"
            
            with patch('esperanto.providers.reranker.transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
                 patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM.from_pretrained'):
                
                reranker = TransformersRerankerModel(model_name="Qwen/Qwen3-Reranker-4B")
                
                # Test that to_langchain returns a wrapper
                langchain_reranker = reranker.to_langchain()
                assert langchain_reranker is not None
                assert hasattr(langchain_reranker, 'compress_documents')

    @patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True)
    def test_error_handling_in_score_all_pairs(self):
        """Test error handling when strategy methods fail."""
        with patch('esperanto.providers.reranker.transformers.torch'), \
             patch('esperanto.providers.reranker.transformers.AutoTokenizer'), \
             patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM'):
            
            mock_tokenizer = Mock()
            mock_tokenizer.convert_tokens_to_ids.side_effect = lambda token: 100 if token == "yes" else 200
            mock_tokenizer.decode.return_value = "decoded_text"
            
            with patch('esperanto.providers.reranker.transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
                 patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM.from_pretrained'):
                
                reranker = TransformersRerankerModel(model_name="Qwen/Qwen3-Reranker-4B")
                
                # Test that the causal_lm strategy handles errors gracefully
                # by directly calling the strategy method that has error handling
                scores = reranker._rerank_causal_lm("test query", ["doc1", "doc2"])
                
                # Should return zero scores when processing fails
                assert scores == [0.0, 0.0]


class TestTransformersRerankerInputValidation:
    """Test input validation across all strategies."""

    @patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True)
    def test_input_validation(self):
        """Test input validation works correctly."""
        with patch('esperanto.providers.reranker.transformers.torch'), \
             patch('esperanto.providers.reranker.transformers.AutoTokenizer'), \
             patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM'):
            
            mock_tokenizer = Mock()
            mock_tokenizer.convert_tokens_to_ids.side_effect = lambda token: 100 if token == "yes" else 200
            mock_tokenizer.decode.return_value = "decoded_text"
            
            with patch('esperanto.providers.reranker.transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
                 patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM.from_pretrained'):
                
                reranker = TransformersRerankerModel(model_name="Qwen/Qwen3-Reranker-4B")
                
                # Test empty query
                with pytest.raises(ValueError, match="Query cannot be empty"):
                    reranker.rerank("", ["doc1"])
                
                # Test empty documents
                with pytest.raises(ValueError, match="Documents list cannot be empty"):
                    reranker.rerank("query", [])
                
                # Test invalid top_k
                with pytest.raises(ValueError, match="top_k must be positive"):
                    reranker.rerank("query", ["doc1"], top_k=0)

    @patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True)
    def test_model_name_validation(self):
        """Test model name validation for security."""
        with patch('esperanto.providers.reranker.transformers.torch'), \
             patch('esperanto.providers.reranker.transformers.AutoTokenizer'), \
             patch('esperanto.providers.reranker.transformers.AutoModelForCausalLM'):
            
            # Test path traversal attempt
            with pytest.raises(ValueError, match="Invalid model name format"):
                TransformersRerankerModel(model_name="../malicious/model")
            
            # Test invalid characters
            with pytest.raises(ValueError, match="Model name contains invalid characters"):
                TransformersRerankerModel(model_name="model@name")
            
            # Test suspicious patterns (semicolon gets caught as invalid character first)
            with pytest.raises(ValueError, match="Model name contains invalid characters"):
                TransformersRerankerModel(model_name="model;rm -rf /")

    @patch('esperanto.providers.reranker.transformers.TRANSFORMERS_AVAILABLE', True) 
    def test_trust_remote_code_parameter(self):
        """Test trust_remote_code parameter handling."""
        with patch('esperanto.providers.reranker.transformers.AutoModelForSequenceClassification') as mock_model_class, \
             patch('esperanto.providers.reranker.transformers.AutoTokenizer'):
            
            mock_model = Mock()
            mock_tokenizer = Mock() 
            mock_model_class.from_pretrained.return_value = mock_model
            
            # Test default (False)
            reranker = TransformersRerankerModel(model_name="jinaai/jina-reranker-v2-base-multilingual")
            
            # Verify trust_remote_code=False was used
            mock_model_class.from_pretrained.assert_called_with(
                "jinaai/jina-reranker-v2-base-multilingual",
                torch_dtype="auto",
                trust_remote_code=False,
                cache_dir=None
            )
            
            # Test explicit True
            mock_model_class.reset_mock()
            reranker_trust = TransformersRerankerModel(
                model_name="jinaai/jina-reranker-v2-base-multilingual",
                trust_remote_code=True
            )
            
            # Verify trust_remote_code=True was used
            mock_model_class.from_pretrained.assert_called_with(
                "jinaai/jina-reranker-v2-base-multilingual", 
                torch_dtype="auto",
                trust_remote_code=True,
                cache_dir=None
            )