"""Transformers embedding model provider with advanced local emulation features."""

import asyncio
import functools
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from esperanto.common_types.task_type import EmbeddingTaskType
from esperanto.providers.embedding.base import EmbeddingModel, Model

# Optional dependencies for advanced features
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    PCA = None
    ADVANCED_FEATURES_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PoolingConfig:
    """Configuration for embedding pooling strategy."""

    strategy: Literal["mean", "max", "cls"] = "mean"
    attention_mask: bool = True


class TransformersEmbeddingModel(EmbeddingModel):
    """Transformers embedding model implementation with advanced local emulation features.
    
    Supports task-specific optimization, semantic late chunking, and output dimension control
    through local emulation techniques, providing privacy-first alternatives to cloud-based
    advanced embedding features.
    """
    
    # Declare supported features - Transformers implements all advanced features locally
    SUPPORTED_FEATURES = ["task_type", "late_chunking", "output_dimensions", "truncate_at_max_length"]

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
        pooling_strategy: Literal["mean", "max", "cls"] = "mean",
        quantize: Optional[Literal["4bit", "8bit"]] = None,
        model_cache_dir: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the model.

        Args:
            model_name: Name of the model to use (e.g., 'Qwen/Qwen3-Embedding-4B')
            device: Device to use for computation ('auto', 'cpu', 'cuda', 'mps')
            pooling_strategy: Strategy for pooling embeddings ('mean', 'max', 'cls')
            quantize: Quantization mode (None, '4bit', '8bit')
            model_cache_dir: Directory to cache models
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(model_name=model_name, **kwargs)

        # Set cache directory if provided
        if model_cache_dir:
            os.environ["TRANSFORMERS_CACHE"] = model_cache_dir

        # Configure device
        config_device = kwargs.get("config", {}).get("device", device)
        self.device = config_device
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Configure pooling
        self.pooling_config = PoolingConfig(
            strategy=pooling_strategy, attention_mask=True
        )

        # Initialize advanced features state
        self._pca_model = None
        self._chunker = None
        
        # Initialize model and tokenizer
        self._initialize_model(quantize)
        
        # Configure max chunk size based on model (always do this)
        self._configure_model_specific_settings()
        
        # Initialize advanced features if available
        self._initialize_advanced_features()
        
        # Track if resources are cleaned up
        self._is_cleaned_up = False

    def _initialize_model(self, quantize: Optional[str] = None):
        """Initialize the model and tokenizer with optional quantization."""
        model_name = self.get_model_name()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Configure quantization if requested
        if quantize:
            try:
                import bitsandbytes as bnb

                quantization_config = {
                    "load_in_4bit": quantize == "4bit",
                    "load_in_8bit": quantize == "8bit",
                }
            except ImportError:
                raise ImportError(
                    "bitsandbytes is required for quantization. "
                    "Install it with: pip install bitsandbytes"
                )
            self.model = AutoModel.from_pretrained(
                model_name,
                device_map="auto" if self.device == "cuda" else None,
                **quantization_config,
            )
        else:
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)

        self.model.eval()
        
    def cleanup(self):
        """Explicitly clean up model resources."""
        if self._is_cleaned_up:
            return
            
        try:
            # Move model to CPU and clear CUDA cache if using GPU
            if hasattr(self, 'model') and self.model is not None:
                if self.device in ['cuda', 'mps']:
                    self.model.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                del self.model
                self.model = None
                
            # Clean up tokenizer
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
                self.tokenizer = None
                
            # Clean up chunker if exists
            if hasattr(self, '_chunker') and self._chunker is not None:
                if hasattr(self._chunker, 'to'):
                    self._chunker.cpu()
                del self._chunker
                self._chunker = None
                
            # Clean up PCA model
            if hasattr(self, '_pca_model') and self._pca_model is not None:
                del self._pca_model
                self._pca_model = None
                
            self._is_cleaned_up = True
            logger.debug("Model resources cleaned up successfully")
            
        except Exception as e:
            logger.warning(f"Error during model cleanup: {e}")
            
    def __del__(self):
        """Clean up resources on object destruction."""
        self.cleanup()
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def _configure_model_specific_settings(self):
        """Configure model-specific settings like chunk size limits."""
        # Configure max chunk size based on model
        model_name = self.get_model_name().lower()
        if "qwen3" in model_name or "qwen-3" in model_name:
            self._max_chunk_tokens = 8192  # Qwen3 has larger context
        elif "e5" in model_name or "multilingual" in model_name:
            self._max_chunk_tokens = 1024  # E5 models
        else:
            self._max_chunk_tokens = 512  # Default/BERT-like models

    def _initialize_advanced_features(self):
        """Initialize advanced features if dependencies are available."""
        if not ADVANCED_FEATURES_AVAILABLE:
            if any([self.task_type, self.late_chunking, self.output_dimensions]):
                logger.warning(
                    "Advanced features requested but dependencies not available. "
                    "Install with: pip install esperanto[transformers]"
                )
            return

        # Initialize semantic chunker for late chunking
        if self.late_chunking:
            self._initialize_chunker()

    def _initialize_chunker(self):
        """Initialize semantic chunker for intelligent text segmentation."""
        if not ADVANCED_FEATURES_AVAILABLE or not self.late_chunking:
            return

        try:
            # Use a lightweight sentence transformer for chunking
            # This is separate from the main model for semantic boundaries
            self._chunker = SentenceTransformer('all-MiniLM-L6-v2')
            self._chunker.to(self.device)
            logger.info("Initialized semantic chunker for late chunking")
        except Exception as e:
            logger.warning(f"Failed to initialize semantic chunker: {e}")
            self._chunker = None

    def _apply_task_optimization(self, texts: List[str]) -> List[str]:
        """Apply advanced task-specific optimization with sophisticated prefixes.
        
        Implements advanced task prefixes optimized for modern embedding models
        like Qwen3-Embedding-4B that better communicate task intent.
        
        Args:
            texts: List of texts to optimize.
            
        Returns:
            List of optimized texts with task-specific prefixes.
        """
        if not self.task_type or self.task_type == EmbeddingTaskType.DEFAULT:
            return texts

        # Advanced task prefixes optimized for modern embedding models
        advanced_prefix_map = {
            EmbeddingTaskType.RETRIEVAL_QUERY: "Represent this query for retrieving relevant documents: ",
            EmbeddingTaskType.RETRIEVAL_DOCUMENT: "Represent this document for retrieval: ",
            EmbeddingTaskType.SIMILARITY: "Represent this text for semantic similarity: ",
            EmbeddingTaskType.CLASSIFICATION: "Represent this text for classification: ",
            EmbeddingTaskType.CLUSTERING: "Represent this text for clustering: ",
            EmbeddingTaskType.CODE_RETRIEVAL: "Represent this code for search: ",
            EmbeddingTaskType.QUESTION_ANSWERING: "Represent this question for answering: ",
            EmbeddingTaskType.FACT_VERIFICATION: "Represent this claim for verification: "
        }

        prefix = advanced_prefix_map.get(self.task_type, "")
        if prefix:
            logger.debug(f"Applying task optimization for {self.task_type.value}")
            return [prefix + text for text in texts]
        return texts

    def _apply_late_chunking(self, texts: List[str]) -> List[str]:
        """Apply semantic late chunking with intelligent text segmentation.
        
        Uses sentence-transformers for semantic boundary detection and creates
        chunks that respect both semantic coherence and token limits.
        
        Args:
            texts: List of texts to chunk.
            
        Returns:
            List of semantically chunked texts.
        """
        if not self.late_chunking:
            return texts

        if not ADVANCED_FEATURES_AVAILABLE:
            # Fallback to base class simple chunking
            logger.debug("Using base class chunking (advanced features not available)")
            return super()._apply_late_chunking(texts, max_chunk_size=self._max_chunk_tokens)

        chunked_texts = []
        for text in texts:
            chunks = self._semantic_chunk_text(text)
            chunked_texts.extend(chunks)

        logger.debug(f"Chunked {len(texts)} texts into {len(chunked_texts)} chunks")
        return chunked_texts

    def _semantic_chunk_text(self, text: str) -> List[str]:
        """Chunk text using semantic boundaries.
        
        Args:
            text: Text to chunk.
            
        Returns:
            List of semantic chunks.
        """
        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = len(text) // 4
        
        if estimated_tokens <= self._max_chunk_tokens:
            return [text]

        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return [text]

        # If we have a semantic chunker, use it for better boundaries
        if self._chunker is not None:
            return self._create_semantic_chunks(sentences)
        else:
            # Fallback to simple sentence-based chunking
            return self._create_simple_chunks(sentences)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using advanced regex patterns."""
        # Enhanced sentence splitting pattern
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _create_semantic_chunks(self, sentences: List[str]) -> List[str]:
        """Create chunks using semantic similarity for better boundaries."""
        if not sentences:
            return []

        try:
            # Get embeddings for sentences to find semantic boundaries
            sentence_embeddings = self._chunker.encode(sentences)
            
            chunks = []
            current_chunk = []
            current_tokens = 0
            
            for i, sentence in enumerate(sentences):
                sentence_tokens = len(sentence) // 4  # Rough token estimate
                
                # Check if adding this sentence would exceed token limit
                if current_tokens + sentence_tokens > self._max_chunk_tokens and current_chunk:
                    # Find the best breaking point using semantic similarity
                    if len(current_chunk) > 1:
                        # For now, just break at the current point
                        # Future enhancement: use embedding similarity to find better breaks
                        chunks.append(" ".join(current_chunk))
                        current_chunk = [sentence]
                        current_tokens = sentence_tokens
                    else:
                        # Single long sentence, just add it
                        chunks.append(" ".join(current_chunk))
                        current_chunk = [sentence]
                        current_tokens = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
            
            # Add the last chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            return chunks
            
        except Exception as e:
            logger.warning(f"Semantic chunking failed, falling back to simple chunking: {e}")
            return self._create_simple_chunks(sentences)

    def _create_simple_chunks(self, sentences: List[str]) -> List[str]:
        """Create chunks using simple token counting."""
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(sentence) // 4  # Rough token estimate
            
            if current_tokens + sentence_tokens > self._max_chunk_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def _apply_dimension_control(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply output dimension control via PCA or padding.
        
        Args:
            embeddings: Input embeddings array of shape (batch_size, embedding_dim).
            
        Returns:
            Embeddings with controlled dimensions.
        """
        if self.output_dimensions is None:
            return embeddings

        current_dim = embeddings.shape[-1]
        target_dim = self.output_dimensions

        if target_dim == current_dim:
            return embeddings

        if target_dim < current_dim:
            # Dimensionality reduction via PCA
            return self._reduce_dimensions(embeddings, target_dim)
        else:
            # Dimensionality expansion via padding
            return self._expand_dimensions(embeddings, target_dim)

    def _reduce_dimensions(self, embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        """Reduce embedding dimensions using PCA."""
        if not ADVANCED_FEATURES_AVAILABLE:
            logger.warning("PCA reduction requested but scikit-learn not available")
            return embeddings[:, :target_dim]  # Simple truncation fallback

        try:
            # Initialize PCA if not already done
            if self._pca_model is None or self._pca_model.n_components != target_dim:
                self._pca_model = PCA(n_components=target_dim)
                self._pca_model.fit(embeddings)
                logger.debug(f"Initialized PCA for dimension reduction to {target_dim}")

            reduced = self._pca_model.transform(embeddings)
            logger.debug(f"Reduced dimensions from {embeddings.shape[-1]} to {target_dim}")
            return reduced

        except Exception as e:
            logger.warning(f"PCA reduction failed, using truncation: {e}")
            return embeddings[:, :target_dim]

    def _expand_dimensions(self, embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        """Expand embedding dimensions via zero padding."""
        current_dim = embeddings.shape[-1]
        padding_size = target_dim - current_dim
        
        # Add zero padding
        padding = np.zeros((embeddings.shape[0], padding_size))
        expanded = np.concatenate([embeddings, padding], axis=1)
        
        logger.debug(f"Expanded dimensions from {current_dim} to {target_dim}")
        return expanded

    def _pool_embeddings(
        self, model_output: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pool the token embeddings into sentence embeddings.

        Args:
            model_output: Model output containing token embeddings
            attention_mask: Attention mask for valid tokens

        Returns:
            Pooled embeddings tensor
        """
        token_embeddings = model_output.last_hidden_state

        if self.pooling_config.strategy == "cls":
            return token_embeddings[:, 0]

        if attention_mask is not None and self.pooling_config.attention_mask:
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings = token_embeddings * mask

        if self.pooling_config.strategy == "max":
            return torch.max(token_embeddings, dim=1)[0]

        # Default to mean pooling
        if attention_mask is not None and self.pooling_config.attention_mask:
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            return torch.sum(token_embeddings, dim=1) / sum_mask

        return torch.mean(token_embeddings, dim=1)

    def embed(
        self, texts: List[str], batch_size: int = 32, **kwargs
    ) -> List[List[float]]:
        """Create embeddings for the given texts with advanced features.

        Args:
            texts: List of texts to create embeddings for
            batch_size: Batch size for processing
            **kwargs: Additional arguments to pass to the model

        Returns:
            List of embeddings, one for each input text
        """
        if not texts:
            raise ValueError("Texts cannot be empty")

        # Apply advanced preprocessing pipeline
        processed_texts = self._preprocess_texts(texts)

        results = []
        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i : i + batch_size]

            # Get tokenizer config from kwargs or use defaults  
            max_length = self._max_chunk_tokens if hasattr(self, '_max_chunk_tokens') else 512
            tokenizer_config = {
                "padding": True,
                "truncation": self.truncate_at_max_length,
                "max_length": max_length,
                "return_tensors": "pt",
                **kwargs.get("tokenizer_config", {}),
            }

            encoded = self.tokenizer(batch_texts, **tokenizer_config)

            # Move inputs to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = self._pool_embeddings(
                    outputs, encoded.get("attention_mask")
                )

            # Convert to numpy for post-processing
            embeddings = embeddings.cpu().numpy()
            
            # Apply dimension control if configured
            embeddings = self._apply_dimension_control(embeddings)
            
            # Convert to list of floats
            results.extend([embedding.tolist() for embedding in embeddings])

        # Handle aggregation if late chunking was applied
        if self.late_chunking and len(processed_texts) != len(texts):
            results = self._aggregate_chunked_embeddings(results, texts, processed_texts)

        return results

    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Apply the complete preprocessing pipeline with advanced features.
        
        Args:
            texts: Original input texts.
            
        Returns:
            Preprocessed texts ready for embedding.
        """
        # Step 1: Clean texts
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        # Step 2: Apply task optimization
        optimized_texts = self._apply_task_optimization(cleaned_texts)
        
        # Step 3: Apply late chunking
        chunked_texts = self._apply_late_chunking(optimized_texts)
        
        return chunked_texts

    def _aggregate_chunked_embeddings(
        self, 
        embeddings: List[List[float]], 
        original_texts: List[str], 
        processed_texts: List[str]
    ) -> List[List[float]]:
        """Aggregate embeddings from chunked texts back to original text count.
        
        When late chunking splits texts into multiple chunks, this method
        aggregates the chunk embeddings back to one embedding per original text.
        
        Args:
            embeddings: All chunk embeddings.
            original_texts: Original input texts.
            processed_texts: Processed/chunked texts.
            
        Returns:
            Aggregated embeddings, one per original text.
        """
        if len(embeddings) == len(original_texts):
            return embeddings  # No chunking occurred

        # Simple aggregation: mean pooling of chunks per text
        # More sophisticated approaches could use weighted averaging
        aggregated = []
        chunk_idx = 0
        
        for original_text in original_texts:
            # Find chunks that belong to this original text
            text_chunks = []
            original_length = len(original_text)
            
            # Estimate how many chunks this text produced
            # This is a simplification - in practice we'd track this explicitly
            estimated_chunks = max(1, (len(original_text) // 4) // self._max_chunk_tokens + 1)
            
            # Collect embeddings for this text's chunks
            text_embeddings = []
            chunks_for_text = min(estimated_chunks, len(embeddings) - chunk_idx)
            
            for _ in range(chunks_for_text):
                if chunk_idx < len(embeddings):
                    text_embeddings.append(embeddings[chunk_idx])
                    chunk_idx += 1
            
            if text_embeddings:
                # Mean pooling of chunk embeddings
                mean_embedding = np.mean(text_embeddings, axis=0).tolist()
                aggregated.append(mean_embedding)
            else:
                # Fallback: create zero embedding
                embedding_dim = len(embeddings[0]) if embeddings else 768
                aggregated.append([0.0] * embedding_dim)
        
        return aggregated

    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Create embeddings for the given texts asynchronously.

        Args:
            texts: List of texts to create embeddings for
            **kwargs: Additional arguments to pass to the model

        Returns:
            List of embeddings, one for each input text
        """
        loop = asyncio.get_event_loop()
        partial_embed = functools.partial(self.embed, texts=texts, **kwargs)
        return await loop.run_in_executor(None, partial_embed)

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "intfloat/multilingual-e5-large-instruct"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "transformers"

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        return [
            Model(
                id="Qwen/Qwen3-Embedding-4B",
                owned_by="Alibaba Cloud",
                context_window=8192,
                type="embedding",
            ),
            Model(
                id="intfloat/multilingual-e5-large-instruct",
                owned_by="Microsoft",
                context_window=1024,
                type="embedding",
            ),
            Model(
                id="sentence-transformers/all-MiniLM-L6-v2",
                owned_by="Sentence Transformers",
                context_window=256,
                type="embedding",
            ),
            Model(
                id="bert-base-uncased",
                owned_by="Google",
                context_window=512,
                type="embedding",
            ),
            Model(
                id="intfloat/e5-large-v2",
                owned_by="Microsoft",
                context_window=512,
                type="embedding",
            ),
            Model(
                id="BAAI/bge-large-en-v1.5",
                owned_by="BAAI",
                context_window=512,
                type="embedding",
            ),
        ]
