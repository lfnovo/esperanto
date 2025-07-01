"""Universal transformers reranker provider with 4-strategy architecture."""

import asyncio
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

from esperanto.common_types import Model
from esperanto.common_types.reranker import RerankResponse, RerankResult
from .base import RerankerModel

if TYPE_CHECKING:
    import torch

# Optional transformers import with helpful error message
try:
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    AutoModelForSequenceClassification = None
    AutoConfig = None

# Optional sentence_transformers import (part of transformers dependency)
try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    CrossEncoder = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Optional mxbai-rerank import
try:
    from mxbai_rerank import MxbaiRerankV2
    MXBAI_AVAILABLE = True
except ImportError:
    MxbaiRerankV2 = None
    MXBAI_AVAILABLE = False


# Define a no-op decorator when torch is not available
def no_grad_decorator(func):
    """Decorator that applies torch.no_grad() when available, otherwise returns function as-is."""
    if torch is not None and hasattr(torch, 'no_grad'):
        return torch.no_grad()(func)
    return func


@dataclass
class TransformersRerankerModel(RerankerModel):
    """Universal transformers-based reranker supporting multiple architectures.
    
    Supports 4 strategies:
    1. sentence_transformers - CrossEncoder models (cross-encoder/*, BAAI/bge-reranker-*, mixedbread v1)
    2. sequence_classification - AutoModelForSequenceClassification (jinaai/jina-reranker-*)
    3. causal_lm - AutoModelForCausalLM (Qwen/*-Reranker-*)
    4. mixedbread_v2 - mxbai-rerank library (mixedbread-ai/mxbai-rerank-*-v2)
    """
    
    device: Optional[str] = None
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False

    # Model strategy patterns for auto-detection
    MODEL_STRATEGY_PATTERNS = {
        "cross-encoder/": "sentence_transformers",
        "BAAI/bge-reranker-": "sentence_transformers", 
        "mixedbread-ai/mxbai-rerank-base-v1": "sentence_transformers",
        "mixedbread-ai/mxbai-rerank-large-v1": "sentence_transformers", 
        "mixedbread-ai/mxbai-rerank-xsmall-v1": "sentence_transformers",
        "mixedbread-ai/mxbai-rerank-base-v2": "mixedbread_v2",
        "mixedbread-ai/mxbai-rerank-large-v2": "mixedbread_v2",
        "jinaai/jina-reranker-": "sequence_classification",
        "Qwen/": "causal_lm"
    }

    def __post_init__(self):
        """Initialize universal transformers reranker after dataclass initialization."""
        super().__post_init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library not installed. Install with: pip install esperanto[transformers]"
            )
        
        # Set tokenizers parallelism to avoid fork warnings (especially in Jupyter)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        
        # Set cache directory if provided
        if self.cache_dir:
            os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        
        # Validate model name for security
        self._validate_model_name()
        
        # Auto-detect device
        self.device = self._detect_device()
        
        # Initialize model based on detected strategy
        self._load_model()

    def _validate_model_name(self):
        """Validate model name to prevent path traversal and other security issues."""
        model_name = self.get_model_name()
        
        # Check for path traversal attempts
        if ".." in model_name or "/" in model_name.replace("/", "", 1).replace("/", "", 1):
            # Allow at most one '/' for org/model format (e.g., "microsoft/model")
            parts = model_name.split("/")
            if len(parts) > 2:
                raise ValueError(f"Invalid model name format: {model_name}")
        
        # Check for invalid characters
        if not re.match(r"^[a-zA-Z0-9._/-]+$", model_name):
            raise ValueError(f"Model name contains invalid characters: {model_name}")
        
        # Check for suspicious patterns
        suspicious_patterns = ["\\", "..", "~", "$", "%", "@", "&", "|", ";", "`"]
        for pattern in suspicious_patterns:
            if pattern in model_name:
                raise ValueError(f"Model name contains suspicious pattern '{pattern}': {model_name}")

    def _detect_device(self) -> str:
        """Auto-detect the best available device."""
        if self.device:
            return self.device
            
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _get_model_strategy(self, model_name: str) -> str:
        """Automatically detect the correct strategy for the model."""
        # Check exact patterns first
        for pattern, strategy in self.MODEL_STRATEGY_PATTERNS.items():
            if model_name.startswith(pattern):
                return strategy
        
        # Fallback: inspect model config
        return self._detect_strategy_from_config(model_name)

    def _detect_strategy_from_config(self, model_name: str) -> str:
        """Detect strategy by inspecting model configuration."""
        try:
            config = AutoConfig.from_pretrained(model_name)
            
            # Map architectures to strategies
            if hasattr(config, 'architectures') and config.architectures:
                arch_str = str(config.architectures).lower()
                if "sequenceclassification" in arch_str:
                    return "sequence_classification"
                elif "causallm" in arch_str:
                    return "causal_lm"
                    
        except Exception:
            pass
        
        # If all detection fails
        supported_patterns = list(self.MODEL_STRATEGY_PATTERNS.keys())
        raise ValueError(
            f"Could not detect reranker strategy for model '{model_name}'. "
            f"Supported model patterns: {supported_patterns}"
        )

    def _load_model(self):
        """Load the appropriate model based on detected strategy."""
        model_name = self.get_model_name()
        strategy = self._get_model_strategy(model_name)
        
        if strategy == "sentence_transformers":
            self._load_sentence_transformers_model()
        elif strategy == "sequence_classification":
            self._load_sequence_classification_model()
        elif strategy == "causal_lm":
            self._load_causal_lm_model()
        elif strategy == "mixedbread_v2":
            self._load_mixedbread_v2_model()
        else:
            raise ValueError(f"Unknown reranker strategy: {strategy}")
        
        self.strategy = strategy

    def _load_sentence_transformers_model(self):
        """Load sentence_transformers CrossEncoder model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers library not available. Install with: pip install esperanto[transformers]"
            )
        
        try:
            self.model = CrossEncoder(self.get_model_name())
        except Exception as e:
            raise RuntimeError(f"Failed to load CrossEncoder model {self.get_model_name()}: {str(e)}")

    def _load_sequence_classification_model(self):
        """Load AutoModelForSequenceClassification model."""
        try:
            model_name = self.get_model_name()
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype="auto",
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load sequence classification model {self.get_model_name()}: {str(e)}")

    def _load_causal_lm_model(self):
        """Load AutoModelForCausalLM model (Qwen style)."""
        try:
            model_name = self.get_model_name()
            
            # Load tokenizer with left padding (required for Qwen reranker)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                padding_side='left'
            )
            
            # Load model as causal LM
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32
            )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            # Setup Qwen-specific configuration
            self._setup_qwen_reranker()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load causal LM model {self.get_model_name()}: {str(e)}")

    def _setup_qwen_reranker(self):
        """Setup Qwen-specific reranker configuration."""
        # Get token IDs for "yes" and "no" 
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        
        # Define prompt templates
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        
        # Encode prefix and suffix tokens
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        
        # Set max length for the model
        self.max_length = 8192

    def _load_mixedbread_v2_model(self):
        """Load Mixedbread v2 models with optional dependency."""
        if not MXBAI_AVAILABLE:
            raise ImportError(
                "mxbai-rerank library required for Mixedbread v2 models. "
                "Install with: pip install mxbai-rerank"
            )
        
        try:
            self.model = MxbaiRerankV2(self.get_model_name())
        except Exception as e:
            raise RuntimeError(f"Failed to load Mixedbread v2 model {self.get_model_name()}: {str(e)}")

    def _score_all_pairs(self, query: str, documents: List[str]) -> List[float]:
        """Universal scoring method that dispatches to appropriate strategy."""
        if self.strategy == "sentence_transformers":
            return self._rerank_sentence_transformers(query, documents)
        elif self.strategy == "sequence_classification":
            return self._rerank_sequence_classification(query, documents)
        elif self.strategy == "causal_lm":
            return self._rerank_causal_lm(query, documents)
        elif self.strategy == "mixedbread_v2":
            return self._rerank_mixedbread_v2(query, documents)
        else:
            raise ValueError(f"Unknown reranker strategy: {self.strategy}")

    def _rerank_sentence_transformers(self, query: str, documents: List[str]) -> List[float]:
        """Rerank using sentence_transformers CrossEncoder."""
        try:
            # Use CrossEncoder.rank method
            ranks = self.model.rank(query, documents)
            
            # Reorder scores to match original document order
            ordered_scores = [0.0] * len(documents)
            for rank in ranks:
                ordered_scores[rank['corpus_id']] = rank['score']
            
            return ordered_scores
            
        except Exception as e:
            import logging
            logging.warning(f"CrossEncoder reranker error: {str(e)}")
            return [0.0] * len(documents)

    def _rerank_sequence_classification(self, query: str, documents: List[str]) -> List[float]:
        """Rerank using AutoModelForSequenceClassification."""
        try:
            # Format query-document pairs 
            sentence_pairs = [[query, doc] for doc in documents]
            
            # Compute scores using model's compute_score method
            scores = self.model.compute_score(sentence_pairs, max_length=1024)
            
            return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
            
        except Exception as e:
            import logging
            logging.warning(f"Sequence classification reranker error: {str(e)}")
            return [0.0] * len(documents)

    def _rerank_causal_lm(self, query: str, documents: List[str]) -> List[float]:
        """Rerank using AutoModelForCausalLM (Qwen style)."""
        try:
            # Format all query-document pairs using Qwen instruction format
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
            pairs = [
                self._format_instruction(instruction, query, doc) 
                for doc in documents
            ]
            
            # Process inputs and compute scores
            inputs = self._process_inputs(pairs)
            scores = self._compute_logits(inputs)
            
            return scores
            
        except Exception as e:
            import logging
            logging.warning(f"Causal LM reranker error: {str(e)}")
            return [0.0] * len(documents)

    def _rerank_mixedbread_v2(self, query: str, documents: List[str]) -> List[float]:
        """Rerank using mxbai-rerank library."""
        try:
            # Use mxbai library rank method
            results = self.model.rank(query, documents, return_documents=True, top_k=len(documents))
            
            # Extract scores and reorder to match original document order
            scores = [0.0] * len(documents)
            for i, result in enumerate(results):
                # mxbai returns RankResult objects with attributes, not dicts
                original_index = getattr(result, 'corpus_id', None)
                score = getattr(result, 'score', 0.0)
                
                if original_index is not None and 0 <= original_index < len(scores):
                    scores[original_index] = score
                else:
                    # Handle missing or invalid corpus_id
                    import logging
                    logging.warning(f"Invalid or missing corpus_id for result {i}: {original_index}")
            
            return scores
            
        except Exception as e:
            import logging
            logging.warning(f"Mixedbread v2 reranker error: {str(e)}")
            return [0.0] * len(documents)

    def _format_instruction(self, instruction: Optional[str], query: str, doc: str) -> str:
        """Format the instruction for Qwen reranker."""
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(self, pairs: List[str]) -> Dict[str, "torch.Tensor"]:
        """Process input pairs for Qwen reranker."""
        # Build full text strings with prefix and suffix for direct tokenization
        # This avoids the warning about using encode() followed by pad()
        prefix_text = self.tokenizer.decode(self.prefix_tokens, skip_special_tokens=False)
        suffix_text = self.tokenizer.decode(self.suffix_tokens, skip_special_tokens=False)
        
        full_texts = [prefix_text + pair + suffix_text for pair in pairs]
        
        # Use the faster __call__ method as recommended by the warning
        inputs = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True
        )
        
        # Move to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        
        return inputs

    @no_grad_decorator
    def _compute_logits(self, inputs: Dict[str, "torch.Tensor"]) -> List[float]:
        """Compute logits and extract relevance scores for Qwen."""
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: Optional[int] = None, 
        **kwargs
    ) -> RerankResponse:
        """Rerank documents using universal transformers model.
        
        Args:
            query: The search query to rank documents against.
            documents: List of documents to rerank.
            top_k: Maximum number of results to return.
            **kwargs: Additional arguments.
            
        Returns:
            RerankResponse with ranked results.
        """
        # Validate inputs
        query, documents, top_k = self._validate_inputs(query, documents, top_k)
        
        # Score all query-document pairs using appropriate strategy
        raw_scores = self._score_all_pairs(query, documents)
        
        # Normalize scores using base class method
        normalized_scores = self._normalize_scores(raw_scores)
        
        # Create results with original indices
        results = []
        for i, (document, score) in enumerate(zip(documents, normalized_scores)):
            results.append(RerankResult(
                index=i,
                document=document,
                relevance_score=score
            ))
        
        # Sort by relevance score (highest first)
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Apply top_k limit
        if top_k < len(results):
            results = results[:top_k]
        
        return RerankResponse(
            results=results,
            model=self.get_model_name(),
            usage=None  # Local models don't have usage stats
        )

    async def arerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: Optional[int] = None, 
        **kwargs
    ) -> RerankResponse:
        """Async rerank documents using universal transformers model.
        
        Args:
            query: The search query to rank documents against.
            documents: List of documents to rerank.
            top_k: Maximum number of results to return.
            **kwargs: Additional arguments.
            
        Returns:
            RerankResponse with ranked results.
        """
        # Run the sync rerank method in a thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor, 
                self.rerank, 
                query, 
                documents, 
                top_k
            )
        return result

    def to_langchain(self):
        """Convert to LangChain-compatible reranker."""
        try:
            from langchain.schema import Document
            from langchain_core.callbacks.manager import Callbacks
        except ImportError:
            raise ImportError(
                "LangChain not installed. Install with: pip install langchain"
            )
        
        class LangChainUniversalReranker:
            def __init__(self, transformers_reranker):
                self.transformers_reranker = transformers_reranker
            
            def compress_documents(
                self, 
                documents: List[Document], 
                query: str, 
                callbacks: Optional[Callbacks] = None
            ) -> List[Document]:
                """Compress documents using universal transformers reranker."""
                # Note: callbacks parameter is part of LangChain interface but not used in local processing
                _ = callbacks  # Acknowledge parameter to avoid warnings
                
                # Extract text content from documents
                texts = [doc.page_content for doc in documents]
                
                # Rerank using transformers
                rerank_response = self.transformers_reranker.rerank(query, texts)
                
                # Convert back to LangChain documents
                reranked_docs = []
                for result in rerank_response.results:
                    if result.index < len(documents):
                        original_doc = documents[result.index]
                        # Add relevance score to metadata
                        new_metadata = original_doc.metadata.copy()
                        new_metadata["relevance_score"] = result.relevance_score
                        
                        reranked_docs.append(Document(
                            page_content=original_doc.page_content,
                            metadata=new_metadata
                        ))
                
                return reranked_docs
        
        return LangChainUniversalReranker(self)

    def _get_default_model(self) -> str:
        """Get default transformers model."""
        return "Qwen/Qwen3-Reranker-4B"

    @property
    def provider(self) -> str:
        """Provider name."""
        return "transformers"

    @property 
    def models(self) -> List[Model]:
        """Available universal transformers reranker models."""
        return [
            # Qwen models (causal_lm strategy)
            Model(
                id="Qwen/Qwen3-Reranker-4B",
                owned_by="Qwen",
                context_window=8192,
                type="reranker"
            ),
            Model(
                id="Qwen/Qwen3-Reranker-0.6B",
                owned_by="Qwen",
                context_window=8192,
                type="reranker"
            ),
            # CrossEncoder models (sentence_transformers strategy)
            Model(
                id="cross-encoder/ms-marco-MiniLM-L-6-v2",
                owned_by="microsoft",
                context_window=512,
                type="reranker"
            ),
            Model(
                id="cross-encoder/ms-marco-electra-base",
                owned_by="microsoft",
                context_window=512,
                type="reranker"
            ),
            Model(
                id="BAAI/bge-reranker-base",
                owned_by="BAAI",
                context_window=512,
                type="reranker"
            ),
            Model(
                id="BAAI/bge-reranker-large",
                owned_by="BAAI",
                context_window=512,
                type="reranker"
            ),
            # Mixedbread v1 models (sentence_transformers strategy)
            Model(
                id="mixedbread-ai/mxbai-rerank-xsmall-v1",
                owned_by="mixedbread-ai",
                context_window=512,
                type="reranker"
            ),
            Model(
                id="mixedbread-ai/mxbai-rerank-base-v1",
                owned_by="mixedbread-ai",
                context_window=512,
                type="reranker"
            ),
            Model(
                id="mixedbread-ai/mxbai-rerank-large-v1",
                owned_by="mixedbread-ai",
                context_window=512,
                type="reranker"
            ),
            # Jina models (sequence_classification strategy)
            Model(
                id="jinaai/jina-reranker-v2-base-multilingual",
                owned_by="jinaai",
                context_window=1024,
                type="reranker"
            ),
            # Mixedbread v2 models (mixedbread_v2 strategy)
            Model(
                id="mixedbread-ai/mxbai-rerank-base-v2",
                owned_by="mixedbread-ai",
                context_window=512,
                type="reranker"
            ),
            Model(
                id="mixedbread-ai/mxbai-rerank-large-v2",
                owned_by="mixedbread-ai",
                context_window=512,
                type="reranker"
            ),
        ]