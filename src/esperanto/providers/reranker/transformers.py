"""Transformers reranker provider implementation."""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional

from esperanto.common_types import Model
from esperanto.common_types.reranker import RerankResponse, RerankResult
from .base import RerankerModel

# Optional transformers import with helpful error message
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None


@dataclass
class TransformersRerankerModel(RerankerModel):
    """Local transformers-based reranker using Qwen3-Reranker-4B."""
    
    device: Optional[str] = None
    max_length: int = 512
    cache_dir: Optional[str] = None

    def __post_init__(self):
        """Initialize Transformers reranker after dataclass initialization."""
        super().__post_init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library not installed. Install with: pip install esperanto[transformers]"
            )
        
        # No API key required for local models
        # Set cache directory if provided
        if self.cache_dir:
            os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        
        # Auto-detect device
        self.device = self._detect_device()
        
        # Get config values
        self.max_length = self._config.get("max_length", self.max_length)
        
        # Initialize model and tokenizer
        self._load_model()

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

    def _load_model(self):
        """Load the reranker model and tokenizer."""
        try:
            model_name = self.get_model_name()
            
            # Load tokenizer with left padding (required for Qwen reranker)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                padding_side='left'
            )
            
            # Load model as causal LM (not sequence classification)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32
            )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize Qwen-specific tokens and prompts
            self._setup_qwen_reranker()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.get_model_name()}: {str(e)}")

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
        self.qwen_max_length = 8192

    def _format_instruction(self, instruction: str, query: str, doc: str) -> str:
        """Format the instruction for Qwen reranker."""
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(self, pairs: List[str]) -> Dict[str, torch.Tensor]:
        """Process input pairs for Qwen reranker."""
        # Calculate effective max length for content
        content_max_length = self.qwen_max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        
        # Tokenize all pairs at once with proper padding and max_length
        inputs = self.tokenizer(
            pairs, 
            padding='max_length',
            truncation='longest_first',
            max_length=content_max_length,
            return_tensors="pt",
            return_attention_mask=True
        )
        
        # Add prefix and suffix tokens to each sequence
        batch_size = inputs['input_ids'].shape[0]
        new_input_ids = []
        new_attention_mask = []
        
        for i in range(batch_size):
            # Get the tokenized content (remove padding for processing)
            content_tokens = inputs['input_ids'][i]
            attention_mask = inputs['attention_mask'][i]
            
            # Find the actual content length (non-padded part)
            content_length = attention_mask.sum().item()
            actual_content = content_tokens[:content_length].tolist()
            
            # Create full sequence with prefix and suffix
            full_sequence = self.prefix_tokens + actual_content + self.suffix_tokens
            
            # Pad or truncate to max_length
            if len(full_sequence) > self.qwen_max_length:
                full_sequence = full_sequence[:self.qwen_max_length]
            else:
                # Pad with tokenizer's pad_token_id
                pad_length = self.qwen_max_length - len(full_sequence)
                full_sequence.extend([self.tokenizer.pad_token_id] * pad_length)
            
            # Create attention mask
            actual_length = len(self.prefix_tokens) + len(actual_content) + len(self.suffix_tokens)
            actual_length = min(actual_length, self.qwen_max_length)
            mask = [1] * actual_length + [0] * (self.qwen_max_length - actual_length)
            
            new_input_ids.append(full_sequence)
            new_attention_mask.append(mask)
        
        # Convert to tensors and move to device
        final_inputs = {
            'input_ids': torch.tensor(new_input_ids, dtype=torch.long).to(self.device),
            'attention_mask': torch.tensor(new_attention_mask, dtype=torch.long).to(self.device)
        }
        
        return final_inputs

    @torch.no_grad()
    def _compute_logits(self, inputs: Dict[str, torch.Tensor]) -> List[float]:
        """Compute logits and extract relevance scores."""
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def _score_all_pairs(self, query: str, documents: List[str]) -> List[float]:
        """Score all query-document pairs using Qwen reranker format.
        
        Args:
            query: The search query.
            documents: List of documents to score.
            
        Returns:
            List of relevance scores.
        """
        try:
            # Format all query-document pairs using Qwen instruction format
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
            pairs = [
                self._format_instruction(instruction, query, doc) 
                for doc in documents
            ]
            
            # Process inputs
            inputs = self._process_inputs(pairs)
            
            # Compute scores
            scores = self._compute_logits(inputs)
            
            return scores
            
        except Exception as e:
            # Log the error for debugging and return zero scores rather than failing
            import logging
            logging.warning(f"Transformers reranker error: {str(e)}")
            return [0.0] * len(documents)

    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: Optional[int] = None, 
        **kwargs
    ) -> RerankResponse:
        """Rerank documents using local Transformers model.
        
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
        
        # Score all query-document pairs
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
        """Async rerank documents using local Transformers model.
        
        This is implemented as a wrapper around the sync method using 
        a thread pool executor for CPU-bound operations.
        
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
        
        class LangChainTransformersReranker:
            def __init__(self, transformers_reranker):
                self.transformers_reranker = transformers_reranker
            
            def compress_documents(
                self, 
                documents: List[Document], 
                query: str, 
                callbacks: Optional[Callbacks] = None
            ) -> List[Document]:
                """Compress documents using Transformers reranker."""
                # Extract text content from documents
                texts = [doc.page_content for doc in documents]
                
                # Rerank using Transformers
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
        
        return LangChainTransformersReranker(self)

    def _get_default_model(self) -> str:
        """Get default Transformers model."""
        return "Qwen/Qwen3-Reranker-4B"

    @property
    def provider(self) -> str:
        """Provider name."""
        return "transformers"

    @property
    def models(self) -> List[Model]:
        """Available Transformers reranker models."""
        return [
            Model(
                id="Qwen/Qwen3-Reranker-4B",
                owned_by="Qwen",
                context_window=8192,
                type="reranker"
            ),
        ]