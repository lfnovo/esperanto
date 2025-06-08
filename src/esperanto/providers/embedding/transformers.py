"""Transformers embedding model provider."""

import asyncio
import functools
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import torch
from transformers import AutoModel, AutoTokenizer

from esperanto.providers.embedding.base import EmbeddingModel, Model


@dataclass
class PoolingConfig:
    """Configuration for embedding pooling strategy."""

    strategy: Literal["mean", "max", "cls"] = "mean"
    attention_mask: bool = True


class TransformersEmbeddingModel(EmbeddingModel):
    """Transformers embedding model implementation."""

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
            model_name: Name of the model to use (e.g., 'bert-base-uncased')
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

        # Initialize model and tokenizer
        self._initialize_model(quantize)

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
        """Create embeddings for the given texts.

        Args:
            texts: List of texts to create embeddings for
            batch_size: Batch size for processing
            **kwargs: Additional arguments to pass to the model

        Returns:
            List of embeddings, one for each input text
        """
        if not texts:
            raise ValueError("Texts cannot be empty")

        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Clean and tokenize texts
            cleaned_texts = [text.replace("\n", " ") for text in batch_texts]

            # Get tokenizer config from kwargs or use defaults
            tokenizer_config = {
                "padding": True,
                "truncation": True,
                "max_length": 512,  # Default BERT max length
                "return_tensors": "pt",
                **kwargs.get("tokenizer_config", {}),
            }

            encoded = self.tokenizer(cleaned_texts, **tokenizer_config)

            # Move inputs to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = self._pool_embeddings(
                    outputs, encoded.get("attention_mask")
                )

            # Convert to list of floats
            embeddings = embeddings.cpu().numpy()
            results.extend([embedding.tolist() for embedding in embeddings])

        return results

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
        return "bert-base-uncased"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "transformers"

    @property
    def models(self) -> List[Model]:
        """List all available models for this provider."""
        # For now, return a static list of popular models
        return [
            Model(
                id="bert-base-uncased",
                owned_by="Hugging Face",
                context_window=512,
                type="embedding",
            ),
            Model(
                id="intfloat/multilingual-e5-large-instruct",
                owned_by="Hugging Face",
                context_window=1024,
                type="embedding",
            ),
            Model(
                id="sentence-transformers/all-MiniLM-L6-v2",
                owned_by="Hugging Face",
                context_window=256,
                type="embedding",
            ),
        ]
