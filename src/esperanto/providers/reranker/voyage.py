"""Voyage reranker provider implementation."""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from esperanto.common_types import Model
from esperanto.common_types.reranker import RerankResponse, RerankResult
from esperanto.common_types.response import Usage
from .base import RerankerModel


@dataclass
class VoyageRerankerModel(RerankerModel):
    """Voyage reranker provider with HTTP API integration."""

    def __post_init__(self):
        """Initialize Voyage reranker after dataclass initialization."""
        super().__post_init__()
        
        # Authentication
        self.api_key = self.api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Voyage API key not found. Set VOYAGE_API_KEY environment variable or pass api_key parameter."
            )
        
        # API configuration
        self.base_url = self.base_url or "https://api.voyageai.com/v1"
        
        # Get timeout from config or use default
        timeout = self._config.get("timeout", 30.0)
        
        # HTTP client setup
        self.client = httpx.Client(timeout=timeout)
        self.async_client = httpx.AsyncClient(timeout=timeout)

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers for Voyage API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _build_request_payload(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Build request payload for Voyage rerank API.
        
        Args:
            query: The search query.
            documents: List of documents to rerank.
            top_k: Maximum number of results to return.
            **kwargs: Additional arguments.
            
        Returns:
            Request payload dict.
        """
        payload = {
            "model": self.get_model_name(),
            "query": query,
            "documents": documents,
            "top_k": top_k,
            "return_documents": True  # Always return documents for consistent interface
        }
        
        return payload

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle error responses from Voyage API.
        
        Args:
            response: HTTP response object.
            
        Raises:
            RuntimeError: With details from the error response.
        """
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", "Unknown error")
            error_type = error_data.get("error", {}).get("type", "Unknown")
            raise RuntimeError(f"Voyage API error ({error_type}): {error_message}")
        except (KeyError, ValueError):
            raise RuntimeError(f"Voyage API error: {response.status_code} - {response.text}")

    def _parse_response(self, response_data: Dict[str, Any], documents: List[str]) -> RerankResponse:
        """Parse Voyage API response into standardized format.
        
        Args:
            response_data: Raw response from Voyage API.
            documents: Original documents list for fallback.
            
        Returns:
            Standardized RerankResponse.
        """
        results = []
        raw_results = response_data.get("data", [])
        
        # Extract raw scores for normalization
        raw_scores = [result.get("relevance_score", 0.0) for result in raw_results]
        normalized_scores = self._normalize_scores(raw_scores)
        
        for i, result in enumerate(raw_results):
            index = result.get("index", i)
            document = result.get("document")
            
            # Use original document if not returned in response
            if document is None and index < len(documents):
                document = documents[index]
            elif document is None:
                document = ""
            
            results.append(RerankResult(
                index=index,
                document=document,
                relevance_score=normalized_scores[i] if i < len(normalized_scores) else 0.0
            ))
        
        # Create usage info if available
        usage = None
        if "usage" in response_data:
            usage_data = response_data["usage"]
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )
        
        return RerankResponse(
            results=results,
            model=response_data.get("model", self.get_model_name()),
            usage=usage
        )

    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: Optional[int] = None, 
        **kwargs
    ) -> RerankResponse:
        """Rerank documents using Voyage API.
        
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
        
        # Build request
        payload = self._build_request_payload(query, documents, top_k, **kwargs)
        
        try:
            response = self.client.post(
                f"{self.base_url}/rerank",
                json=payload,
                headers=self._get_headers()
            )
            
            if response.status_code != 200:
                self._handle_error(response)
                
            response_data = response.json()
            return self._parse_response(response_data, documents)
            
        except httpx.TimeoutException:
            raise RuntimeError("Request to Voyage API timed out")
        except httpx.RequestError as e:
            raise RuntimeError(f"Network error calling Voyage API: {str(e)}")

    async def arerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: Optional[int] = None, 
        **kwargs
    ) -> RerankResponse:
        """Async rerank documents using Voyage API.
        
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
        
        # Build request
        payload = self._build_request_payload(query, documents, top_k, **kwargs)
        
        try:
            response = await self.async_client.post(
                f"{self.base_url}/rerank",
                json=payload,
                headers=self._get_headers()
            )
            
            if response.status_code != 200:
                self._handle_error(response)
                
            response_data = response.json()
            return self._parse_response(response_data, documents)
            
        except httpx.TimeoutException:
            raise RuntimeError("Request to Voyage API timed out")
        except httpx.RequestError as e:
            raise RuntimeError(f"Network error calling Voyage API: {str(e)}")

    def to_langchain(self):
        """Convert to LangChain-compatible reranker."""
        try:
            from langchain.schema import Document
            from langchain_core.callbacks.manager import Callbacks
        except ImportError:
            raise ImportError(
                "LangChain not installed. Install with: pip install langchain"
            )
        
        class LangChainVoyageReranker:
            def __init__(self, voyage_reranker):
                self.voyage_reranker = voyage_reranker
            
            def compress_documents(
                self, 
                documents: List[Document], 
                query: str, 
                callbacks: Optional[Callbacks] = None
            ) -> List[Document]:
                """Compress documents using Voyage reranker."""
                # Extract text content from documents
                texts = [doc.page_content for doc in documents]
                
                # Rerank using Voyage
                rerank_response = self.voyage_reranker.rerank(query, texts)
                
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
        
        return LangChainVoyageReranker(self)

    def _get_default_model(self) -> str:
        """Get default Voyage model."""
        return "rerank-2"

    @property
    def provider(self) -> str:
        """Provider name."""
        return "voyage"

    @property
    def models(self) -> List[Model]:
        """Available Voyage reranker models."""
        return [
            Model(
                id="rerank-2",
                owned_by="voyage",
                context_window=4000,
                type="reranker"
            ),
            Model(
                id="rerank-1",
                owned_by="voyage", 
                context_window=4000,
                type="reranker"
            ),
        ]

    def __del__(self):
        """Clean up HTTP clients on destruction."""
        try:
            if hasattr(self, 'client') and not self.client.is_closed:
                self.client.close()
        except Exception:
            pass