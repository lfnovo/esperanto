"""Jina reranker provider implementation."""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from esperanto.common_types import Model
from esperanto.common_types.reranker import RerankResponse, RerankResult
from esperanto.common_types.response import Usage

from .base import RerankerModel


@dataclass
class JinaRerankerModel(RerankerModel):
    """Jina reranker provider with HTTP API integration."""

    def __post_init__(self):
        """Initialize Jina reranker after dataclass initialization."""
        super().__post_init__()
        
        # Authentication
        self.api_key = self.api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Jina API key not found. Set JINA_API_KEY environment variable or pass api_key parameter."
            )
        
        # API configuration
        self.base_url = self.base_url or "https://api.jina.ai/v1"
        
        # Get timeout from config or use default
        timeout = self._config.get("timeout", 30.0)
        
        # HTTP client setup
        self.client = httpx.Client(timeout=timeout)
        self.async_client = httpx.AsyncClient(timeout=timeout)

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers for Jina API."""
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
        """Build request payload for Jina rerank API.
        
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
            "top_n": top_k,  # Jina uses top_n instead of top_k
            "return_documents": True  # Always return documents for consistent interface
        }
        
        return payload

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle error responses from Jina API.
        
        Args:
            response: HTTP response object.
            
        Raises:
            RuntimeError: With details from the error response.
        """
        # Debug logging
        
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", "Unknown error")
            error_type = error_data.get("error", {}).get("type", "Unknown")
            raise RuntimeError(f"Jina API error ({error_type}): {error_message}")
        except (KeyError, ValueError):
            raise RuntimeError(f"Jina API error: {response.status_code} - {response.text}")

    def _parse_response(self, response_data: Dict[str, Any], documents: List[str]) -> RerankResponse:
        """Parse Jina API response into standardized format.
        
        Args:
            response_data: Raw response from Jina API.
            documents: Original documents list for fallback.
            
        Returns:
            Standardized RerankResponse.
        """
        results = []
        raw_results = response_data.get("results", [])
        
        # Extract raw scores for normalization
        raw_scores = [result.get("relevance_score", 0.0) for result in raw_results]
        normalized_scores = self._normalize_scores(raw_scores)
        
        for i, result in enumerate(raw_results):
            index = result.get("index", i)
            document = result.get("document")
            
            # Handle Jina's document format with robust validation
            if isinstance(document, dict):
                # Try common text fields in order of preference
                if "text" in document:
                    document = document["text"]
                elif "content" in document:
                    document = document["content"]
                elif "body" in document:
                    document = document["body"]
                else:
                    # If dict doesn't have expected text fields, stringify it
                    document = str(document)
            elif document is None and index < len(documents):
                # Use original document if not returned in response
                document = documents[index]
            elif document is None:
                document = ""
            
            # Ensure document is a string
            if not isinstance(document, str):
                document = str(document)
            
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
        """Rerank documents using Jina API.
        
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
            raise RuntimeError("Request to Jina API timed out")
        except httpx.RequestError as e:
            raise RuntimeError(f"Network error calling Jina API: {str(e)}")

    async def arerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: Optional[int] = None, 
        **kwargs
    ) -> RerankResponse:
        """Async rerank documents using Jina API.
        
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
            raise RuntimeError("Request to Jina API timed out")
        except httpx.RequestError as e:
            raise RuntimeError(f"Network error calling Jina API: {str(e)}")

    def to_langchain(self):
        """Convert to LangChain-compatible reranker."""
        try:
            from langchain.schema import Document
            from langchain_core.callbacks.manager import Callbacks
        except ImportError:
            raise ImportError(
                "LangChain not installed. Install with: pip install langchain"
            )
        
        class LangChainJinaReranker:
            def __init__(self, jina_reranker):
                self.jina_reranker = jina_reranker
            
            def compress_documents(
                self, 
                documents: List[Document], 
                query: str, 
                callbacks: Optional[Callbacks] = None
            ) -> List[Document]:
                """Compress documents using Jina reranker."""
                # Extract text content from documents
                texts = [doc.page_content for doc in documents]
                
                # Rerank using Jina
                rerank_response = self.jina_reranker.rerank(query, texts)
                
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
        
        return LangChainJinaReranker(self)

    def _get_default_model(self) -> str:
        """Get default Jina model."""
        return "jina-reranker-v2-base-multilingual"

    @property
    def provider(self) -> str:
        """Provider name."""
        return "jina"

    @property
    def models(self) -> List[Model]:
        """Available Jina reranker models."""
        return [
            Model(
                id="jina-reranker-v2-base-multilingual",
                owned_by="jina",
                context_window=1024,
                type="reranker"
            ),
            Model(
                id="jina-reranker-v1-base-en",
                owned_by="jina", 
                context_window=512,
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