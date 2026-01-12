"""DashScope reranker provider implementation."""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from ...common_types import Model
from ...common_types.reranker import RerankResponse, RerankResult
from ...common_types.response import Usage
from .base import RerankerModel


@dataclass
class DashScopeRerankerModel(RerankerModel):
    """DashScope reranker provider with HTTP API integration."""

    # Name of DashScope reranker models that support including documents in responses.
    __DOC_RETURN_AVAIL = ["gte-rerank-v2"]
    __INSTRUCT_AVAIL = ["qwen3-rerank"]

    def __post_init__(self):
        """Initialize Jina reranker after dataclass initialization."""
        super().__post_init__()

        # Authentication
        self.api_key = self.api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DashScope API key not found. Set DASHSCOPE_API_KEY environment variable or pass api_key parameter."
            )

        # API configuration
        self.base_url = self.base_url or "https://dashscope.aliyuncs.com/api/v1"

        # Initialize HTTP clients with configurable timeout
        self._create_http_clients()

        # Model specific parameter cases
        _model_name = self.get_model_name()
        # return_documents
        self.__doc_return_avail: bool = _model_name in self.__DOC_RETURN_AVAIL
        # instruct
        self.__instruct_avail: bool = _model_name in self.__INSTRUCT_AVAIL


    def _get_headers(self) -> Dict[str, str]:
        """Get request headers for DashScope API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _get_default_model(self) -> str:
        """Get default DashScope model."""
        return "qwen3-rerank"

    @property
    def provider(self) -> str:
        return "dashscope"

    def _get_models(self) -> List[Model]:
        """Available DashScope reranker models"""
        return [
            Model(
                id="qwen3-rerank",
                owned_by="dashscope",
                context_window=4000
            ),
            Model(
                id="gte-rerank-v2",
                owned_by="dashscope",
                context_window=4000
            ),
        ]

    def _build_request_payload(
        self,
        query: str,
        documents: List[str],
        top_k: int,
        instruct: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Build request payload for DashScope rerank API.

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
            "top_n": top_k  # DashScope uses top_n instead of top_k
        }

        # Handle other model-specific parameters
        # return_documents
        # NOTE: for consistency and saving network cost, disable
        payload.update({"return_documents": False})

        # instruct
        if instruct is not None and self.__instruct_avail:
            payload.update({"instruct": instruct})

        return payload

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle error responses from DashScope API.

        Args:
            responses: HTTP response object.

        Raises:
            RuntimeError: With details from the error response.
        """
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", "Unknown error")
            error_type = error_data.get("error", {}).get("code", "Unknown")
            raise RuntimeError(f"DashScope API error ({error_type}): {error_msg}")
        except (KeyError, ValueError):
            raise RuntimeError(
                f"DashScope API error: {response.status_code} - {response.text}"
            )

    def _parse_response(self, response_data: Dict[str, Any], documents: List[str]) -> RerankResponse:
        """Parse DashScope API response into standardized format.

        Args:
            response_data: Raw response from DashScope API.
            documents: Original documents list for fallback.

        Returns:
            Standardized RerankResponse.
        """
        raw_results = response_data.get("output", {}).get("results", [])
        rerank_results: List[RerankResult] = []

        # Extract raw scores for normalization
        raw_scores = [res.get("relevance_score", 0.0) for res in raw_results]
        normalized_scores = self._normalize_scores(raw_scores)

        for result, norm_score in zip(raw_results, normalized_scores):
            doc_idx = result.get("index")
            if doc_idx is None or doc_idx < -0 or doc_idx >= len(documents):
                cur_doc = ""
            else:
                cur_doc = documents[doc_idx]

            rerank_results.append(RerankResult(
                index=doc_idx,
                document=cur_doc,
                relevance_score=norm_score
            ))

        # Create usage
        if "usage" in response_data:
            usage_data = response_data["usage"]
            usage = Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=usage_data.get("total_tokens", 0)
            )  # DashScope API only provide 'total_tokens'
        else:
            usage = None

        return RerankResponse(
            results=rerank_results,
            model=self.get_model_name(),
            usage=usage
        )

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        instruct: Optional[str] = None,
        **kwargs
    ) -> RerankResponse:
        """Rerank documents using DashScope API.

        Args:
            query: The search query to rank documents against.
            documents: List of documents to rerank.
            top_k: Maximum number of results to return.
            instruct: System instruct passed to reranker.

        Returns:
            RerankResponse with ranked results.
        """
        # Validate inputs
        query, documents, top_k = self._validate_inputs(query, documents, top_k)

        # Build request
        payload = self._build_request_payload(query, documents, top_k, instruct=instruct)

        try:
            response = self.client.post(
                f"{self.base_url}/services/rerank/text-rerank/text-rerank",
                json=payload,
                headers=self._get_headers()
            )

            if response.status_code != 200:
                self._handle_error(response)

            response_data = response.json()
            return self._parse_response(response_data, documents)

        except httpx.TimeoutException:
            raise RuntimeError("Request to DashScope API timed out.")
        except httpx.RequestError as e:
            raise RuntimeError(f"Network error calling DashScope API: {str(e)}")

    async def arerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        instruct: Optional[str] = None,
        **kwargs
    ) -> RerankResponse:
        """Async rerank documents using DashScope API.

        Args:
            query: The search query to rank documents against.
            documents: List of documents to rerank.
            top_k: Maximum number of results to return.
            instruct: System instruct passed to reranker.

        Returns:
            RerankResponse with ranked results.
        """
        # Validate inputs
        query, documents, top_k = self._validate_inputs(query, documents, top_k)

        # Build request
        payload = self._build_request_payload(query, documents, top_k, instruct=instruct)

        try:
            response = self.async_client.post(
                f"{self.base_url}/services/rerank/text-rerank/text-rerank",
                json=payload,
                headers=self._get_headers()
            )

            if response.status_code != 200:
                self._handle_error(response)

            response_data = response.json()
            return self._parse_response(response_data, documents)

        except httpx.TimeoutException:
            raise RuntimeError("Request to DashScope API timed out.")
        except httpx.RequestError as e:
            raise RuntimeError(f"Network error calling DashScope API: {str(e)}")

    def to_langchain(self):
        """Convert to LangChain-compatible reranker."""
        try:
            from langchain_core.documents import Document
            from langchain_core.callbacks.base import Callbacks
        except ImportError:
            raise ImportError(
                "LangChain not installed. Install with: pip install langchain."
            )


        class LangChainDashScopeReranker:
            def __init__(self, dashscope_reranker: "DashScopeRerankerModel"):
                self.reranker = dashscope_reranker

            def compress_documents(
                self,
                documents: List[Document],
                query: str,
                callbacks: Optional[Callbacks] = None
            ) -> List[Document]:
                """Compress documents using DashScope reranker."""
                texts = [doc.page_content for doc in documents]

                # Rerank using DashScope
                rerank_response = self.reranker.rerank(query, texts)

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


        return LangChainDashScopeReranker(self)

    def __del__(self):
        """Clean up HTTP clients on destruction."""
        try:
            if not self.client.is_closed:
                self.client.close()
        except Exception:
            pass